"""Evaluate H2PPO protocol throughput and admitted-to-confirmed latency."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import blockchain.params as p
from model_io import make_agent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("h2ppo",), default="h2ppo")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrival-rate", type=int, required=True)
    parser.add_argument(
        "--arrival-rate-unit", choices=("per_slot", "per_second"), default="per_slot",
        help=("Interpret --arrival-rate as transactions per simulated slot or as "
              "transactions per second. per_second multiplies it by --slot-seconds "
              "before sampling the slot batch."),
    )
    parser.add_argument(
        "--slot-seconds", type=float, default=1.0,
        help="Simulated duration of one arrival/decision/consensus slot.",
    )
    parser.add_argument(
        "--block-generation-seconds", type=float, default=8.0,
        help="Block generation interval used by the resource formula.",
    )
    parser.add_argument("--arrival-process", choices=("checkpoint","fixed","poisson"),
                        default="checkpoint")
    parser.add_argument(
        "--arrival-order",
        choices=("csv", "admitted_parent_ready", "confirmed_parent_ready",
                 "hybrid_parent_ready"),
        default="csv",
        help=("CSV order; a bounded topological replay that allows children into the "
              "measured queue after their parents are admitted; or a stricter replay "
              "that waits for parent confirmation."),
    )
    parser.add_argument("--arrival-reorder-scan-multiple", type=int, default=8)
    parser.add_argument(
        "--minimum-admission-fraction", type=float, default=0.95,
        help=("For hybrid_parent_ready, minimum fraction of the nominal slot batch "
              "admitted after confirmed-ready transactions are exhausted."),
    )
    parser.add_argument(
        "--max-idle-rounds", type=int, default=50,
        help="Abort with stalled-state.json after this many rounds with no arrival or confirmation.",
    )
    parser.add_argument("--transactions", type=int, default=1_000_000)
    parser.add_argument(
        "--transaction-window", type=int, default=50_000,
        help="Independent transaction-count window for cross-shard-rate and load-CV output.",
    )
    parser.add_argument("--placement-chunk-size", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--fast-exit-after-write", action="store_true",
        help=("After all result files and logs are closed, bypass Python interpreter "
              "teardown of the multi-million-transaction object graph."),
    )
    parser.add_argument("--input", nargs="+", default=None)
    parser.add_argument("--cpu-budget", type=float, default=36.0)
    parser.add_argument("--bandwidth-budget", type=float, default=40.0)
    parser.add_argument(
        "--override-checkpoint-budget", action="store_true",
        help=("Use --cpu-budget and --bandwidth-budget for a learned policy instead "
              "of the resource budget stored in its checkpoint."),
    )
    parser.add_argument("--cpu-weight", type=float, default=0.18)
    parser.add_argument("--bandwidth-weight", type=float, default=0.10)
    parser.add_argument("--valid-transaction-ratio", type=float, default=1.0)
    parser.add_argument("--critical-capacity-tps", type=float, default=8000.0)
    parser.add_argument("--cpu-tps-per-core", type=float, default=312.5)
    parser.add_argument("--fixed-block-size", type=int, default=6000)
    return parser.parse_args()


def encode(value):
    return json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode(value) for key, value in row.items()})


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = arguments()
    if min(args.arrival_rate, args.transactions, args.transaction_window,
           args.placement_chunk_size,
           args.torch_threads, args.slot_seconds,
           args.block_generation_seconds, args.arrival_reorder_scan_multiple) <= 0:
        raise ValueError("All count arguments must be positive")
    if args.max_idle_rounds < 1:
        raise ValueError("--max-idle-rounds must be positive")
    if not 0 < args.minimum_admission_fraction <= 1:
        raise ValueError("--minimum-admission-fraction must lie in (0,1]")
    if args.arrival_rate_unit == "per_second":
        arrivals_per_slot_target = int(round(args.arrival_rate * args.slot_seconds))
        arrival_rate_target_tps = float(args.arrival_rate)
    else:
        arrivals_per_slot_target = int(args.arrival_rate)
        arrival_rate_target_tps = args.arrival_rate / args.slot_seconds
    if arrivals_per_slot_target < 1:
        raise ValueError("Arrival-rate settings produce fewer than one transaction per slot")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for H2PPO")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = ((saved.get("info") or {}).get("protocol") or {}) if saved else {}
    inputs = args.input or protocol.get("input_files") or [p.FileInput1, p.FileInput2, p.FileInput3]
    missing = [str(path) for path in inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    from msschain_env import Supervisor

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    shards = int(saved["num_shards"]) if saved else args.shards
    if args.shards != shards:
        raise ValueError("--shards differs from the checkpoint shard count")
    p.ShardNum = shards
    p.BlockSize = args.fixed_block_size
    cli_budget = (float(args.cpu_budget), float(args.bandwidth_budget))
    checkpoint_resource_max = tuple(saved.get("resource_max", cli_budget)) if saved else None
    checkpoint_network_budget = saved.get("network_resource_budget") if saved else None
    if saved and not args.override_checkpoint_budget:
        resource_max = checkpoint_resource_max
        network_budget = checkpoint_network_budget or resource_max
        resource_budget_source = "checkpoint"
    else:
        resource_max = cli_budget
        network_budget = cli_budget
        resource_budget_source = "cli_override" if saved else "command_line"
    resource_weights = tuple(protocol.get(
        "resource_weights_cpu_bandwidth", (args.cpu_weight, args.bandwidth_weight)))
    resource_model = protocol.get("resource_model", "paper_eq10")
    valid_transaction_ratio = float(protocol.get("valid_transaction_ratio", args.valid_transaction_ratio))
    cpu_tps_per_core = float(protocol.get("cpu_candidate_tps_per_core", args.cpu_tps_per_core))
    resource_units = protocol.get("resource_units", ["CPU cores", "kb/s"])
    block_range = protocol.get(
        "block_size_action_range_transactions",
        [saved["bmin"], saved["bmax"]] if saved else [2000, args.fixed_block_size]
    )
    arrival_process = (protocol.get("arrival_process", "fixed")
                       if args.arrival_process == "checkpoint" else args.arrival_process)
    initial = (
        np.asarray(network_budget, dtype=float) / shards
        if network_budget is not None
        else np.asarray(resource_max, dtype=float)
    ).tolist()
    service_capacity_scale = 1.0
    source_transaction_limit = (
        args.transactions if args.arrival_order == "csv" else None
    )
    environment = Supervisor(
        file_paths=inputs,
        output_dir=args.output,
        batch_size=arrivals_per_slot_target,
        window_size=int(protocol.get("window_size", 50_000)),
        transaction_limit=source_transaction_limit,
        retain_blocks=False,
        track_latency=True,
        enhanced_observation=True,
        stability_observation=(saved is not None and int(saved["num_states"]) == 12 * shards),
        resource_control=True,
        resource_max=resource_max,
        initial_resources=initial,
        network_resource_budget=network_budget,
        resource_weights=resource_weights,
        resource_model=resource_model,
        valid_transaction_ratio=valid_transaction_ratio,
        cpu_tps_per_core=cpu_tps_per_core,
        slot_duration=args.slot_seconds,
        block_interval=args.block_generation_seconds,
        block_min=int(block_range[0]),
        block_max=int(block_range[1]),
        arrival_process=arrival_process,
        arrival_order=args.arrival_order,
        arrival_reorder_scan_multiple=args.arrival_reorder_scan_multiple,
        minimum_admission_fraction=args.minimum_admission_fraction,
        arrival_candidate_storage=(
            "compressed_memory" if args.arrival_order in (
                "confirmed_parent_ready", "hybrid_parent_ready")
            else "memory"),
        alpha=float(protocol.get("reward_locality_weight", 0.5)),
        decay_seconds=float(protocol.get("dependency_decay_seconds", 86_400.0)),
        zeta=float(protocol.get("balance_reward_relative_deviation_sensitivity", 0.1)),
        load_feature_overload_power=protocol.get("load_feature_overload_power"),
        address_affinity=float(protocol.get("address_affinity_weight", 0.0)) > 0,
        cross_penalty=float(protocol.get("placement_cross_penalty", 0.0)),
        service_capacity_scale=service_capacity_scale,
        seed=int(protocol.get("seed", 0)),
    )
    agent = make_agent(saved, resource_max, network_budget)
    agent.actor.eval()
    agent.critic.eval()
    state = environment.reset()
    environment.begin_metric_reporting(args.transaction_window)
    started = time.time()
    log_file = (args.output / "run.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        line = f"[{time.time() - started:9.1f}s] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")

    rows = []
    cumulative_delay_lower_bound = 0.0
    total_latency_sum = 0.0
    total_latency_samples = 0
    max_internal_conservation_error = 0
    idle_rounds = 0
    log(
        f"start method={args.method}, label={args.label or args.method}, "
        f"mean total-system arrival={arrivals_per_slot_target:,} Tx/slot "
        f"({arrival_rate_target_tps:,.3f} Tx/s), "
        f"process={arrival_process}, transactions={args.transactions:,}, "
        "event order=arrival->placement->model block size->consensus"
    )

    while environment.accepted < args.transactions:
        remaining = args.transactions - environment.accepted
        if len(environment.arriveTxArrive) > remaining:
            state = environment.limit_current_arrivals(remaining)
        accepted_before = environment.accepted
        confirmed_before = environment.confirmed
        cross_before = environment.cross_count
        outstanding_before = accepted_before - confirmed_before
        queue_before = environment.queue_lengths()
        arrivals_this_round = len(environment.arriveTxArrive)
        with torch.no_grad():
            state, _, done = agent.interact_batched(
                environment, test=True, states=state, chunk_size=args.placement_chunk_size
            )
        if done and environment.accepted < args.transactions:
            raise RuntimeError(f"Dataset exhausted at {environment.accepted:,} transactions")

        # The environment records latency in slots. Convert to seconds before
        # aggregation so a 10-second slot is not reported as one second.
        step_latencies = [
            float(value) * args.slot_seconds for value in environment._latencies
        ]
        latency_sum = float(sum(step_latencies))
        latency_samples = len(step_latencies)
        total_latency_sum += latency_sum
        total_latency_samples += latency_samples
        cumulative_delay_lower_bound += (
            outstanding_before + arrivals_this_round
        ) * args.slot_seconds
        environment._latencies.clear()
        environment._confirmation_log.clear()

        protocol_queues = environment.queue_lengths()
        unconfirmed_by_shard = environment.partition_sizes - environment.confirmed_by_shard
        if (unconfirmed_by_shard < 0).any() or int(unconfirmed_by_shard.sum()) != environment.accepted - environment.confirmed:
            raise AssertionError("Per-shard original-transaction queue conservation failed")
        internal_error = int(np.abs(
            environment.q - (queue_before + environment.arrive - environment.b)
        ).sum())
        max_internal_conservation_error = max(max_internal_conservation_error, internal_error)
        block_sizes = np.asarray(environment.metrics["block_sizes"], dtype=int)
        resource_allocations = np.asarray(
            environment.metrics["resource_allocations"], dtype=float
        )
        service_capacities = np.asarray([
            min(shard.blocksize, shard.compute_resource(shard.resources, types=0))
            for shard in environment.shards
        ], dtype=float)
        confirmed_this_round = environment.confirmed - confirmed_before
        row = {
            "arrivals_per_slot_target": arrivals_per_slot_target,
            "arrival_rate_target_tps": arrival_rate_target_tps,
            "slot_seconds": args.slot_seconds,
            "consensus_round": environment.clock,
            "arrived_transactions": arrivals_this_round,
            "workload_generator_pending_dependencies": environment.arrival_candidate_pool_size,
            "cumulative_accepted": environment.accepted,
            "cross_shard_transactions": environment.cross_count - cross_before,
            "cross_shard_rate_in_consensus_arrivals": (
                (environment.cross_count - cross_before) / max(1,arrivals_this_round)
            ),
            "cumulative_cross_shard_rate_including_coinbase": (
                environment.cross_count / max(1,environment.accepted)
            ),
            "confirmed_transactions": confirmed_this_round,
            "confirmed_transactions_per_slot": confirmed_this_round,
            "throughput_tps": confirmed_this_round / args.slot_seconds,
            "completed_mean_latency_seconds": (
                latency_sum / latency_samples if latency_samples else None
            ),
            "completed_samples": latency_samples,
            "unconfirmed_original_transactions_total": int(unconfirmed_by_shard.sum()),
            "unconfirmed_original_transactions_mean_per_shard": float(unconfirmed_by_shard.mean()),
            "unconfirmed_original_transactions_min_shard": int(unconfirmed_by_shard.min()),
            "unconfirmed_original_transactions_max_shard": int(unconfirmed_by_shard.max()),
            "unconfirmed_original_transactions_by_shard": unconfirmed_by_shard.astype(int).tolist(),
            "protocol_queue_items_total": int(protocol_queues.sum()),
            "protocol_queue_items_mean_per_shard": float(protocol_queues.mean()),
            "protocol_queue_items_max_shard": int(protocol_queues.max()),
            "protocol_queue_items_by_shard": protocol_queues.astype(int).tolist(),
            "model_block_size_mean": float(block_sizes.mean()),
            "model_block_size_min": int(block_sizes.min()),
            "model_block_size_max": int(block_sizes.max()),
            "model_block_sizes_by_shard": block_sizes.tolist(),
            "resource_cpu_total": float(resource_allocations[:, 0].sum()),
            "resource_bandwidth_total": float(resource_allocations[:, 1].sum()),
            "resource_cpu_by_shard": resource_allocations[:, 0].tolist(),
            "resource_bandwidth_by_shard": resource_allocations[:, 1].tolist(),
            "resource_formula_service_capacity_total_tps": float(service_capacities.sum()),
            "resource_formula_service_capacity_mean_per_shard_tps": float(
                service_capacities.mean()
            ),
            "resource_formula_service_capacity_min_shard_tps": float(
                service_capacities.min()
            ),
            "resource_formula_service_capacity_max_shard_tps": float(
                service_capacities.max()
            ),
            "resource_formula_service_capacity_by_shard_tps": service_capacities.tolist(),
            "all_admitted_delay_lower_bound_seconds": (
                cumulative_delay_lower_bound / environment.accepted
            ),
            "original_transaction_conservation_error": int(
                environment.accepted - environment.confirmed - unconfirmed_by_shard.sum()
            ),
            "internal_queue_conservation_error_l1": internal_error,
        }
        rows.append(row)
        idle_rounds = (idle_rounds + 1
                       if arrivals_this_round == 0 and confirmed_this_round == 0
                       else 0)
        if idle_rounds >= args.max_idle_rounds:
            pool_items = []
            original_items = []
            for sid, shard in enumerate(environment.shards):
                for tx in shard.txPools.TxsQueue.values():
                    pool_items.append({
                        "shard": sid, "hash": tx.Hash, "original_hash": tx.OrigTxHash,
                        "size_bytes": tx.Size, "inputs": tx.TxInCount,
                        "outputs": tx.TxOutCount, "has_outputs": tx.TxOuts is not None,
                        "tx_consensus": bool(tx.tx_consensus),
                    })
                for tx in shard.crossTxPool.OriginalTxsQueue.values():
                    original_items.append({
                        "shard": sid, "hash": tx.Hash, "original_hash": tx.OrigTxHash,
                        "size_bytes": tx.Size, "inputs": tx.TxInCount,
                        "outputs": tx.TxOutCount,
                    })
            candidates = []
            for tx in environment.arrival_candidate_samples(20):
                candidates.append({
                    "hash": tx.Hash, "size_bytes": tx.Size,
                    "unconfirmed_parents": [
                        parent for parent in environment._parent_hashes(tx)
                        if not environment.TxIsProcessed.get(parent,False)
                    ],
                })
            snapshot = {
                "idle_rounds": idle_rounds,
                "accepted": environment.accepted,
                "confirmed": environment.confirmed,
                "protocol_queue_items": pool_items,
                "cross_originals": original_items,
                "workload_generator_candidates": environment.arrival_candidate_pool_size,
                "candidate_sample": candidates,
            }
            dump(args.output / "stalled-state.json", snapshot)
            raise RuntimeError(
                f"No arrival or confirmation for {idle_rounds} rounds; "
                f"diagnostics written to {args.output / 'stalled-state.json'}")
        # Every round is written to CSV. Keep the terminal readable while
        # still exposing regular progress and the final partial round.
        if environment.clock <= 5 or environment.clock % 10 == 0 or environment.accepted == args.transactions:
            log(
                f"round={environment.clock:04d}, arrived={arrivals_this_round:,}, "
                f"confirmed/slot={confirmed_this_round:,}, "
                f"throughput={row['throughput_tps']:.3f} Tx/s, "
                f"latency={row['completed_mean_latency_seconds']}, "
                f"unconfirmed={row['unconfirmed_original_transactions_total']:,}, "
                f"queue/shard={row['unconfirmed_original_transactions_mean_per_shard']:.3f}, "
                f"block(mean/min/max)={row['model_block_size_mean']:.1f}/"
                f"{row['model_block_size_min']}/{row['model_block_size_max']}"
            )

    throughput = np.asarray([row["throughput_tps"] for row in rows], dtype=float)
    round_latency = np.asarray([
        np.nan if row["completed_mean_latency_seconds"] is None
        else row["completed_mean_latency_seconds"] for row in rows
    ])
    original_queue = np.asarray([
        row["unconfirmed_original_transactions_mean_per_shard"] for row in rows
    ], dtype=float)
    protocol_queue = np.asarray([row["protocol_queue_items_mean_per_shard"] for row in rows])
    block_size = np.asarray([row["model_block_size_mean"] for row in rows])
    block_vectors = np.asarray([row["model_block_sizes_by_shard"] for row in rows])
    cpu_total = np.asarray([row["resource_cpu_total"] for row in rows])
    bandwidth_total = np.asarray([row["resource_bandwidth_total"] for row in rows])
    service_capacity = np.asarray([
        row["resource_formula_service_capacity_total_tps"] for row in rows
    ])
    unconfirmed_vectors = np.asarray([
        row["unconfirmed_original_transactions_by_shard"] for row in rows
    ])
    summary = {
        "method": args.method,
        "label": args.label or args.method,
        "arrivals_per_slot_target": arrivals_per_slot_target,
        "arrival_rate_target_tps": arrival_rate_target_tps,
        "arrival_rate_input_value": args.arrival_rate,
        "arrival_rate_input_unit": args.arrival_rate_unit,
        "slot_seconds": args.slot_seconds,
        "block_generation_seconds": args.block_generation_seconds,
        "arrival_process": arrival_process,
        "arrival_order": args.arrival_order,
        "arrival_reorder_scan_multiple": args.arrival_reorder_scan_multiple,
        "minimum_admission_fraction": args.minimum_admission_fraction,
        "arrival_scope": "total system across all shards",
        "transactions": args.transactions,
        "consensus_rounds": len(rows),
        "actual_mean_arrivals_per_slot": args.transactions / len(rows),
        "actual_mean_arrival_rate_tps": (
            args.transactions / (len(rows) * args.slot_seconds)
        ),
        "mean_consensus_round_throughput_tps": float(throughput.mean()),
        "mean_consensus_round_completed_latency_seconds": float(np.nanmean(round_latency)),
        "overall_completed_latency_seconds_sample_weighted": (
            total_latency_sum / total_latency_samples if total_latency_samples else None
        ),
        "mean_consensus_round_unconfirmed_original_transactions_total": float(
            original_queue.mean() * shards
        ),
        "mean_consensus_round_unconfirmed_original_transactions_per_shard": float(
            original_queue.mean()
        ),
        "mean_unconfirmed_original_transactions_by_shard": unconfirmed_vectors.mean(0).tolist(),
        "final_unconfirmed_original_transactions": rows[-1]["unconfirmed_original_transactions_total"],
        "overall_cross_shard_rate_including_coinbase": (
            environment.cross_count / max(1,environment.accepted)
        ),
        "transaction_window_size": args.transaction_window,
        "transaction_window_count": len(environment.metric_series),
        "mean_transaction_window_cross_shard_rate": (
            float(np.mean([row["interval_cross_shard_rate"]
                           for row in environment.metric_series]))
            if environment.metric_series else None
        ),
        "mean_transaction_window_protocol_load_cv": (
            float(np.mean([row["interval_protocol_load_cv"]
                           for row in environment.metric_series]))
            if environment.metric_series else None
        ),
        "mean_consensus_round_protocol_queue_items_total": float(protocol_queue.mean() * shards),
        "mean_consensus_round_protocol_queue_items_per_shard": float(protocol_queue.mean()),
        "mean_model_block_size": float(block_size.mean()),
        "mean_model_block_size_by_shard": block_vectors.mean(0).tolist(),
        "model_block_size_min_observed": int(block_vectors.min()),
        "model_block_size_max_observed": int(block_vectors.max()),
        "mean_allocated_cpu_total": float(cpu_total.mean()),
        "mean_allocated_bandwidth_total": float(bandwidth_total.mean()),
        "mean_resource_formula_service_capacity_total_tps": float(
            service_capacity.mean()
        ),
        "min_resource_formula_service_capacity_total_tps": float(
            service_capacity.min()
        ),
        "max_resource_formula_service_capacity_total_tps": float(
            service_capacity.max()
        ),
        "all_admitted_delay_lower_bound_seconds_at_end": rows[-1]["all_admitted_delay_lower_bound_seconds"],
        "rejected_or_dropped_transactions": 0,
        "arrival_reordered_transactions": environment.arrival_reordered_transactions,
        "max_workload_generator_pending_dependencies": environment.max_arrival_candidate_pool,
        "source_transactions_scanned": environment.source_transactions_scanned,
        "max_original_transaction_conservation_error": max(
            abs(row["original_transaction_conservation_error"]) for row in rows
        ),
        "max_internal_queue_conservation_error_l1": max_internal_conservation_error,
        "wall_seconds": time.time() - started,
    }
    run_protocol = {
        "method": args.method,
        "label": args.label or args.method,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "evaluated_arrival_range_after_reordering": [0, args.transactions],
        "source_transaction_limit": source_transaction_limit,
        "event_order_each_second": [
            "admit transactions",
            f"{args.label or args.method} placement",
            "model block-size/resource output",
            "consensus/service"
        ],
        "arrival_process": arrival_process,
        "arrival_order": args.arrival_order,
        "minimum_admission_fraction": args.minimum_admission_fraction,
        "arrival_order_definition": (
            "Admit only transactions whose data-set parents have completed confirmation; "
            "retain unresolved transactions in the workload-generator candidate pool and "
            "backfill with later ready transactions without dropping any transaction."
            if args.arrival_order == "confirmed_parent_ready" else
            "Prefer confirmed-parent transactions, then maintain the configured admission "
            "floor with a topological prefix whose parents are already admitted; unresolved "
            "children remain in the measured W-Shard dependency queue until confirmation."
            if args.arrival_order == "hybrid_parent_ready" else
            "Replay valid transactions in CSV order."
        ),
        "arrival_rate_scope": "mean total-system arrivals per slot across all shards",
        "arrivals_per_slot_target": arrivals_per_slot_target,
        "arrival_rate_target_tps": arrival_rate_target_tps,
        "arrival_rate_input_unit": args.arrival_rate_unit,
        "slot_seconds": args.slot_seconds,
        "block_generation_seconds": args.block_generation_seconds,
        "report_frequency": "after every consensus round",
        "transaction_window_reporting": (
            f"independent non-overlapping windows of {args.transaction_window} admitted transactions"
        ),
        "transaction_window_cross_shard_rate_definition": (
            "cross-shard original transactions divided by all original transactions, including coinbase, "
            "within the independent transaction window"
        ),
        "transaction_window_cv_definition": (
            "population standard deviation divided by mean of per-shard protocol workload within the "
            "independent transaction window; workload counts internal originals, cross-shard originals, "
            "and generated cross-shard subtransactions"
        ),
        "throughput_definition": (
            "original transactions confirmed in the slot divided by slot_seconds"
        ),
        "completed_latency_definition": (
            "admission-to-confirmation slots multiplied by slot_seconds"
        ),
        "primary_shard_queue_definition": (
            "admitted originals assigned to each shard minus confirmed originals assigned to that shard"
        ),
        "protocol_queue_definition": "ordinary tx-pool plus cross-shard protocol queue items",
        "block_size_definition": f"{shards}-shard model output vector",
        "resource_budget_cpu_bandwidth": list(network_budget) if network_budget is not None else None,
        "resource_budget_source": resource_budget_source,
        "checkpoint_resource_budget_cpu_bandwidth": (
            list(checkpoint_network_budget or checkpoint_resource_max)
            if saved else None
        ),
        "resource_weights_cpu_bandwidth": list(resource_weights),
        "resource_model": resource_model,
        "valid_transaction_ratio": valid_transaction_ratio,
        "cpu_candidate_tps_per_core": cpu_tps_per_core,
        "scalar_paper_queue_capacity_scale": service_capacity_scale,
        "full_protocol_capacity_scale": 1.0,
        "full_protocol_capacity_note": (
            "The scalar paper-queue conversion is not applied a second time: "
            "cross-shard requests, responses, and final transactions already "
            "consume explicit block bytes and Eq. (10) resource capacity."
        ),
        "critical_capacity_tps": args.critical_capacity_tps,
        "resource_units": resource_units,
        "block_size_action_range_transactions": block_range,
        "resource_formula_service_capacity_definition": ({
            "huang_eq2": (
                "sum over shards of min(block_size, sqrt(cpu_weight*cpu_cores)+"
                "sqrt(bandwidth_weight*bandwidth_kbps)); one value per timeslot"
            ),
            "physical_bottleneck": (
                "sum over shards of valid_ratio*min(block_size/block_interval, "
                "cpu_cores*cpu_candidate_tps_per_core, bandwidth_kbps*1000/"
                "(8*average_transaction_bytes))"
            ),
            "paper_eq10": (
                "sum over shards of min(block_size, block_size*valid_ratio/block_interval*"
                "(sqrt(cpu_weight*cpu)+sqrt(bandwidth_weight*bandwidth)))"
            ),
        }[resource_model]),
        "final_aggregation": "arithmetic mean across all consensus-round observations",
        "external_inference_constraints": None,
        "drain_after_evaluation": False,
        "input_files": [str(Path(path).resolve()) for path in inputs],
    }
    write_csv(args.output / "consensus-rounds.csv", rows)
    if environment.metric_series:
        write_csv(args.output / "transaction-windows.csv", environment.metric_series)
    dump(args.output / "summary.json", summary)
    dump(args.output / "protocol.json", run_protocol)
    log(
        f"complete: rounds={len(rows)}, mean throughput={summary['mean_consensus_round_throughput_tps']:.6f}, "
        f"latency={summary['mean_consensus_round_completed_latency_seconds']:.6f}, "
        f"queue/shard={summary['mean_consensus_round_unconfirmed_original_transactions_per_shard']:.6f}, "
        f"block={summary['mean_model_block_size']:.6f}"
    )
    environment.close()
    log_file.close()
    if args.fast_exit_after_write:
        # All durable outputs are complete at this point.  Normal interpreter
        # teardown can spend minutes recursively releasing millions of Python
        # transaction objects; _exit avoids that non-measurement tail.
        os._exit(0)


if __name__ == "__main__":
    main()
