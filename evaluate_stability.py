"""Evaluate one trained H2PPO policy under a total-system arrival burst.

This uses the same CSV-order, one-second ``paper_queue`` abstraction as the
placement evaluation.  It does not run baseline policies or reorder Bitcoin
transactions.  The separate throughput mode uses ``hybrid_parent_ready``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import blockchain.params as params
from model_io import make_agent
from msschain_env import Supervisor


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--input', nargs='+', required=True, type=Path)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--base-rate', type=int, default=2000)
    parser.add_argument('--burst-rate', type=int, default=8000)
    parser.add_argument('--pre-seconds', type=int, default=500)
    parser.add_argument('--burst-seconds', type=int, default=100)
    parser.add_argument('--total-seconds', type=int, default=1200)
    parser.add_argument('--cpu-budget', type=float, default=36.)
    parser.add_argument('--bandwidth-budget', type=float, default=64.)
    parser.add_argument('--critical-capacity-tps', type=float, default=8000.)
    parser.add_argument('--placement-chunk-size', type=int, default=1024)
    return parser.parse_args()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_at(second, args):
    if second < args.pre_seconds:
        return 'baseline_before', args.base_rate
    if second < args.pre_seconds + args.burst_seconds:
        return 'burst', args.burst_rate
    return 'recovery', args.base_rate


def main():
    args = arguments()
    counts = (args.shards, args.threads, args.base_rate, args.burst_rate,
              args.pre_seconds, args.burst_seconds, args.total_seconds,
              args.placement_chunk_size)
    if min(counts) <= 0 or args.pre_seconds + args.burst_seconds >= args.total_seconds:
        raise ValueError('All counts must be positive and the burst must end before the horizon')
    if min(args.cpu_budget, args.bandwidth_budget, args.critical_capacity_tps) <= 0:
        raise ValueError('Resource budgets and critical capacity must be positive')
    inputs = [path.resolve() for path in args.input]
    checkpoint = args.checkpoint.resolve()
    for path in [checkpoint, *inputs]:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if (output / 'summary.json').exists():
        raise FileExistsError(f'Completed output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    shards = int(saved['num_shards'])
    if shards != args.shards:
        raise ValueError('Checkpoint shard count differs from --shards')
    params.ShardNum = shards
    protocol = (saved.get('info') or {}).get('protocol') or {}
    budget = (float(args.cpu_budget), float(args.bandwidth_budget))
    weights = tuple(protocol.get('resource_weights_cpu_bandwidth', (0.18, 0.10)))
    if len(weights) != 2 or min(weights) <= 0:
        raise ValueError('Checkpoint has invalid Eq. (10) resource weights')
    block_max = int(saved['bmax'])
    raw_reference = shards * min(
        float(block_max),
        float(block_max) / 8.0 * sum(
            np.sqrt(weight * amount / shards)
            for weight, amount in zip(weights, budget)
        ),
    )
    scale = args.critical_capacity_tps / raw_reference
    environment = Supervisor(
        file_paths=inputs, batch_size=args.base_rate,
        window_size=int(protocol.get('window_size', 50_000)),
        retain_blocks=False, track_latency=True,
        enhanced_observation=True,
        stability_observation=int(saved['num_states']) == 12 * shards,
        resource_control=True, resource_max=budget,
        network_resource_budget=budget,
        initial_resources=(np.asarray(budget) / shards).tolist(),
        resource_weights=weights, resource_model='paper_eq10',
        valid_transaction_ratio=float(protocol.get('valid_transaction_ratio', 1.0)),
        cpu_tps_per_core=float(protocol.get('cpu_candidate_tps_per_core', 312.5)),
        block_min=int(saved['bmin']), block_max=block_max,
        arrival_process='poisson', arrival_order='csv',
        # Match training: 1-second decisions with the paper's 8-second
        # block-time coefficient in Eq. (10)'s service formula.
        slot_duration=1., block_interval=8.,
        alpha=float(protocol.get('reward_locality_weight', 0.5)),
        decay_seconds=float(protocol.get('dependency_decay_seconds', 31_536_000.)),
        zeta=float(protocol.get('balance_reward_relative_deviation_sensitivity', 8.)),
        load_feature_overload_power=protocol.get('load_feature_overload_power', 2.),
        address_affinity=float(protocol.get('address_affinity_weight', 0.)) > 0,
        cross_penalty=float(protocol.get('placement_cross_penalty', 80.)),
        seed=args.seed, queue_service_mode='paper_queue',
        service_capacity_scale=scale,
        peak_queue_drift_weight=float(protocol.get('peak_queue_drift_weight', 4.)),
    )
    agent = make_agent(saved, budget, budget)
    agent.actor.eval()
    agent.critic.eval()
    rows = []
    total_latency = 0.
    total_latency_samples = 0
    started = time.monotonic()
    state = environment.reset()
    current_rate = args.base_rate
    try:
        for second in range(args.total_seconds):
            phase, target_rate = phase_at(second, args)
            if target_rate != current_rate:
                state = environment.set_arrival_rate(target_rate, resample_current=True)
                current_rate = target_rate
            arrivals = len(environment.arriveTxArrive)
            accepted_before = environment.accepted
            confirmed_before = environment.confirmed
            cross_before = environment.cross_count
            work_before = environment.workload_totals.copy()
            cpu_before = np.asarray([shard.cpu_needs for shard in environment.shards], dtype=float)
            bw_before = np.asarray([shard.bw_needs for shard in environment.shards], dtype=float)
            with torch.no_grad():
                state, _, done = agent.interact_batched(
                    environment, test=True, states=state,
                    chunk_size=args.placement_chunk_size)
            if done and second + 1 < args.total_seconds:
                raise RuntimeError(f'Transaction input exhausted at t={second + 1}s')
            admitted = environment.accepted - accepted_before
            confirmed = environment.confirmed - confirmed_before
            if admitted != arrivals:
                raise AssertionError('Generated and admitted transaction counts differ')
            queue = environment.partition_sizes - environment.confirmed_by_shard
            outstanding = environment.accepted - environment.confirmed
            if outstanding != int(queue.sum()):
                raise AssertionError('Original transaction conservation failed')
            work = environment.workload_totals - work_before
            mean_work = float(work.mean())
            latencies = np.asarray(environment._latencies, dtype=float)
            total_latency += float(latencies.sum())
            total_latency_samples += int(latencies.size)
            allocations = np.asarray(environment.metrics['resource_allocations'], dtype=float)
            blocks = np.asarray(environment.metrics['block_sizes'], dtype=int)
            row = {
                'second': second + 1,
                'phase': phase,
                'target_arrival_rate_tps': target_rate,
                'actual_arrival_rate_tps': arrivals,
                'admitted_transactions': admitted,
                'processing_rate_tps': confirmed,
                'throughput_tps': confirmed,
                'completed_mean_latency_seconds': (float(latencies.mean()) if latencies.size else ''),
                'completed_latency_samples': int(latencies.size),
                'cross_shard_transactions': environment.cross_count - cross_before,
                'cross_shard_rate_including_coinbase':
                    (environment.cross_count - cross_before) / max(1, admitted),
                'workload_cv_in_second': float(work.std() / mean_work) if mean_work else 0.,
                'unconfirmed_transactions_total': outstanding,
                'unconfirmed_transactions_mean_per_shard': float(queue.mean()),
                'unconfirmed_transactions_max_shard': int(queue.max()),
                'allocated_cpu_cores_total': float(allocations[:, 0].sum()),
                'consumed_cpu_cores_total': float(sum(
                    shard.cpu_needs - before
                    for shard, before in zip(environment.shards, cpu_before))),
                'allocated_bandwidth_mbps_total': float(allocations[:, 1].sum()),
                'consumed_bandwidth_mbps_total': float(sum(
                    shard.bw_needs - before
                    for shard, before in zip(environment.shards, bw_before))),
                'block_size_transactions_mean': float(blocks.mean()),
                'cumulative_arrivals': environment.accepted,
                'cumulative_confirmed': environment.confirmed,
                'cumulative_cross_shard_rate':
                    environment.cross_count / max(1, environment.accepted),
                'dropped_transactions': 0,
            }
            rows.append(row)
            environment._latencies.clear()
            environment._confirmation_log.clear()
            if (second + 1) % 100 == 0 or second + 1 in (args.pre_seconds,
                    args.pre_seconds + args.burst_seconds):
                print(f't={second + 1}/{args.total_seconds}s phase={phase} '
                      f'arrival={arrivals} confirmed={confirmed} '
                      f'unconfirmed={outstanding}', flush=True)

        write_csv(output / 'seconds.csv', rows)
        phases = []
        for name in ('baseline_before', 'burst', 'recovery'):
            subset = [row for row in rows if row['phase'] == name]
            phases.append({
                'phase': name, 'seconds': len(subset),
                'mean_arrival_rate_tps': float(np.mean([r['actual_arrival_rate_tps'] for r in subset])),
                'mean_processing_rate_tps': float(np.mean([r['processing_rate_tps'] for r in subset])),
                'mean_queue_transactions_per_shard': float(np.mean([
                    r['unconfirmed_transactions_mean_per_shard'] for r in subset])),
                'mean_consumed_cpu_cores': float(np.mean([r['consumed_cpu_cores_total'] for r in subset])),
                'mean_consumed_bandwidth_mbps': float(np.mean([
                    r['consumed_bandwidth_mbps_total'] for r in subset])),
                'ending_unconfirmed_transactions': subset[-1]['unconfirmed_transactions_total'],
            })
        write_csv(output / 'phases.csv', phases)
        summary = {
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'shards': shards, 'seed': args.seed,
            'simulated_seconds': args.total_seconds,
            'total_arrived_and_admitted': environment.accepted,
            'total_confirmed': environment.confirmed,
            'final_unconfirmed_transactions': environment.accepted - environment.confirmed,
            'drop_rate': 0., 'post_horizon_drain': False,
            'completed_transaction_mean_latency_seconds':
                total_latency / total_latency_samples if total_latency_samples else None,
            'completed_transaction_latency_samples': total_latency_samples,
            'phases': phases, 'wall_seconds': time.monotonic() - started,
        }
        (output / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        run_protocol = {
            'schedule_total_system_tps': [[0, args.pre_seconds, args.base_rate],
                [args.pre_seconds, args.pre_seconds + args.burst_seconds, args.burst_rate],
                [args.pre_seconds + args.burst_seconds, args.total_seconds, args.base_rate]],
            'arrival_process': 'Poisson', 'arrival_order': 'csv',
            'slot_seconds': 1, 'service_rounds_per_slot': 1,
            'eq10_blocktime_seconds': 8,
            'queue_service_mode': 'paper_queue',
            'resource_budget_cpu_cores_bandwidth_mbps': list(budget),
            'resource_weights_eq10': list(weights),
            'service_capacity_scale': scale,
            'reference_capacity_tps': raw_reference,
            'critical_capacity_tps': args.critical_capacity_tps,
            'consumption_definition': 'Eq. (10) modeled usage accumulated by each shard, not measured hardware counters',
            'queue_definition': 'admitted original transactions minus confirmed originals, divided by shards for mean',
            'latency_definition': 'admission to confirmation of completed originals; no generator waiting',
            'drop_policy': 'open admission; unconfirmed transactions remain in queues at horizon',
            'inputs': [str(path) for path in inputs],
        }
        (output / 'protocol.json').write_text(
            json.dumps(run_protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        from plot_stability import draw
        draw(rows, output / 'stability-curves.png')
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        environment.close()


if __name__ == '__main__':
    main()
