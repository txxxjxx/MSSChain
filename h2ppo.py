"""Portable entry point for H2PPO training and single-policy evaluations."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def command(args):
    print('RUN: ' + shlex.join([str(a) for a in args]), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'placement', 'throughput', 'stability'])
    p.add_argument('--input', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--shards', type=int, default=32)
    p.add_argument('--transactions', type=int)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--load-prior-weight', type=float, default=256.)
    p.add_argument('--actor-lr', type=float, default=2e-5)
    p.add_argument('--cpu-budget', type=float, default=36.)
    p.add_argument('--bandwidth-budget', type=float, default=64.)
    p.add_argument('--base-rate', type=int, default=2000)
    p.add_argument('--burst-rate', type=int, default=8000)
    p.add_argument('--pre-seconds', type=int, default=500)
    p.add_argument('--burst-seconds', type=int, default=100)
    p.add_argument('--total-seconds', type=int, default=1200)
    a=p.parse_args()
    if min(a.shards,a.threads,a.cpu_budget,a.bandwidth_budget)<=0:
        p.error('shards, threads and budgets must be positive')
    a.input=[path.resolve() for path in a.input]
    a.output=a.output.resolve()
    if a.checkpoint:
        a.checkpoint=a.checkpoint.resolve()
    for path in a.input + ([a.checkpoint] if a.checkpoint else []):
        if not path.is_file():
            p.error(f'Missing file: {path}')
    if a.mode == 'stability':
        if a.transactions is not None:
            p.error('--transactions is not used for a fixed-duration stability test')
    else:
        a.transactions=a.transactions or (5_000_000 if a.mode=='train' else 2_000_000)
        if a.transactions<=0 or a.transactions%50_000:
            p.error('transactions must be a positive multiple of 50,000')
    if a.mode!='train' and a.checkpoint is None:
        p.error('--checkpoint is required for evaluation')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='train':
        if a.checkpoint is not None:
            p.error('Training starts from random initialization. --checkpoint is for evaluation only.')
        if (a.cpu_budget,a.bandwidth_budget)!=(36.,64.):
            p.error('Release training uses fixed network totals: 36 cores, 64 Mbps')
        if (a.output/'summary.json').exists():
            p.error('Training output already exists; choose a new output directory')
        cmd=[sys.executable,'-u',ROOT/'run_h2ppo_5m.py',
             '--output',a.output,'--shards',a.shards,'--train-only',
             '--train-transactions',a.transactions,'--evaluation-transactions',2_000_000,
             '--evaluation-from-start','--arrival-rate',8000,'--arrival-process','poisson',
             '--queue-service-mode','paper_queue','--critical-capacity-tps',8000,
             '--report-interval',50000,'--window-size',50000,
             '--control-mode','hierarchical','--control-architecture','equivariant',
             '--peak-queue-drift-weight',4,'--block-min',2000,'--block-max',6000,
             '--placement-chunk-size',1024,'--ppo-repeat',3,'--placement-update-size',2048,
             '--rollout-steps',8,'--placement-temporal-weight',1,
             '--actor-lr',a.actor_lr,'--critic-lr',0.0001,'--target-kl',0.01,
             '--seed',a.seed,'--torch-threads',a.threads,
             '--cpu-budget',36,'--bandwidth-budget',64,'--bandwidth-unit','Mb/s',
             '--cpu-weight',0.18,'--bandwidth-weight',0.1,'--resource-model','paper_eq10',
             '--valid-transaction-ratio',1,'--cpu-tps-per-core',312.5,
             '--locality-weight',0.5,'--reward-locality-weight',0.5,
             '--load-weight',a.load_prior_weight,'--prior-temperature',4,
             '--decay-seconds',31536000,'--cross-penalty',80,
             '--balance-sensitivity',8,'--load-feature-overload-power',2,
             '--queue-readiness-state','--address-affinity-weight',1,
             '--dependency-weight',3.2,'--neighbor-count-weight',48,
             '--neighbor-location-weight',16]
        cmd+=['--queue-prior-weight',0.1]
        cmd+=['--input',*a.input]
        command(cmd)
    elif a.mode=='stability':
        command([sys.executable,'-u',ROOT/'evaluate_stability.py',
                 '--checkpoint',a.checkpoint,'--shards',a.shards,'--output',a.output,
                 '--seed',a.seed,'--threads',a.threads,
                 '--cpu-budget',a.cpu_budget,'--bandwidth-budget',a.bandwidth_budget,
                 '--base-rate',a.base_rate,'--burst-rate',a.burst_rate,
                 '--pre-seconds',a.pre_seconds,'--burst-seconds',a.burst_seconds,
                 '--total-seconds',a.total_seconds,'--input',*a.input])
    elif a.mode=='throughput':
        command([sys.executable,'-u',ROOT/'evaluate_throughput.py',
                 '--checkpoint',a.checkpoint,'--shards',a.shards,'--output',a.output,
                 '--arrival-rate',8000,'--arrival-rate-unit','per_second',
                 '--slot-seconds',8,'--block-generation-seconds',8,
                 '--arrival-process','poisson','--arrival-order','hybrid_parent_ready',
                 '--minimum-admission-fraction',0.95,'--arrival-reorder-scan-multiple',64,
                 '--transactions',a.transactions,'--transaction-window',50000,
                 '--cpu-budget',a.cpu_budget,'--bandwidth-budget',a.bandwidth_budget,
                 '--override-checkpoint-budget','--placement-chunk-size',1024,
                 '--torch-threads',a.threads,'--fast-exit-after-write','--input',*a.input])
    else:
        placement(a)

def placement(a):
    import numpy as np
    import torch
    import blockchain.params as params
    from msschain_env import Supervisor
    from model_io import make_agent
    if (a.output/'summary.json').exists():
        raise ValueError('Evaluation output exists; choose a fresh directory')
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    shards=int(saved['num_shards'])
    if shards!=a.shards:
        raise ValueError('Checkpoint shard count differs from --shards')
    params.ShardNum=shards
    budget=(a.cpu_budget,a.bandwidth_budget)
    # Same environment for candidate, V22 and V26; checkpoint architecture is preserved.
    raw=shards*min(6000.,6000/8*sum(np.sqrt(w*b/shards) for w,b in zip((.18,.1),budget)))
    env=Supervisor(file_paths=a.input,batch_size=8000,window_size=50000,
        transaction_limit=a.transactions,retain_blocks=False,track_latency=False,
        enhanced_observation=True,stability_observation=int(saved['num_states'])==12*shards,
        resource_control=True,resource_max=budget,network_resource_budget=budget,
        initial_resources=np.asarray(budget)/shards,resource_weights=(.18,.1),
        resource_model='paper_eq10',valid_transaction_ratio=1.,cpu_tps_per_core=312.5,
        block_min=2000,block_max=6000,arrival_process='poisson',arrival_order='csv',
        alpha=.5,decay_seconds=31536000.,zeta=8.,load_feature_overload_power=2.,
        address_affinity=True,cross_penalty=80.,seed=a.seed,
        queue_service_mode='paper_queue',service_capacity_scale=8000/raw,
        peak_queue_drift_weight=4.)
    agent=make_agent(saved,budget,budget)
    agent.actor.eval(); agent.critic.eval()
    state=env.reset(); env.begin_metric_reporting(50000)
    start=time.monotonic(); reported=0; rows=[]
    with (a.output/'windows.csv').open('w',encoding='utf-8',newline='') as f:
        writer=None
        while env.accepted<a.transactions:
            remain=a.transactions-env.accepted
            if len(env.arriveTxArrive)>remain:
                state=env.limit_current_arrivals(remain)
            with torch.no_grad():
                state,_,done=agent.interact_batched(env,test=True,states=state,chunk_size=1024)
            if done and env.accepted<a.transactions:
                raise RuntimeError(f'Dataset exhausted at {env.accepted}')
            for metric in env.metric_series[reported:]:
                load=np.asarray(metric['interval_protocol_workload'],dtype=float)
                row=dict(window=reported+1,transactions=metric['transactions'],
                    window_transactions=50000,cross_shard_rate=metric['interval_cross_shard_rate'],
                    load_cv=float(load.std()/load.mean()) if load.mean() else 0.,
                    load_min=float(load.min()),load_max=float(load.max()),
                    load_mean=float(load.mean()),load_total=float(load.sum()),
                    load_by_shard=json.dumps(load.tolist()))
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=row.keys()); writer.writeheader()
                writer.writerow(row); f.flush(); rows.append(row); reported+=1
                print(f'[{time.monotonic()-start:.1f}s] {reported}/{a.transactions//50000}: cross={row["cross_shard_rate"]:.6f}, CV={row["load_cv"]:.6f}',flush=True)
    if reported!=a.transactions//50000:
        raise AssertionError('Incomplete 50k windows')
    summary=dict(checkpoint=str(a.checkpoint),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        transactions=env.accepted,windows=reported,shards=shards,
        mean_cross_shard_rate=float(np.mean([r['cross_shard_rate'] for r in rows])),
        mean_load_cv=float(np.mean([r['load_cv'] for r in rows])),
        mean_load_min=float(np.mean([r['load_min'] for r in rows])),
        mean_load_max=float(np.mean([r['load_max'] for r in rows])),
        mean_load_mean=float(np.mean([r['load_mean'] for r in rows])),
        mean_load_total=float(np.mean([r['load_total'] for r in rows])),
        arrival_order='csv',queue_service_mode='paper_queue',cpu_budget=budget[0],bandwidth_mbps=budget[1],
        alpha=.5,beta=.5,seed=a.seed,slot_seconds=1,arrival_rate_tps=8000,
        window_size=50000,service_capacity_scale=8000/raw,
        load_definition='original placement plus generated cross-shard protocol subtransactions',
        evaluation_replays_training_prefix=True,inputs=[str(x) for x in a.input],
        wall_seconds=time.monotonic()-start)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    env.close()
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
    import os
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
