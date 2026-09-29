"""Combine completed H2PPO placement and throughput experiments without mixing definitions."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for s in (4,8,16,32):
        placement=a.root/f's{s}/placement/summary.json'
        throughput=a.root/f's{s}/throughput/summary.json'
        if not placement.exists():
            continue
        v=json.loads(placement.read_text(encoding='utf-8'))
        t=json.loads(throughput.read_text(encoding='utf-8')) if throughput.exists() else {}
        rows.append(dict(shards=s,placement_order='csv',placement_transactions=v['transactions'],
            placement_window_cross_rate=v['mean_cross_shard_rate'],placement_window_cv=v['mean_load_cv'],
            mean_window_min_load=v['mean_load_min'],mean_window_max_load=v['mean_load_max'],
            mean_window_mean_load=v['mean_load_mean'],mean_window_total_load=v['mean_load_total'],
            throughput_order=t.get('arrival_order'),throughput_transactions=t.get('transactions'),
            throughput_window_cross_rate=t.get('mean_transaction_window_cross_shard_rate'),
            throughput_window_cv=t.get('mean_transaction_window_protocol_load_cv'),
            throughput_tps=t.get('mean_consensus_round_throughput_tps'),
            admitted_to_confirmed_latency_seconds=t.get('overall_completed_latency_seconds_sample_weighted'),
            final_unconfirmed=t.get('final_unconfirmed_original_transactions')))
    if not rows:
        raise ValueError('No completed placement evaluations found')
    with (a.root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    main()
