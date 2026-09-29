"""Plot the six H2PPO stability series from an evaluated seconds.csv file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(rows, output):
    def values(key):
        return [float(row[key]) if row[key] not in ('', None) else float('nan')
                for row in rows]

    seconds = values('second')
    boundaries = [float(rows[index]['second']) - 1
                  for index in range(1, len(rows))
                  if rows[index]['phase'] != rows[index - 1]['phase']]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    panels = [
        (axes[0, 0], [('target_arrival_rate_tps', 'Target'),
                      ('actual_arrival_rate_tps', 'Arrived'),
                      ('processing_rate_tps', 'Confirmed')], 'Transactions / s'),
        (axes[0, 1], [('unconfirmed_transactions_mean_per_shard', 'Per-shard mean')],
         'Unconfirmed originals / shard'),
        (axes[1, 0], [('allocated_cpu_cores_total', 'Allocated'),
                      ('consumed_cpu_cores_total', 'Eq. (10) consumed')], 'CPU cores'),
        (axes[1, 1], [('allocated_bandwidth_mbps_total', 'Allocated'),
                      ('consumed_bandwidth_mbps_total', 'Eq. (10) consumed')], 'Bandwidth (Mb/s)'),
        (axes[2, 0], [('cross_shard_rate_including_coinbase', 'Cross-shard rate'),
                      ('workload_cv_in_second', 'Workload CV')], 'Per-second fraction'),
        (axes[2, 1], [('completed_mean_latency_seconds', 'Completed mean')],
         'Admission-to-confirmation latency (s)'),
    ]
    for ax, lines, ylabel in panels:
        for key, label in lines:
            ax.plot(seconds, values(key), linewidth=1.2, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_xlabel('Simulated time (s)')
        for boundary in boundaries:
            ax.axvline(boundary, color='grey', linestyle='--', linewidth=.8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No stability rows')
    draw(rows, args.output)
