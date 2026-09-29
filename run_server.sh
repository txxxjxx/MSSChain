#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/server}"
TRAIN_TRANSACTIONS="${TRAIN_TRANSACTIONS:-5000000}"
EVAL_TRANSACTIONS="${EVAL_TRANSACTIONS:-2000000}"
TORCH_THREADS="${TORCH_THREADS:-8}"
SHARDS="${SHARDS:-4 8 16 32}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"
RUN_STABILITY="${RUN_STABILITY:-0}"
if (( $# < 1 )); then
  echo "Usage: bash run_server.sh INPUT0.csv INPUT1.csv [INPUT2.csv ...]" >&2
  exit 2
fi
INPUTS=("$@")
for input in "${INPUTS[@]}"; do
  [[ -f "$input" ]] || { echo "Missing data: $input" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/run.lock"
flock -n 9 || { echo "This output directory is already in use" >&2; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$TORCH_THREADS" MKL_NUM_THREADS="$TORCH_THREADS"
ulimit -n 4096
for s in $SHARDS; do
  out="$OUTPUT_ROOT/s$s"
  mkdir -p "$out"
  checkpoint="$out/train/h2ppo-$TRAIN_TRANSACTIONS.pt"
  if [[ ! -f "$checkpoint" || ! -f "$out/train/summary.json" ]]; then
    "$PYTHON" -u h2ppo.py train --shards "$s" --transactions "$TRAIN_TRANSACTIONS" --threads "$TORCH_THREADS" --output "$out/train" --input "${INPUTS[@]}" 2>&1 | tee "$out/train.log"
  fi
  if [[ ! -f "$out/placement/summary.json" ]]; then
    "$PYTHON" -u h2ppo.py placement --checkpoint "$checkpoint" --shards "$s" --transactions "$EVAL_TRANSACTIONS" --threads "$TORCH_THREADS" --output "$out/placement" --input "${INPUTS[@]}" 2>&1 | tee "$out/placement.log"
  fi
  if [[ "$RUN_THROUGHPUT" == 1 && ! -f "$out/throughput/summary.json" ]]; then
    "$PYTHON" -u h2ppo.py throughput --checkpoint "$checkpoint" --shards "$s" --transactions "$EVAL_TRANSACTIONS" --threads "$TORCH_THREADS" --output "$out/throughput" --input "${INPUTS[@]}" 2>&1 | tee "$out/throughput.log"
  fi
  if [[ "$RUN_STABILITY" == 1 && ! -f "$out/stability/summary.json" ]]; then
    "$PYTHON" -u h2ppo.py stability --checkpoint "$checkpoint" --shards "$s" --seed 0 --threads "$TORCH_THREADS" --output "$out/stability" --input "${INPUTS[@]}" 2>&1 | tee "$out/stability.log"
  fi
done
"$PYTHON" summarize.py --root "$OUTPUT_ROOT"
date -Iseconds > "$OUTPUT_ROOT/complete.txt"
echo "Completed: $OUTPUT_ROOT"
