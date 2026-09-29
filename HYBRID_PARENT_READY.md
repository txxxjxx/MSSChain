# `hybrid_parent_ready` transaction generation

The implementation is included in [`msschain_env.py`](msschain_env.py), not
downloaded from an external package. `Supervisor.gen_tx()` draws a Poisson
arrival count for each slot and invokes the `hybrid_parent_ready` branch when
the throughput evaluation requests that order. The branch and its candidate
cache helpers perform the following steps:

1. Release candidates whose Bitcoin input parents have confirmed, then fill
   the slot from this ready pool.
2. Scan later CSV transactions up to `arrival_count *
   arrival_reorder_scan_multiple`. Admit coinbase transactions and those whose
   parents are confirmed. Defer the rest, keyed by missing parent hash.
3. If the confirmed-ready scan cannot reach the configured minimum admission
   fraction, release a topological prefix whose parents are already admitted
   or precede the child in the current slot. Such children stay unconfirmed
   in W-Shard queues until their parents confirm. Raise an error if even this
   minimum cannot be reached; do not silently claim the nominal arrival rate.

This is a replay-order policy for the *protocol throughput and latency test*.
It may scan beyond the first 2M CSV records to admit 2M eligible originals.
Source-scan counts, reordered transactions and deferred-candidate counts are
reported in that test's `summary.json`. It neither creates new Bitcoin
transactions nor changes transaction hashes, inputs or outputs. The 5M
training run, the original-order 2M placement evaluation and the 1200-second
stability test all retain CSV order.

Run it through the public entry point:

```bash
python -u h2ppo.py throughput --checkpoint results/new-s32-train/h2ppo-5000000.pt --shards 32 --transactions 2000000 --output results/new-s32-throughput --input /data/transactions-000.csv /data/transactions-001.csv
```
