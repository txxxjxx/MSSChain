# MSSChain H2PPO source: training, retraining and stability

This source-only package implements H2PPO. It includes fresh training, checkpoint warm-start retraining, H2PPO-only stability simulation, and the previously requested placement/throughput verification paths. It contains no pretrained model, dataset, comparison baseline, or historical experimental output. See `运行说明.md` for commands.

## Public modes

| Mode | Transaction order | Default size | Output |
|---|---|---:|---|
| `train` | CSV order across supplied files | 5M | Newly trained model and training metrics |
| `retrain` | CSV order across supplied files | 1M | Warm-started actor/critic and new training metrics |
| `placement` | Original CSV prefix | 2M | 40 independent 50k windows: crossing rate, CV, shard loads |
| `throughput` | `hybrid_parent_ready` | 2M admitted originals | Per-consensus throughput, confirmed latency and backlog |
| `stability` | CSV order | 1200 simulated seconds | Per-second arrival, service, queue, resource and delay series |

Fresh training initializes a new model for each shard count (4, 8, 16, 32). `retrain` requires a compatible checkpoint and starts a new PPO optimization run from its actor and critic weights; optimizer state and prior simulation queues are not restored. Retraining reads the supplied CSV files from their beginning, so use a distinct later dataset when disjoint fine-tuning data is required. Architecture: hierarchical H2PPO, shared placement actor, equivariant resource controller.

## Parameters and interpretation

- Network totals: **36 CPU cores / 40 Mb/s**, shared by all shards; Eq. (10) CPU/bandwidth coefficients **0.03 / 0.02**. Mean total arrival rate **8,000 Tx/s**, Poisson arrivals. Model block actions **2,000–6,000 transactions**.
- Training objective **alpha=beta=0.5**. Both `locality-weight` and `reward-locality-weight` are 0.5. The trainable feature prior TD=3.2, Nbr=48, ANbr=16, load=256, temperature=4, and cross penalty 80 are separate shaping coefficients; equal alpha/beta does not equalize all raw reward magnitudes.
- Training, retraining and ordinary placement evaluation use the `paper_queue` abstraction. The Eq. (10) full-budget reference is calibrated to 8,000 Tx/s, computed from 36/40 and the shard count, and recorded in the protocol. This is a simulation calibration, not a physical CPU performance guarantee.
- Only throughput/latency tests use the explicit cross-shard protocol with `hybrid_parent_ready`, scan multiple 64, admission floor 0.95, 8-second arrival slots and 8-second block generation. **No paper-queue capacity multiplier** is applied in this mode. Blocks and resources remain model outputs.
- The single-policy stability test uses 1-second Poisson arrival and service slots: 2000 Tx/s in [0,500), 8000 Tx/s in [500,600), then 2000 Tx/s in [600,1200), all rates for the **whole system**. Its default network budget is 36 CPU cores / 40 Mb/s, using the checkpoint's Eq. (10) weights, **8-second Eq. (10) block-time coefficient**, and the same 8000 Tx/s paper-queue capacity calibration as training and the ordinary placement evaluation. One service decision still executes every simulated second in this abstraction; this is not an 8-second consensus-round test. It does not use `hybrid_parent_ready`; that policy is confined to the protocol throughput/latency test.
- No cross-rate target, CV projection, or forced resource saturation is enabled. The replay admission floor is a generator policy, separate from placement constraints.

## Statistics

Cross-rate = cross-shard originals / all originals including coinbase in each independent 50,000-transaction window. CV = population standard deviation / mean of that window's protocol-workload vector. Aggregate CV is the arithmetic mean of the 40 window CVs.

Protocol workload counts one internal original at its shard. A cross-shard original with m distinct remote parent shards adds one original plus (2m+1) generated work items to its origin and one work item to each remote parent shard. This is admission-attributed protocol work, not already-confirmed work. The controller's Eq. (6) load is a separate internal feature.

Throughput counts confirmed original transactions per simulated second. The primary completed latency is transaction-weighted admission-to-confirmation time; pre-admission generator waiting has no external send timestamp and is not included. Unconfirmed originals are reported separately and the run does not drain after arrival completion.

The hybrid test stops at 2M admitted originals after replay scheduling. It may scan beyond the first 2M CSV records, so its transaction set can differ from the original-order placement evaluation prefix. It reports source transactions scanned and deferred candidates.

Ordinary evaluation replays the first 2M original transactions; it overlaps the training prefix if the same input files are used. It is not a held-out generalization test. No performance claim is made for a model that has not been trained with this package.

## Files

- `h2ppo.py`: public `train`, `retrain`, `placement`, `throughput`, `stability` CLI; portable explicit input paths.
- `run_h2ppo_5m.py`: low-level PPO implementation; invoke through the public CLI for release defaults.
- `evaluate_throughput.py`: H2PPO protocol evaluation without unrelated baselines.
- `evaluate_stability.py`: H2PPO-only arrival-burst stability test, without comparison methods.
- `plot_stability.py`: six-panel plot from the H2PPO-only per-second stability series.
- `msschain_env.py`: contains the actual `hybrid_parent_ready` generator in `Supervisor.gen_tx`, with the parent-candidate cache and ready/deferred release logic in its helper methods. The throughput entry point selects it; see `HYBRID_PARENT_READY.md`.
- `model_io.py`, `algo/`, `blockchain/`, `shard/`, `msschain_env.py`: required dependencies.
- `run_server.sh`: serial fresh training for selected shard counts; optional placement, throughput and stability stages are opt-in; retraining uses `h2ppo.py retrain` explicitly.
- `summarize.py`: aggregates completed experiments without mixing CSV/hybrid cross-rate and CV.
- `SOURCE_MANIFEST.json`: hashes of packaged source files; `MANIFEST.sha256`: hashes of final upload files.

Training writes protocol, checkpoints, PPO update statistics and window metrics. Placement writes `windows.csv` and `summary.json`. Throughput writes `consensus-rounds.csv`, `transaction-windows.csv`, `protocol.json`, `summary.json` and `run.log`.
Stability writes `seconds.csv`, `phases.csv`, `summary.json`, `protocol.json`, and `stability-curves.png`; the per-second queue counts unconfirmed original transactions, and consumption is the simulator's Eq. (10) estimate rather than physical hardware telemetry. Completed latency excludes generator waiting before admission. The simulation leaves unfinished transactions in the queues and has no admission shedding.

Use fresh output directories when changing inputs or settings. The server launcher skips existing completed stages and locks its output root against duplicate launches. Successful evaluation closes files before fast process exit; failures return a nonzero status. The package does not alter existing experiment directories.
