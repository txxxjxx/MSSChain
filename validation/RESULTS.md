# Code validation, not a final model-performance claim

The final upload configuration was checked with a newly initialized S=32 model. No pretrained checkpoint was loaded. Previous warm-start trials were excluded after clarification of the requested fresh-training protocol.

| Check | Completed scope | Outcome |
|---|---:|---|
| Fresh training | 100,000 original-order transactions | Checkpoint written; 36 cores / 64 Mbps; alpha=beta=0.5; no cross/CV guard |
| Ordinary model evaluation | First 200,000 CSV transactions, four independent 50k windows | Completed with `arrival_order=csv` |
| Throughput/latency flow | 50,000 admitted originals | Completed with `hybrid_parent_ready`; original and internal queue conservation errors both zero |
| Core regression checks | Five selected protocol-load/window/hybrid/H2PPO tests | All passed against packaged modules |
| Network resource allocation | S=4/8/16/32, 100 sampled actions each | CPU and bandwidth actions respect network totals |
| Portability | Python source imports/parsing and `bash -n run_server.sh` | Passed; shell script uses LF newlines |

The short CSV replay measured cross-rate 4.3990% and mean window protocol-load CV 16.4926%. These four early windows are not representative of the complete 2M replay and are not offered as a V22/V26 superiority result. The first 50k throughput flow check is also not a steady-state throughput benchmark.

The release defaults are 5M training and 2M evaluation. These full experiments have not been completed for a freshly initialized release model. No checkpoint or short-run experimental dataset is included in the upload package.

The added H2PPO-only stability entry point was smoke-tested at a 2000→8000→2000 Tx/s schedule for three one-second slots with a short fresh-training checkpoint and CSV data. It wrote all expected CSV/JSON/PNG artifacts, switched arrival phases correctly, and conserved generated, admitted, confirmed and queued original transactions; the burst left a nonzero queue. The included `hybrid_parent_ready` generator was separately checked with a parent-child synthetic transaction pair. These checks validate the code path, not the 1200-second performance result.

The actual validation model SHA256 was `fe7a09f5c48e7234b531525e6cbdb40af030bc7ed2c34502390cce5dadf0b720`; it is retained outside the upload directory for audit only. Runtime library versions are recorded in `environment.json`.
