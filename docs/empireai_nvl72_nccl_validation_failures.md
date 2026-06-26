<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Empire AI NVL72 Validation Communication Failures

This note captures the important symptoms and Slurm job IDs from the Empire AI
Beta NVL72 validation debugging runs.

## Environment

- Cluster: Empire AI Beta
- Data: `/mnt/home/jrusak/data/om4_halfdeg`
- Training launcher: `/mnt/home/jrusak/train_26_05_validation_repro.sbatch`
- Normal network settings used in these runs:
  - `NCCL_SOCKET_IFNAME=bond0`
  - `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5`
  - `MELLANOX_VISIBLE_DEVICES=all`
- Old image:
  - `ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:d7cf627257df3e1fda74202435e00f55c15ebe2d2be9648f47daf31905f51bad`
  - NCCL `2.27.7+cuda13.0`
- Latest image:
  - `ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:79564be9d5aa48acc40a803bf60f5b277bacee0fb9ad11e60735ee599e407421`
  - NCCL `2.29.7+cuda13.2`

The verbose debug setting used for comparison runs was:

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,COLL
TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_CPP_LOG_LEVEL=INFO
```

The narrow NCCL-only debug setting tested in `26224` was:

```bash
NCCL_DEBUG=INFO
```

The split debug runs tested after that were:

```bash
# 26226
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,COLL

# 26227
TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_CPP_LOG_LEVEL=INFO
```

The cuMem allocator toggle tested without debug vars in `26233` was:

```bash
NCCL_CUMEM_ENABLE=1
```

The NVLS toggle tested without debug vars in `26234` was:

```bash
NCCL_NVLS_ENABLE=0
```

## Failure Classes

| Class | Signature | Observed jobs |
| --- | --- | --- |
| A | `torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error`; Slurm logs also show `NCCL WARN Cuda failure 226`. This occurs during validation aggregation, after one-step validation has completed. | `26171`, `26193` |
| B | 2-node validation aggregation failure/hang with NCCL `unhandled system error` on an `ALLREDUCE`. This was distinct from class A. | `26172` |
| C | Latest-image 2-node post-validation hang. It reaches `Aggregating validation logs`, then Torch elastic reports an exit barrier timeout after 300 seconds. This is related to the validation aggregation area, but it is not the exact class-B NCCL `unhandled system error`. | `26201` |

## Runs

| Job | Image | Nodes / GPUs | Node placement | Debug vars | Result | Important symptom |
| ---: | --- | ---: | --- | --- | --- | --- |
| `26171` | old | 9 hosts / 36 GPUs | `b1-11,b1-12,b1-14,b1-15,b1-16,b1-17,b1-18,b1-28,b1-29` | off | Failed | Class A during validation aggregation at `(gen - target).cpu().numpy()`. |
| `26172` | old | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Failed/hung, later cancelled | Class B: NCCL `unhandled system error` on validation `ALLREDUCE`. |
| `26173` | old | 2 hosts / 8 GPUs | not rechecked | off | Completed | W&B SDK disabled; useful control for the validation image path. |
| `26190` | latest | 2 hosts / 8 GPUs | `b3-11,b3-12` | on | Completed | Latest image completed 2-host validation with debug enabled. |
| `26191` | old | 2 hosts / 8 GPUs | `b3-11,b3-12` | off | Completed | Adjacent 2-host old-image validation passed. |
| `26192` | old | 2 hosts / 8 GPUs | `b3-11,b3-12` | off | Completed | Repeat adjacent 2-host old-image validation passed. |
| `26193` | latest | 9 hosts / 36 GPUs | `b3-11,b3-12,b3-13,b3-14,b3-15,b3-16,b3-17,b3-18,b3-29` | off | Failed | Class A reproduced on latest image without verbose debug vars. |
| `26194` | latest | 9 hosts / 36 GPUs | same as `26193` | on | Completed | Same topology as `26193`, but validation aggregation completed. |
| `26195` | latest | 9 hosts / 36 GPUs | same as `26193` | on | Completed | Repeat debug run also completed. |
| `26201` | latest | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Hung, cancelled | Class C: reached `Aggregating validation logs`, then Torch elastic exit barrier timed out after 300 seconds. No exact class-B NCCL `unhandled system error` was found. |
| `26204` | latest | 2 hosts / 8 GPUs | `b3-12,b3-16` | on | Completed | Same placement as `26201`, but validation aggregation and process-group shutdown completed. |
| `26206` | latest | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Completed | Validation-shaped collective stress: 20 repeats of scalar reductions plus 154 half-degree map reductions with small rank-0 CPU-copy delay. |
| `26207` | latest | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Completed | Stronger validation-shaped collective stress: 50 repeats with `STRESS_RENDER_SLEEP=0.05`. |
| `26208` | old | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Completed | Old-image validation-shaped collective stress: 30 repeats with `STRESS_RENDER_SLEEP=0.05`. |
| `26210` | latest | 9 hosts / 36 GPUs | same as `26193` | off | Completed | 9-host validation-shaped collective stress: 50 repeats with `STRESS_RENDER_SLEEP=0.05`. |
| `26221` | latest | 2 hosts / 8 GPUs | `b3-12,b3-16` | off | Completed | Persistent-view validation stress: in-place reductions on `select()` views into long-lived map buffers, 20 repeats, `STRESS_RENDER_SLEEP=0.05`, 512 MiB CUDA allocation churn. |
| `26222` | latest | 9 hosts / 9 GPUs | `b3-11,b3-12,b3-13,b3-14,b3-15,b3-16,b3-17,b3-18,b3-29` | off | Completed | Persistent-view validation stress, 1 GPU per host, 20 repeats, `STRESS_RENDER_SLEEP=0.05`, 512 MiB CUDA allocation churn. |
| `26223` | latest | 9 hosts / 36 GPUs | same as `26222` | off | Completed | Persistent-view validation stress, 4 GPUs per host, 20 repeats, `STRESS_RENDER_SLEEP=0.05`, 512 MiB CUDA allocation churn. |
| `26224` | latest | 9 hosts / 36 GPUs | same as `26222` | `NCCL_DEBUG=INFO` only | Failed | Class A reproduced during real 1-epoch training. One-step validation completed, then validation map aggregation failed at `(gen - target).cpu().numpy()` with `NCCL WARN Cuda failure 226`. |
| `26225` | latest | 9 hosts / 36 GPUs | same as `26222` | `NCCL_DEBUG=INFO` only | Failed | Same as `26224`, but requested 72 CPUs per node (`--cpus-per-task=72`, 648 CPUs total). Class A still reproduced during validation map aggregation. |
| `26226` | latest | 9 hosts / 36 GPUs | same as `26222` | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,COLL` | Failed | Class A reproduced. NCCL subsystem logging alone did not prevent validation map aggregation from failing with `NCCL WARN Cuda failure 226`. |
| `26227` | latest | 9 hosts / 36 GPUs | same as `26222` | `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_CPP_LOG_LEVEL=INFO` | Completed | Real 1-epoch training completed. Validation aggregation finished, validation loss logged, and checkpoints were saved. |
| `26233` | latest | 9 hosts / 36 GPUs | same as `26222` | `NCCL_CUMEM_ENABLE=1` | Failed | Class A reproduced without debug vars. Validation reached `Aggregating validation logs`, then failed at `map.py:100` with `NCCL WARN Cuda failure 226` and `ALLREDUCE` errors on `NumelIn=259200` map tensors. |
| `26234` | latest | 9 hosts / 36 GPUs | same as `26222` | `NCCL_NVLS_ENABLE=0` | Failed/hung, cancelled | Disabling NVLS did not fix the failure. Validation reached `Aggregating validation logs`, then emitted NCCL watchdog `ALLREDUCE` errors on `NumelIn=259200` map tensors and later torch-elastic exit-barrier timeouts. Cancelled after it kept holding the allocation. |

## Code Path Notes

The suspicious validation path is W&B-enabled validation image aggregation:

- `src/samudra/train.py::validate_one_epoch()` logs `Aggregating validation logs`
  immediately before calling `val_aggregator.get_logs(label="val")`.
- `src/samudra/aggregator/main.py::get_validation_aggregator()` always includes
  reduced scalar metrics and adds `SnapshotAggregator` plus `MapAggregator` when
  validation images are enabled.
- `src/samudra/aggregator/validate/reduced.py::MeanAggregator._get_data()` does
  many scalar `all_reduce_mean()` calls.
- `src/samudra/aggregator/validate/map.py::MapAggregator.get_logs()` then does
  two full-map `all_reduce_mean()` calls per variable: one generated map and one
  target map.
- For half-degree OM4, each map tensor is `360 x 720`, or `259200` float values.
  This matches the `NumelIn=259200` / `TensorShape=[360, 720]` failure and debug
  signatures.
- With `thermo_dynamic_all`, the map path covers 77 prognostic variables, so
  image logging performs roughly 154 full-map all-reduces after the scalar
  reductions.
- Only rank 0 renders images after each generated/target map pair. Nonzero ranks
  immediately enter the next collective, so rank 0 can repeatedly lag while the
  other ranks wait in the next full-map all-reduce.
- The validation aggregation itself is single-threaded Python on each rank.
  Validation data can be prefetched by PyTorch DataLoader worker processes, and
  `TorchTrainDataset` can use a per-worker `ThreadPoolExecutor` to materialize
  xarray variables, but `ValidateAggregator.record_validation_batch()` and
  `ValidateAggregator.get_logs()` do not use Python-side thread pools or async
  tasks.

Successful debug logs from `26204` and `26194` confirm long runs of
`ALLREDUCE TensorShape=[360, 720]` during this phase. They also show visible
rank skew: some ranks enter a sequence number hundreds of milliseconds before
rank 0 in the 2-host run, and the 9-host run shows larger cross-rank spread.

## Validation-Shaped Stress Probe

`scripts/empireai_validation_collective_stress.sbatch` was added to reproduce
the suspected communication pattern without Samudra data loading or model code.
It runs:

- scalar all-reduces, matching reduced validation metric aggregation;
- generated and target `360 x 720` CUDA tensor all-reduces, matching
  `MapAggregator`;
- rank-0-only CPU copies of error, generated, and target maps between map
  collectives, matching the rank asymmetry in image rendering;
- optional `STRESS_RENDER_SLEEP` to increase rank-0 lag.

Current negative repro results:

- `26206`: latest image, `b3-12,b3-16`, 20 repeats,
  `STRESS_RENDER_SLEEP=0.02`; completed.
- `26207`: latest image, `b3-12,b3-16`, 50 repeats,
  `STRESS_RENDER_SLEEP=0.05`; completed.
- `26208`: old image, `b3-12,b3-16`, 30 repeats,
  `STRESS_RENDER_SLEEP=0.05`; completed.
- `26210`: latest image, same 9-host topology as `26193`, 50 repeats,
  `STRESS_RENDER_SLEEP=0.05`; completed.
- `26221`: latest image, `b3-12,b3-16`, 20 repeats, `STRESS_RENDER_SLEEP=0.05`,
  `STRESS_VIEW_MODE=1`, `STRESS_HIST=1`, `STRESS_VAL_BATCHES=8`,
  `STRESS_ALLOC_CHURN_MB=512`; completed.
- `26222`: latest image, 9 hosts / 9 GPUs, same 9-host topology as `26193`,
  20 repeats, `STRESS_RENDER_SLEEP=0.05`, `STRESS_VIEW_MODE=1`,
  `STRESS_HIST=1`, `STRESS_VAL_BATCHES=8`, `STRESS_ALLOC_CHURN_MB=512`;
  completed.
- `26223`: latest image, 9 hosts / 36 GPUs, same 9-host topology as `26193`,
  20 repeats, `STRESS_RENDER_SLEEP=0.05`, `STRESS_VIEW_MODE=1`,
  `STRESS_HIST=1`, `STRESS_VAL_BATCHES=8`, `STRESS_ALLOC_CHURN_MB=512`;
  completed.

The `26221` probe is closer to the suspicious `MapAggregator` path than the
earlier stress jobs. It accumulates generated and target maps into persistent
`[time, height, width]` CUDA buffers, then calls `dist.all_reduce()` in place on
`buffer.select(dim=0, index=hist)` views before rank 0 immediately transfers the
generated, target, and error maps to CPU.

These results mean the reduced-plus-map collective pattern and rank-0 CPU-copy
lag are not sufficient by themselves to reproduce either the 2-host failure on
the known bad placement or the 9-host class-A failure on the known 9-host
topology. In-place reductions on selected views into persistent validation-style
buffers are also not sufficient by themselves to reproduce the 2-host failure.
The remaining difference from full training is likely in the surrounding state:
preceding DDP/backward collectives and CUDA work, the actual validation tensors
coming from model outputs and masks, W&B/matplotlib rendering work instead of
only CPU copies/sleeps, process-group shutdown, or some interaction with the
training launcher environment.

## Takeaways

- The 9-host class-A failure reproduces without verbose debug vars on both old
  and latest images.
- On the same 9-host latest-image topology, enabling verbose NCCL/Torch debug
  vars changed behavior: two back-to-back runs completed.
- The original class-B 2-host NCCL `unhandled system error` has only been seen
  on the old image in job `26172`.
- On the latest image with the same `b3-12,b3-16` placement, the no-debug run
  produced a related post-validation hang instead of the exact class-B error.
- The latest-image 2-host debug run on that same placement completed.
- The validation image aggregation path is still the common suspicious region:
  failures and hangs occur after one-step validation and at or after
  `Aggregating validation logs`.
- Validation-shaped collective stress did not reproduce the 2-host failure on
  `b3-12,b3-16` with either the old or latest image, even when rank-0 lag was
  amplified. The simplest "many full-map all-reduces plus rank-0 CPU copies"
  hypothesis is therefore insufficient.
- The same latest-image stress pattern also completed on the exact 9-host
  topology that produced class A in `26193`.
- A closer persistent-view stress pattern also completed on `b3-12,b3-16`, so
  the `MapAggregator` view aliasing pattern is not enough by itself to trigger
  the observed 2-host failure.
- The same persistent-view stress also completed on the known 9-host topology
  with both 1 GPU per host and 4 GPUs per host, so the view aliasing plus
  9-host/36-rank communicator shape is not enough by itself to trigger class A.
- Real 9-host / 36-GPU training with only `NCCL_DEBUG=INFO` still reproduced
  class A in `26224`. This suggests the behavior change seen with the earlier
  successful debug runs likely depended on the broader debug bundle
  (`NCCL_DEBUG_SUBSYS`, `TORCH_DISTRIBUTED_DEBUG`, and/or
  `TORCH_CPP_LOG_LEVEL`), not on `NCCL_DEBUG=INFO` alone.
- Requesting 50% of each node's CPUs did not change the outcome. `26225`
  requested `--cpus-per-task=72` on 144-CPU hosts, for 648 CPUs total, and
  still reproduced class A at the same validation map aggregation point.
- Splitting the broader debug bundle points at the Torch-side debug settings as
  the behavior-changing piece. `26226` failed with NCCL debug plus NCCL
  subsystem logging, while `26227` completed with `TORCH_DISTRIBUTED_DEBUG` and
  `TORCH_CPP_LOG_LEVEL` only.
- In `26227`, PyTorch reported `TORCH_NCCL_ENABLE_TIMING: 1`,
  `TORCH_NCCL_TRACE_BUFFER_SIZE: 2000`, and
  `TORCH_NCCL_DESYNC_DEBUG: 1` in the ProcessGroupNCCL environment. Those
  derived Torch/NCCL watchdog settings are now the most plausible candidates
  for why the debug run changes behavior.
- `NCCL_CUMEM_ENABLE=1` did not fix the failure. `26233` failed on the same
  9-host topology after one-step validation at validation map aggregation, with
  both the rank-0 `torch.AcceleratorError` at `map.py:100` and NCCL watchdog
  `ALLREDUCE` errors on `259200`-element map tensors.
- `NCCL_NVLS_ENABLE=0` also did not fix the failure. `26234` hit the same
  post-validation map all-reduce failure signature, then hung in torch-elastic
  exit-barrier cleanup until manually cancelled.

## Useful Remote Logs

| Job | Primary logs |
| ---: | --- |
| `26171` | `/mnt/home/jrusak/oe-halfdeg_26171.out`, `/mnt/home/jrusak/oe-halfdeg_26171.err`, `/mnt/home/jrusak/runs/halfdeg-full-1epoch-20260624-191104/experiment.log`, `/mnt/home/jrusak/runs/halfdeg-full-1epoch-20260624-191104/error.log` |
| `26172` | `/mnt/home/jrusak/oe-halfdeg_26172.out`, `/mnt/home/jrusak/oe-halfdeg_26172.err` |
| `26193` | `/mnt/home/jrusak/oe-halfdeg_26193.out`, `/mnt/home/jrusak/oe-halfdeg_26193.err`, `/mnt/home/jrusak/runs/2026-06-25-26-05-9node-validation-repro-nodebug/experiment.log`, `/mnt/home/jrusak/runs/2026-06-25-26-05-9node-validation-repro-nodebug/error.log` |
| `26194` | `/mnt/home/jrusak/oe-halfdeg_26194.out`, `/mnt/home/jrusak/oe-halfdeg_26194.err`, `/mnt/home/jrusak/runs/2026-06-25-26-05-9node-validation-repro-debug/experiment.log` |
| `26195` | `/mnt/home/jrusak/oe-halfdeg_26195.out`, `/mnt/home/jrusak/oe-halfdeg_26195.err`, `/mnt/home/jrusak/runs/2026-06-25-26-05-9node-validation-repro-debug-repeat/experiment.log` |
| `26201` | `/mnt/home/jrusak/oe-halfdeg_26201.out`, `/mnt/home/jrusak/oe-halfdeg_26201.err`, `/mnt/home/jrusak/runs/2026-06-25-26-05-2node-b-repro-nodebug/experiment.log` |
| `26204` | `/mnt/home/jrusak/oe-halfdeg_26204.out`, `/mnt/home/jrusak/oe-halfdeg_26204.err`, `/mnt/home/jrusak/runs/2026-06-25-26-05-2node-b-repro-debug/experiment.log` |
| `26206` | `/mnt/home/jrusak/val-coll-stress_26206.out`, `/mnt/home/jrusak/val-coll-stress_26206.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26206-nodes2` |
| `26207` | `/mnt/home/jrusak/val-coll-stress_26207.out`, `/mnt/home/jrusak/val-coll-stress_26207.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26207-nodes2` |
| `26208` | `/mnt/home/jrusak/val-coll-stress_26208.out`, `/mnt/home/jrusak/val-coll-stress_26208.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26208-nodes2` |
| `26210` | `/mnt/home/jrusak/val-coll-stress_26210.out`, `/mnt/home/jrusak/val-coll-stress_26210.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26210-nodes9` |
| `26221` | `/mnt/home/jrusak/codex_stress/val-coll-stress_26221.out`, `/mnt/home/jrusak/codex_stress/val-coll-stress_26221.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26221-nodes2` |
| `26222` | `/mnt/home/jrusak/codex_stress/val-coll-stress_26222.out`, `/mnt/home/jrusak/codex_stress/val-coll-stress_26222.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26222-nodes9` |
| `26223` | `/mnt/home/jrusak/codex_stress/val-coll-stress_26223.out`, `/mnt/home/jrusak/codex_stress/val-coll-stress_26223.err`, `/mnt/home/jrusak/runs/validation-collective-stress/26223-nodes9` |
| `26224` | `/mnt/home/jrusak/oe-halfdeg_26224.out`, `/mnt/home/jrusak/oe-halfdeg_26224.err`, `/mnt/home/jrusak/runs/2026-06-26-nccl-info-only-9x4/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-nccl-info-only-9x4/error.log` |
| `26225` | `/mnt/home/jrusak/oe-halfdeg_26225.out`, `/mnt/home/jrusak/oe-halfdeg_26225.err`, `/mnt/home/jrusak/runs/2026-06-26-nccl-info-only-9x4-cpus72/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-nccl-info-only-9x4-cpus72/error.log` |
| `26226` | `/mnt/home/jrusak/oe-halfdeg_26226.out`, `/mnt/home/jrusak/oe-halfdeg_26226.err`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-subsys-only/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-subsys-only/error.log` |
| `26227` | `/mnt/home/jrusak/oe-halfdeg_26227.out`, `/mnt/home/jrusak/oe-halfdeg_26227.err`, `/mnt/home/jrusak/runs/2026-06-26-9x4-torch-debug-only/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-9x4-torch-debug-only/error.log` |
| `26233` | `/mnt/home/jrusak/oe-halfdeg_26233.out`, `/mnt/home/jrusak/oe-halfdeg_26233.err`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-cumem/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-cumem/error.log` |
| `26234` | `/mnt/home/jrusak/oe-halfdeg_26234.out`, `/mnt/home/jrusak/oe-halfdeg_26234.err`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-nvls-off/experiment.log`, `/mnt/home/jrusak/runs/2026-06-26-9x4-nccl-nvls-off/error.log` |
