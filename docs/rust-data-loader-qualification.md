<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: Apache-2.0
-->

# Rust data loader production qualification

This report records the Stage 5 flat-OM4 evidence and the follow-on OM4-compact
evidence for the opt-in local Rust loader described in
[the shipping plan](rust-data-loader-plan.md).

## Acceptance thresholds

- CPU and Rust runs must emit identical rank-local batch schedules.
- Per-batch loss must remain within `1e-4` relative and absolute tolerance. This
  is intentionally tighter than the normal cross-hardware reproducibility
  tolerance.
- Single-GPU and two-rank DDP runs must cross an autoregressive step transition,
  validate, checkpoint, resume, and terminate without retained workers or
  prefetch threads.
- Rust must provide at least 1.5x end-to-end throughput, or profiling must show
  exposed data wait is no longer on the critical path.
- Pinned host buffers must remain bounded by the configured queue and reach zero
  in-use bytes after clean exhaustion.

## Local single-GPU setup

- GPU: NVIDIA GB10
- Data: 983 MB local flat-OM4 fixture, 48 time points at 180 x 360
- Variables: `thermo_dynamic_5` prognostics and `tau_hfds` boundaries
- Batch size: 4
- Epochs: 2, transitioning from one to two autoregressive steps at epoch 2
- CPU baseline: four persistent PyTorch workers
- Rust: two prefetched batches, eight native concurrent reads, pinned-buffer and
  CUDA device prefetch enabled
- System sampling interval: 100 ms

The reproducible config is `configs/qualification/rust_loader.yaml`; the runner
is `scripts/qualify_rust_loader.py`.

## Single-GPU results

| metric | CPU baseline | Rust | change |
| --- | ---: | ---: | ---: |
| wall time | 5.165 s | 4.154 s | 1.24x faster |
| measured train throughput | 3.067 batch/s | 3.843 batch/s | 1.25x |
| exposed wait p50 | 0.481 ms | 0.586 ms | effectively hidden |
| exposed wait p95 | 400.6 ms | 89.7 ms | 4.47x lower |
| exposed wait mean | 67.4 ms | 17.4 ms | 3.87x lower |
| raw load p50 | 105.8 ms | 21.2 ms | 4.98x lower |
| raw load p95 | 154.5 ms | 35.9 ms | 4.30x lower |
| process RSS max | 3.148 GB | 2.953 GB | 195 MB lower |
| process CPU mean | 73.7% | 94.8% | 21.1 points higher |
| GPU utilization mean | 14.5% | 43.7% | 29.2 points higher |
| Torch device allocation max | 1.544 GB | 1.638 GB | 94 MB higher |
| Torch device reservation max | 2.389 GB | 2.871 GB | 482 MB higher |

The two loaders produced identical schedule hashes for every train and
validation epoch. Across 12 train batches, maximum loss error was `4.41e-5`
absolute and `1.76e-5` relative.

The raw 1.5x throughput target is not reached by this very short, initialization-
dominated run. It satisfies the alternative criterion: after warm-up the median
exposed wait is below 0.6 ms, mean wait is 17.4 ms versus a 260 ms mean iteration,
and the CUDA trace shows overlap rather than a serialized data critical path.

## Checkpoint and resume

Both loaders completed validation and wrote latest, best-validation, EMA, and
per-epoch checkpoints at the one-to-two-step transition. Resuming `ckpt.pt` ran
epoch 3 at two steps and wrote `ckpt_3.pt`. Resume schedule hashes were identical;
maximum loss error was `1.12e-5` absolute and `5.44e-6` relative. Both processes
exited cleanly. The CPU qualification runner explicitly releases its persistent
workers before process-exit timing.

## Pinned and device-prefetch memory

The Rust training pool allocated 279.9 MB, reused 18 tensors, peaked at 279.9 MB
in use, and ended at zero in-use bytes. The validation pool allocated 163.3 MB,
reused three tensors, peaked at 140.0 MB in use, and also ended at zero. The
different allocation total reflects the final partial validation batch shape.
This is consistent with the configured two-batch queue plus the batch being
prepared; CUDA-event tests prevent reuse before an asynchronous copy completes.

An Nsight Systems trace recorded 270.9 MB of training H2D traffic on the prefetch
stream. Of its 6.951 ms copy time, 2.669 ms (38.4%) overlapped model-stream kernels.
The trace report and derived CSV are local qualification artifacts rather than
source-controlled files.

## Multi-rank DDP

The final production qualification used commit `88b28eb6` in the published
PhysicsNeMo 26.05 image on NYU Torch. Each rank used one RTX6000 GPU; the DDP
runs used two nodes and two ranks. Data came from the local flat-OM4 production
store at `/scratch/jr7309/data/om4_onedeg_v3`. Summary timings exclude the
one-time OCI-to-SIF conversion.

### One-GPU production results

| metric | CPU baseline | Rust | change |
| --- | ---: | ---: | ---: |
| qualification wall time | 19.065 s | 11.538 s | 1.65x faster |
| measured train throughput | 0.746 batch/s | 1.439 batch/s | 1.93x |
| exposed wait p50 | 0.720 ms | 174.8 ms | higher |
| exposed wait p95 | 1.946 s | 510.6 ms | 3.81x lower |
| exposed wait mean | 478.1 ms | 192.9 ms | 2.48x lower |
| raw load p50 | 1.568 s | 212.1 ms | 7.39x lower |
| raw load p95 | 1.936 s | 374.9 ms | 5.16x lower |
| process RSS max | 3.498 GB | 3.226 GB | 272 MB lower |
| process CPU mean | 23.3% | 37.4% | 14.1 points higher |
| GPU utilization mean | 2.1% | 3.3% | 1.2 points higher |
| Torch device allocation max | 1.543 GB | 0.759 GB | 784 MB lower |
| Torch device reservation max | 2.414 GB | 2.091 GB | 323 MB lower |

The CPU loader's near-zero median exposed wait reflects its worker queue being
ready for most batches; its expensive work is visible in the separate load-time
metric and in the long-tail wait. Rust meets the throughput gate and materially
reduces mean and p95 exposed wait.

### Two-rank production results

Ranges below cover rank 0 and rank 1.

| metric | CPU baseline | Rust | change |
| --- | ---: | ---: | ---: |
| qualification wall time | 13.574-13.585 s | 9.301-9.303 s | 1.46x faster |
| measured train throughput | 0.419-0.425 batch/s | 0.681-0.699 batch/s | 1.60-1.67x |
| exposed wait p50 | 0.899-0.906 ms | 20.1-136.4 ms | higher |
| exposed wait p95 | 1.975-2.065 s | 403-557 ms | 3.55-5.12x lower |
| exposed wait mean | 625-648 ms | 126-216 ms | 2.90-5.13x lower |
| raw load p50 | 1.537-1.574 s | 181-199 ms | 7.72-8.70x lower |
| raw load p95 | 1.794-1.857 s | 261-444 ms | 4.04-7.12x lower |
| process RSS max | 3.403-3.512 GB | 3.269-3.318 GB | lower on both ranks |
| process CPU mean | 29.8-34.8% | 40.9-88.9% | higher |
| GPU utilization mean | 3.7-7.6% | 5.8-28.3% | higher |

For both initial epochs, each CPU rank's train and validation schedule hashes
exactly match the corresponding Rust rank. The two ranks have different hashes,
confirming that the evidence is rank-local rather than a duplicated rank-0
record. Maximum initial loss error is `5.56e-5` absolute and `1.93e-5` relative;
the one-GPU maximum is `7.92e-5` absolute and `4.02e-5` relative.

All four initial jobs crossed the one-to-two-step transition, validated, and
wrote checkpoints. Matched resume jobs loaded `ckpt.pt`, ran epoch 3 at two
steps, and wrote `ckpt_3.pt`. Resume schedules match on every rank. Maximum
resume loss error is `1.34e-5` absolute on one GPU and `2.93e-5` absolute in
DDP. All eight jobs exited zero without retained workers, prefetch threads, or
NCCL process-group teardown warnings.

The production Rust pools allocated 115.1 MB for training on every rank and
68.4-91.8 MB for validation. Every pool ended at zero in-use bytes. Torch's
device samples and the loader's own pool counters show bounded host and device
prefetch memory.

The initial Torch jobs were `13854049`, `13855133`, `13855134`, and `13855135`;
the resume jobs were `13855238` through `13855241`.

## OM4-compact production qualification

The compact qualification used a 1.1 GB, 56-time-point local fixture at
`/scratch/jr7309/data/om4_compact_qual_af649b01`. Depth arrays are float32 with
shape `(lev=19, time=56, y=180, x=360)` and chunks `(19, 1, 180, 360)`; surface
arrays use `(time, y, x)` with one time point per chunk. The Rust reader directly
maps canonical channels such as `thetao_4` to `thetao[lev=4]`. It does not create
a canonical-manifest object.

### Correctness, DDP, and resume

Commit `af649b01` completed matched CPU and Rust initial jobs on one GPU and two
RTX6000 ranks. Every rank-local training and validation schedule hash matched.
Maximum initial loss error was `2.90e-5` absolute on one GPU and `1.48e-5` in
DDP. All jobs crossed the one-to-two-step transition, validated, checkpointed,
and exited zero.

The four matched resume jobs loaded `ckpt.pt`, ran epoch 3 at two steps, and
wrote `ckpt_3.pt`. Resume schedule hashes matched on every rank; maximum loss
error was `1.60e-5` on one GPU and `2.93e-5` in DDP. Every Rust training and
validation pool ended at zero in-use bytes. The initial jobs were `13866404`
through `13866407`; resume jobs were `13866805` through `13866808`.

### Production performance and grouped depth reads

The first isolated compact run exposed a prototype gap. Because one physical
chunk contains all 19 levels, reading each of five canonical levels separately
repeated filesystem operations for the same physical variable and time. Rust
managed only 0.300 batch/s versus the CPU loader's 1.530 batch/s, so this result
was treated as a no-go rather than weakening the performance gate.

Commit `4a65b3cb` groups requested logical levels by physical array and reads
the required level span once per variable and time index. A final sequential,
same-image comparison ran ten epochs and 60 train batches per loader on one
RTX6000 with eight CPUs and 64 GB per job:

| metric | CPU baseline | Rust grouped reads | change |
| --- | ---: | ---: | ---: |
| qualification wall time | 33.372 s | 15.600 s | 2.14x faster |
| measured train throughput | 2.263 batch/s | 5.233 batch/s | 2.31x |
| raw load p50 | 805.9 ms | 71.3 ms | 11.31x lower |
| raw load p95 | 928.8 ms | 106.0 ms | 8.76x lower |
| exposed wait p50 | 1.073 ms | 43.2 ms | higher |
| exposed wait p95 | 1.017 s | 174.7 ms | 5.82x lower |
| exposed wait mean | 283.0 ms | 61.2 ms | 4.63x lower |
| process RSS max | 3.574 GB | 3.510 GB | 64 MB lower |
| process CPU mean | 28.3% | 85.6% | 57.3 points higher |
| GPU utilization mean | 5.6% | 11.2% | 5.6 points higher |

All 60 schedules matched and maximum loss error was `4.18e-6` absolute and
`2.31e-6` relative. The Rust pool allocated at most 208.4 MB for training and
141.5 MB for validation, stayed within the configured bounded queue, and ended
at zero in-use bytes. The final comparison jobs were `13870309` and `13870310`.

The initial per-level implementation remains useful evidence: it made the
physical chunk-layout sensitivity visible and established the grouped-read
optimization as a release requirement, not an optional benchmark improvement.

## Opt-in configuration and limitations

Select the loader in a data config:

```yaml
data:
  loading:
    type: rust
    prefetch_batches: 2
    max_concurrent_reads: 8
    prefetch_to_device: true
```

Source checkouts require `uv sync --extra rust`; the published training
container includes the compiled extension. The current path supports local flat
and compact OM4 training and validation. It intentionally does not support S3,
LLC, or inference. Package releases must publish the matching platform-native
loader wheel alongside the Samudra wheel.

## Test evidence

- `332 passed, 2 skipped, 71 deselected, 10 xfailed` for
  `pytest -m "not manual and not cuda"` with CUDA hidden
- `30 passed` in the sampler suite
- `18 passed` in non-CUDA Rust data tests
- `2 passed` in focused CUDA prefetch and pinned-buffer reuse tests
- [final-image CI run 29451910377](https://github.com/m2lines/Samudra/actions/runs/29451910377)
  built and import-smoked the Rust extension on x86_64 and arm64, then passed
  the arm64 CPU and x86_64 CPU/GPU container suites
- all pre-commit hooks pass, including Ruff, mypy, schema validation, secret
  detection, and REUSE lint

OM4-compact follow-on evidence:

- `361 passed, 2 skipped, 71 deselected, 10 xfailed` for the full non-CUDA suite
  with CUDA hidden
- `59 passed, 2 deselected` in focused Rust adapter and native integration tests
- five Rust unit tests and warning-free clippy
- [grouped-read image CI run 29458419589](https://github.com/m2lines/Samudra/actions/runs/29458419589)
  built and import-smoked the extension on x86_64, passed the arm64 build and
  CPU suite, and passed the x86_64 CPU/GPU post-publish suites
- all pre-commit hooks pass

## Release decision

**Go for opt-in local flat and compact OM4 training release.** Single-GPU and
DDP schedule and loss parity, validation, step transitions, checkpoint/resume,
throughput, bounded prefetch memory, CUDA overlap, packaging, and clean-shutdown
gates pass. Keep the loader opt-in and retain the CPU/GPU paths as immediate
fallbacks. LLC, remote storage, and inference remain explicit follow-on work
rather than release blockers for this path.
