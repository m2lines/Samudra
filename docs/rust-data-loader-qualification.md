<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: Apache-2.0
-->

# Rust data loader production qualification

This report records the Stage 5 evidence for the opt-in local flat-OM4 Rust
loader described in [the shipping plan](rust-data-loader-plan.md).

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

Pending the matched two-GPU Torch runs from the published commit image. The
release decision remains **no-go** until rank-local schedule parity, loss parity,
checkpoint/resume, and clean shutdown pass there.

## Test evidence

- `330 passed, 2 skipped, 71 deselected, 10 xfailed` for
  `pytest -m "not manual and not cuda"` with CUDA hidden
- `30 passed` in the sampler suite
- `18 passed` in non-CUDA Rust data tests
- `2 passed` in focused CUDA prefetch and pinned-buffer reuse tests
- all pre-commit hooks pass, including Ruff, mypy, schema validation, secret
  detection, and REUSE lint

## Release decision

**Current status: no-go pending DDP.** Single-GPU correctness, checkpoint/resume,
bounded pinned memory, clean shutdown, and overlap gates pass. Update this section
after the two-rank comparison; do not advertise the Rust extra as production-ready
until that final gate passes.
