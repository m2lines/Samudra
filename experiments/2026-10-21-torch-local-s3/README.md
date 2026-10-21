<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Quarter-degree training from Torch scratch versus OSN

This experiment tests whether direct training from the NYU OSN pod is practical
for full-resolution Samudra. Both arms use the same losslessly compressed dataset,
Rust loader, model, seed, sampler, and eight RTX6000 GPUs on the same Torch node.
This is an experiment/results record, not a scientific model-quality comparison
or a merge candidate in its current form.

## Data and transfer

- Local: `/scratch/jr7309/data/om4_quarterdeg_v2/`.
- Remote: `s3://emulators/jr7309/data/testing/om4_quarterdeg_v2_zstd_20261021/`.
- Endpoint: `https://nyu1.osn.mghpcc.org`.
- Both contain `OM4.zarr`, `OM4_means.zarr`, and `OM4_stds.zarr`.
- Physical arrays retain float32 and `(1,720,1440)` chunks, using Blosc Zstd
  level 5, byte shuffle. This is the existing November 2025 prepared dataset,
  not the newer September 2026 snapshot release.
- Prior compression verified 379,600 chunks across 80 arrays, reducing payload
  from 1,574,277,120,000 to 616,584,025,301 bytes.
- `transfer.sh` runs on `dtn011`, copies with rclone, performs a full
  `check --download`, and compares complete source/destination counts and sizes.

## Matched protocol

`benchmark.py` uses the ordinary `Trainer.train_one_epoch` implementation.
The only benchmark hooks capture existing metrics and stop all ranks at a shared
wall-clock budget after a complete batch. No validation or checkpoint writes are
included in the measurement. Each arm starts a fresh distributed Python process.

- Model: repository `samudra_om4/train.yaml` and its full ConvNeXt U-Net preset,
  instance normalization, bfloat16, activation checkpointing, dynamic MSE loss
  with ratio cap 20. All 77 prognostic channels and physical `tau_hfds` forcing.
- Batch size: 1 per rank, 8 global; gradient accumulation 1; four autoregressive
  training steps. Input/output steps retain the preset defaults.
- Training period: 1975-01-03 through 2013-09-30, identical shuffled sampler.
- Seed: 20261021. Initial model-state hashes and first-epoch sampler hashes
  recorded separately for every rank and checked across arms.
- Loader: Rust, 8 concurrent reads per rank, prefetch queue 2, device prefetch.
- Five warm-up batches, then 3,600 measured seconds per arm. Stop at the end of
  a synchronized batch. Startup duration is reported separately.
- Local arm first, remote arm second, same allocation and node. Standard OS and
  server caching may occur; caches are not forcibly flushed. This is not a
  controlled cold-cache benchmark. Neither arm uses application disk staging.
- W&B disabled; per-rank batch JSONL and summaries retained on scratch.
- GPU utilization, memory utilization, allocated memory, and power sampled every
  five seconds in separate files for each arm. These include startup/warm-up;
  use batch timing boundaries when reporting steady-state utilization.

The S3 backend uses `zarrs_object_store` 0.4 with the existing zarrs 0.21 reader,
an async-to-sync adapter, persistent connections, three bounded retries, and
two process-wide Tokio network threads. The same Rayon pool bounds data reads
and decompression. Metadata and normalization still go through Xarray at startup.
There is no local chunk cache in the S3 arm. Retry counts are not separately
instrumented; retry-induced stalls are included in wall time.

## Jobs and verification

Initial source archive SHA-256:
`8cc47db874a17f3649a9fe25bdf97a3396ce5d0f47df1cb25938bf0a6317c653`.
The benchmark and summary scripts are deployed separately and recorded alongside
the archive; credentials are never included.

Working directory: `/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021`.

- Transfer supervisor: PID 2436127 on `dtn011`.
- Initial CPU build 20560296 failed before compilation because the target-installed
  maturin executable was absent from PATH; build script corrected.
- CPU build 20560497 compiled the wheel, then failed on an unrelated automatic
  pytest plugin import (`zarr.testing`). CPU build/test retry: 20560851, with
  explicit plugin selection.
- Transfer and actual-data parity gate: 20560858, after successful build.
- Matched GPU training: 20560884, after successful parity gate.

CPU job 20560851 completed successfully: 28 native-extension tests and 164
Samudra config/native-loader tests passed (11 CUDA/manual tests deselected).
Local formatting/lint checks and `cargo check` also passed.

The native extension suite covers local/S3 equality for flat and compact stores,
anonymous and signed requests, access-denied failures, and empty selectors.
The CPU gate additionally compares all 80 physical fields at three times and all
normalization fields from the actual uploaded stores. No GPU allocation starts
until build tests and full transfer verification succeed.

Expected artifacts: `TRANSFER_COMPLETE.json`, `PREFLIGHT_COMPLETE.json`,
`runs-20560884/{local,remote}/COMPLETE-rank*.json`, per-rank batch JSONL,
GPU telemetry, and `runs-20560884/COMPARISON_COMPLETE.json`.
The initially queued dependency jobs 20560539 and 20560552 were cancelled
without GPU execution after the test-runner failure.

Final throughput, data-wait distributions, matching common-prefix losses, and
telemetry will determine the conclusion; no training result is available yet.
