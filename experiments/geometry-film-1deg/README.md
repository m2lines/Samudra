<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Gaussian geometry conditioning: 1 degree, seed 15

Experiment/results record, not intended to merge. No scientific results yet.

Compare scratch training of the original convolutional Samudra U-Net with no
geometry, input-concatenated geometry, and spatial FiLM after every encoder,
bottleneck and decoder core block. Only this single-resolution, single-seed
pilot is authorized. Multi-resolution follow-up is not submitted.

## Matched configuration and historical reference

Historical control: stochastic-depth PR #915, Beta job 210664, seed 15,
`current-1deg-control-ga2-210664`. Original configs are preserved in `reference/`.
The source of those files is
`beta:/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006/control/`.

Fresh control and interventions use the same current source, pinned container,
data and native reader. Fixed settings: 70 epochs, seed 15, batch 4/GPU,
4 GPUs and accumulation 2 (effective batch 32), LR 0.0006 cosine, four training
rollout calls, two input and two output states/call, dynamic MSE, bfloat16,
checkpointing all, no output residual, no stochastic depth or shortcut dropout,
no learned positional channels or 3-D coordinates. Boundary inputs are `tau_hfds`
(raw heat flux and wind stresses); no anomaly channel. Rust loading retains
8 concurrent reads, prefetch 2 and prefetch-to-device. Normalization and EMA
settings are unchanged.

Historical training ended 2013-10-05, overlapping the first validation date.
Current code rejects overlapping intervals. All three fresh arms instead end
training on 2013-09-30; validation remains 2013-10-05 through 2014-10-05.
This is also the current preset's split. API migrations are
`checkpoint_validation_metric` -> `validation_score`, `inference_times[0]` ->
`inference_time`, and removal of the obsolete empty `inference_epochs` setting.
The old run is a contextual reference, not the matched causal comparator.

## Geometry and initialization

Use L0 = 100,000 m and Earth radius 6,371,000 m. Validate the actual latitude
centers against Gaussian quadrature nodes and longitude uniformity. Quadrature
weights define cell edges in sin(latitude). dy is the meridional edge distance;
dx is the zonal arc at the midpoint in sin(latitude). Inputs are log(dx/L0),
log(dy/L0). At each 2x2 pooling stage, merge actual cell boundaries and discard
trailing odd cells exactly like pooling; the decoder uses matching skip-scale
geometry. Actual 180x360 dataset coordinates were verified locally.

FiLM uses a shared 2->16->16 pointwise SiLU encoder, with separate zero-initialized
16->2C heads for each of nine core blocks. Apply h'=(1+gamma)h+beta after each
block, before skip storage/addition as appropriate. Geometry is supplied even
on land; the usual final ocean mask remains unchanged. Only static maps are
cached; trainable encoder outputs are recomputed for autograd.

Concatenation widens the first spatial convolution and residual projection by
two input channels, with new columns initialized to zero. Preserve their hidden
widths and all original weights. Additional initialization preserves the CPU RNG
state, so common backbone and final decoder parameters match the control.

Parameter counts: control 84,002,154; concat 84,008,474 (+6,320, 0.0075%);
FiLM 84,124,874 (+122,720, 0.146%). A single-grid gain cannot distinguish physical
geometry learning from latitude-dependent corrections.

## Evaluation and overhead

Retain fixed 360-day validation RMSE and the historical end-of-training evaluation
protocol: epoch 70 plus final EMA, 2014-10-10 through 2022-12-24, 598 forecast
times. Do not select an arm using dynamic weighted loss. Save full prediction
stores, then compute physical-unit RMSE, bias, prediction RMS, absolute maximum,
and nonfinite wet-cell counts by channel, lead and six 30-degree latitude bands
plus global. Use Gaussian area weights and fixed wet masks; nonfinite predictions
are counted and propagate to errors rather than being silently removed.

The batch harness runs full-shape, batch-4, four-call forward/backward CUDA
qualification before starting DDP training. It records five compute-only timing
samples after three warm-ups, parameter count and peak allocated memory.
This excludes optimizer, DDP and I/O. Compare actual steady-state training batch
and epoch times separately from this microbenchmark. Bring-up is pending queue
allocation; CPU tests alone do not establish GPU/runtime success.

## Runtime and provenance

Beta root: `/projects/ny/lz1955/multiscale/jrusak/runs/geometry-film-1deg-20261023`.
Data root: `/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg`.
This is the resolved project-storage target of the historical home-directory
symlink, so no dataset copy/substitution is involved.

Container: `ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4`.
Native loader copied from the historical control runtime. Source and native
checksums are verified before launch; allocation, code commit, W&B URLs, resolved
configs and Slurm logs are retained. W&B is online in
`ocean_emulators/default`, group `geometry-film-1deg-seed15`.

Availability on 2026-10-23: Torch's four-GPU test submission estimated a later
start, Engaging's H200/B200 tests estimated the following day; no immediate
alternative allocation was established. Use production Beta as requested.

Durable local six-hour scheduler checks must verify accounting, training
completion, both prediction stores and latitude/stability metrics for every arm
before publishing a scientific comparison.

Remote callback credential installation was blocked by automatic approval review.
No remote callback is installed; local quiet polling is the active mechanism,
so completion/failure detection can lag by up to six hours.
