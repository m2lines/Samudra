<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# A with 1° latent states and 64 channels

User-authorized separate comparison with A-extend: train from scratch, seed1729,
4,000 mixed updates (50% OM4 / 50% OBS), then 1,000 OBS-only cooldown updates.
Effective batch eight, LR1e-4 in main training, linear cooldown to1e-6 with EMA
reset at the boundary, EMA half-life100, clip1. No replay or latent bound.

Keep A-extend's data windows, normalization, objectives, evaluation, native OM4
stress/heat-flux forcing and learned OBS ERA5 adapter. This is independent of
the shared-ERA5 experiment and does not require the earlier ERA5 preparation.
The original A-extend starts with2k mixed updates and continues another2k;
this run reproduces that4k-main schedule in one uninterrupted fresh trajectory.

Change the retained/autoregressed latent grid from45×90 to180×360 and channel
width from128 to64 **per slot**, retaining both memory slots. Remove the final
4× initializer averaging; the processor receives forcing/context at native1°
resolution. Keep processor depth4, initializer backbone widths64/128/256,
decoder width192 and the same32-step physical diffusion decoder. Input/output
projections necessarily change with latent width. No new physical labels for
latent channels are introduced.

This is not parameter- or compute-matched: latent scalar count increases8×,
from1,036,800 to8,294,400 per example. Processor convolutional receptive field
covers a smaller geographic extent, although initialization and decoding still
have their existing broader receptive fields. Parameter count, peak memory,
actual optimizer-update throughput and serialization equivalence are recorded
by qualification before production is released. Same seed does not make
initial weights identical across different architectures.

Selection/reporting: original nine-origin RMSE-only blend, component curves,
raw and EMA endpoints, fixed4k+1k comparison with A-extend. Retain4k pre-cooldown
checkpoint. Follow with matched annual rollouts and single-draw/ensemble maps,
including supervised OM4 denoising trajectories to assess grain separately
from the observation aggregate score.

Runner: `scripts/run_monthly_replay.py`, mode `fresh-4k`, architecture
`{"latent_width":64,"latent_stride":1}`. Existing modes/default model shapes
remain unchanged. All real data, training and full-size qualifications run on
Beta; local checks use tiny synthetic unit-test tensors only.
