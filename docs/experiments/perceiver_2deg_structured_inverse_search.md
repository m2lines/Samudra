<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Structured Perceiver inverse search at 2°

## Questions and hypotheses

This search asks which single-scale encoder/decoder best combines Samudra's
native-SDPA Perceiver routing with the strongest findings from Jesse's coarse
latent/high-resolution dynamics experiments.

1. Does an explicit area-weighted mean plus phase-sensitive moment state retain
   more forecast-relevant detail than pooled or spatial-query Perceiver states?
2. Was adding absolute geometry directly to moment content harmful, compared
   with keeping routing geometry separate?
3. Does a smooth coordinate-resampled base plus zero-initialized,
   position-anchored local SDPA residual improve loss, seams, spatial gradients,
   and rollout RMSE relative to direct output cross-attention?
4. How much moment capacity is needed, and does a denser spatial query grid
   provide a better capacity tradeoff?

The main hypothesis is that `moment16-local` will provide the strongest
combination: the encoder guarantees a resolved route and retains subpatch phase,
while the decoder begins from a smooth coarse field and learns only a local
continuous correction. The search also includes causal encoder-only and
decoder-only comparisons, the strongest prior spatial-grid decoder, a paired DCT
route, and the historical pooled Perceiver control.

## Interventions

All ten candidates share the same 2° OM4 split, seed, 60×72-or-finer processor
token budget, ConvNeXt processor, residual target, optimizer, and 1/3/6/12 epoch
successive-halving schedule. Promotion uses validation loss. Each rung endpoint
also runs the configured autoregressive validation rollout; rollout RMSE,
velocity RMSE, gradient-magnitude fidelity, and seam diagnostics are retained for
scientific review and Pareto analysis rather than collapsed prematurely into one
score.

Encoder group extents are chosen to make the processor grid comparable rather
than naively held equal: 2×2 spatial-query groups cover 12°×20°, 4×4 groups
cover 24°×40°, and single-token moment, DCT, and pooled routes cover 6°×10°.
At 2° all three choices produce a 30×36 token grid.

Each Alpha worker uses two GPUs with DDP. This resource shape had an immediate
`test`-QoS scheduler estimate on launch day, while Alpha's one-GPU RTX pool was
backlogged until the following day; it also keeps both allocated GPUs doing
model work rather than reserving idle capacity.
Gradient accumulation is 16, preserving the prior effective batch of 32
(`batch_size=1 × accumulation=16 × world_size=2`).

Live scheduler probes showed that CPU shape, rather than GPU or memory demand,
controlled placement on the partially occupied H100 nodes: four CPUs with 28
GiB could backfill immediately, whereas 6--12 CPUs moved the same two-GPU job
behind a later reservation. The final search therefore uses two loader workers,
four CPUs, and 28 GiB per two-GPU candidate.

The candidate matrix is defined in
[`search.yaml`](../../src/samudra/configs/perceiver_structured_inverse_2deg/search.yaml).
The exact launch commit, search run ID, Slurm jobs, and W&B group will be added
once the Alpha correctness probe reaches its first optimizer update.

## Results

Pending.

## Analysis

Pending.

## Conclusion and future work

Pending.
