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
