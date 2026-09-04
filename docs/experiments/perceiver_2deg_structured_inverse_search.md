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

### Torch relaunch revision after prior-evidence review

Before the Torch relaunch, the Perceiver lab notebooks and Jesse's decoder and
coarse-latent reports were reviewed together. Two original Alpha candidates are
not repeated in the Torch matrix:

- `dct14-paired` is excluded because its low short-budget MSE coincided with
  4.15 times the baseline patch-periodic power and visibly basis-aligned,
  unphysical error texture. It remains mechanistic evidence that spatial phase
  matters, not a promotion-eligible architecture.
- `moment16-geometry-direct` is excluded because adding absolute geometry to
  reconstructive content conflicts with the stronger learned-inverse result:
  keep amplitude-bearing state separate and inject geometry as a
  zero-initialized processor sidecar.

The revised Torch search therefore contains eight candidates: the pooled
Perceiver baseline; spatial-grid 2x2 direct and local-decoder arms; a denser
spatial-grid direct arm; moment-4 and moment-16 local arms; and moment-16 direct
width-128 and width-256 arms. On a four-GPU allocation this produces an exact
8 -> 4 -> 2 -> 2 occupancy schedule: four concurrent one-GPU candidates in the
first two rungs and two concurrent two-GPU candidates in the last two rungs.
The effective global batch remains 32 throughout.

The utilization smoke is a systems gate rather than a scientific comparison.
It crosses the two leading phase-preserving encoders (`spatial_grid` and
`patch_moment`) with direct and physically anchored local decoders. A full
search will not be released unless all four arms make finite optimizer progress
and every allocated GPU sustains more than 50% utilization during steady-state
training. Validation loss from this shortened data window will not select an
architecture.

## Launch record

The accepted immutable run is
`perceiver-structured-inverse-2deg--20260831T191355.834091Z` at commit
`5a7fc7716df710c162f2b6cdfab0b4413ea4f772`. The W&B group is the same run ID.

- Alpha launch controller: `56021` (`COMPLETED`)
- two-GPU DDP correctness probe: `56023_0` (`COMPLETED`)
- probe-release controller: `56024` (`COMPLETED`)
- rung-zero candidate array: `56030` (tasks 0--9, concurrency 4)
- rung-zero advancement controller: `56031` (`afterany:56030`)

The probe loaded the public 2° data, built the complete model, ran both DDP
ranks, consumed 16 microbatches, and recorded one real optimizer update with a
finite loss of 2.3714. Its steady-state microbatch time was approximately
0.6--1.5 seconds and peak memory approximately 3.4 GiB per GPU. Rung zero was
released only after this evidence was durable.

At the first status check, array tasks 0--2 were concurrently training
`spatial-grid2-direct`, `moment16-geometry-direct`, and `moment16-direct`.
All three recorded a finite first batch; the geometry-moment candidate had
already recorded two optimizer updates. Remaining tasks backfill as one of the
four concurrent slots becomes available.

## Results

Pending.

## Analysis

Pending.

## Conclusion and future work

Pending.
