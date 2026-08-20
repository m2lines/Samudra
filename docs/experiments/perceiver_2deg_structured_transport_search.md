<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2 2-degree structured latent-transport search

## Status

The immutable search was submitted on 2026-08-20. The `spatial-grid2` optimizer
probe passed with 32 accumulated batches and one finite optimizer update, then
released the nine-candidate array. Eight candidates completed epoch 1 and four
were promoted. All promoted candidates completed epoch 2 but reached the
two-hour rung wall limit before epoch 3, so the search stopped without an
eligible second rung. The partial results are scientifically informative but do
not constitute a completed successive-halving search.

## Motivation

The completed primary search selected direct output queries and found only a
2.2% spread among its three epoch-12 finalists. Changing decoder context,
transport width, or generic encoder latent count therefore appears lower value
than repairing the representation contract itself.

Jesse's experiments identified the larger failure: one mean-pooled vector per
physical patch loses spatial phase before the processor sees it. His structured
coarse representation showed that explicit resolved means and phase-sensitive
moments can retain useful dynamics, but remained behind a native-grid learned
inverse. This search asks whether Perceiver-based, spatially addressable, or
fixed-basis transports can narrow that gap while preserving a scalable coarse
processor grid.

## Research questions

1. Does retaining several ordered Perceiver query outputs improve forecast loss
   over mean-pooling the encoder latent bank?
2. Is it better to pack ordered outputs into channels, or keep them as an
   explicit spatial latent grid through the processor?
3. Can grouping a larger physical region into a coordinate-tied latent grid
   retain the control processor resolution while reducing input-attention
   boundaries?
4. Does an explicit area-weighted mean plus learned anomaly moments outperform
   a generic Perceiver summary?
5. Does a fixed orthonormal mean/detail basis provide a stronger phase route
   than learned moments?
6. Are improvements visible in validation loss without increasing seam tails or
   periodic patch modes?

## Hypotheses

### H1: spatial topology matters more than latent count

The explicit spatial-grid candidates will outperform both the 64-latent
mean-pooled baseline and the packed-query control. Ordered queries that remain
spatial axes give the convolutional processor an unambiguous phase contract;
packing the same queries into channels tests whether ordering alone is enough.

### H2: the resolved mean is a useful physical anchor

The learned-moment encoders will improve early optimization and amplitude
retention because their area-weighted mean bypasses normalization and provides
a direct resolved-state route.

### H3: fixed detail modes are a useful but seam-sensitive representation

The DCT candidates will distinguish equal-mean patterns by construction and
should retain more patch-scale spectrum. Their patch-local synthesis may produce
boundary disagreement, so they advance only if loss improves without unacceptable
window- or patch-frequency tails.

### H4: more generic capacity is not the solution

The width-256 decoder bridge may learn faster, but it should not dominate every
structured representation at the final budget. Likewise, 16 moments or a full
14-mode patch basis should help only if the extra coefficients carry useful
phase rather than make optimization harder.

## Architecture interventions

All candidates use residual prediction, one-ring overlap-add where the direct
decoder applies, zero anonymous decoder context, the same 128-channel processor
interface, learning rate `4e-4`, and the public 2-degree OM4 data.

| Candidate | Encoder/latent contract | Decoder contract | Mechanism |
| --- | --- | --- | --- |
| `mean64-direct` | Current 64-latent mean-pooled patch Perceiver | Direct queries, width 128 | Efficient primary-search baseline |
| `mean64-wide` | Same as baseline | Direct queries, width 256 | Bridge to the fast-learning prior finalist |
| `packed-query2` | Four coordinate-conditioned Perceiver outputs packed into 128 channels per patch | Direct queries | Ordered-information control without spatial axes |
| `spatial-grid2` | A 12 x 20 degree input group produces a 2 x 2 coordinate-tied latent grid | Direct queries at the resulting 6 x 10 degree latent spacing | Topology-preserving grouped Perceiver |
| `spatial-grid4` | A 24 x 40 degree input group produces a 4 x 4 coordinate-tied latent grid | Same 6 x 10 degree latent spacing | Larger-group hierarchical/locality test |
| `resolved-moment4` | Area-weighted mean plus four learned continuous anomaly moments | Direct queries | Minimal resolved/anomaly split |
| `resolved-moment16` | Area-weighted mean plus 16 learned moments | Direct queries | Learned phase-capacity test |
| `dct-detail4` | DC coefficient plus four fixed low-frequency DCT detail coefficients | Paired DCT patch synthesis and identity-initialized pixel refinement | Small fixed mean/detail representation |
| `dct-detail14` | Complete 3 x 5 patch DCT basis | Paired complete-basis synthesis and refinement | Information-rich fixed-basis ceiling |

`spatial-grid2` is the first candidate and therefore the real-data optimizer
probe. It exercises the most important new encoder, the native SDPA PerceiverIO
path, the finer latent geometry passed to the decoder, overlap-add, backward,
and optimizer state before the array is released.

## Deliberate omissions

The GINO-like local operator bridge and native-grid anchor are not forecast arms
in this first implementation. The former needs a channel-mask-aware physical
transport contract and source-grid coordinates through `SamudraMulti`; Jesse's
branch contains those pieces, but mixing that larger temporal/boundary refactor
into this search would confound the representation question. Jesse's completed
native-grid result remains the diagnostic ceiling. A local-operator arm should
be added after this search if explicit spatial grids win or fixed patch grouping
remains the dominant artifact.

The `dct-detail` arm is a single-level mean/detail spike, not yet the full
masked lifting pyramid proposed in the synthesis. It tests whether a fixed,
phase-addressable paired basis is promising before implementing multiple
levels, conservative wet-mask restriction, and detail-state skip paths.

## Correctness and representation gates

Before launch:

- [x] prove the DCT basis is orthonormal and distinguishes a constant field from
      an equal-mean checkerboard;
- [x] prove the spatial-grid encoder leaves query outputs as spatial processor
      axes and reports the corresponding physical decoder extent;
- [x] run forward/backward tests for spatial-query, spatial-grid, DCT encoder,
      and paired DCT decoder paths;
- [x] validate and instantiate all nine resolved candidate models;
- [x] pass the full standard test suite (411 passed, 2 skipped, 10 expected
      failures);
- [x] pass a real 2-degree Slurm optimizer probe through `spatial-grid2`;
- [x] confirm online W&B registration and public worker lifecycle artifacts.

The Slurm probe is a training-correctness gate, not evidence that a
representation is scientifically adequate. Equal-mean front counterfactuals,
spectral retention, and reconstruction should be computed from trained
checkpoints before interpreting a forecast winner as a v2 architecture.

## Search allocation

Successive-halving budgets are cumulative epochs `[1, 3, 6, 12]`, with half of
the competing candidates promoted and at least two retained. Each worker asks
for one RTX6000, four CPUs, and 32 GiB. Concurrency is capped at four to match
the observed Torch account limit rather than submitting an ineffective
eight-GPU request.

The promotion objective is autoregressive validation MSE. Seam jump ratios and
periodic-mode power means, p90s, and maxima are retained as guardrails. Final
selection also requires matched images and short rollout diagnostics; promotion
loss alone does not establish spectral or stability quality.

## Launch record

- Search run:
  `perceiver-structured-transport-2deg--20260820T035545.388436Z`.
- Git revision: `005f952cb4c8ef1b9ce7aca280b48fe6470eaa3d`.
- Code-layer SHA-256:
  `4931762355174d86a2598838c6c8e47a6a9dbe56f0ba3f082cb7a06996333535`.
- W&B group:
  `perceiver-structured-transport-2deg--20260820T035545.388436Z`.
- Public artifacts:
  [OSN search record](https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-structured-transport-2deg--20260820T035545.388436Z/).
- Slurm probe and controller: `16063700` and `16063701`; the probe passed after
  its initial `QOSGrpGRES` wait.
- Rung-0 candidate array and controller: `16070267` and `16070268`.
- Rung-1 candidate array and controller: `16075113` and `16075114`.

## Query templates

<details>

<summary>Latest candidate ranking and artifact guardrails</summary>

```sql
SELECT
    candidate,
    rung,
    epochs,
    eligible,
    validation_loss,
    train_loss,
    "val/seam/window_jump_ratio/channel_p90" AS window_jump_p90,
    "val/seam/window_jump_ratio/channel_max" AS window_jump_max,
    "val/seam/patch_jump_ratio/channel_p90" AS patch_jump_p90,
    "val/seam/patch_periodic_power_ratio/channel_p90" AS patch_power_p90,
    optimizer_steps,
    worker_stage,
    worker_error
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-structured-transport-2deg--20260820T035545.388436Z/results.parquet'
)
ORDER BY rung DESC, eligible DESC, validation_loss ASC NULLS LAST;
```

</details>

<details>

<summary>Learning curves and late crossovers</summary>

```sql
SELECT
    candidate,
    epoch,
    validation_loss,
    train_loss,
    "progress/optimizer_steps" AS optimizer_steps,
    round(epoch_train_seconds / 60, 2) AS train_minutes,
    round(epoch_validation_seconds / 60, 2) AS validation_minutes
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-structured-transport-2deg--20260820T035545.388436Z/epochs.parquet'
)
ORDER BY candidate, epoch;
```

</details>

## Results

### Rung 0: one epoch

Eight of nine candidates completed 89 optimizer steps. `spatial-grid4` failed
before training because its 12-row physical group does not divide the 90-row
2-degree grid. The published worker log reports `ValueError: Input grid must
divide evenly into Perceiver groups.` This is an invalid experiment geometry,
not evidence against a 4 x 4 spatial latent grid.

| Candidate | Validation loss | Train minutes | Window jump mean | Window jump p90 | Patch jump mean | Window power mean | Patch power mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dct-detail14` | **0.068514** | 90.9 | **0.970** | **1.033** | 1.015 | **0.00625** | 0.001892 |
| `dct-detail4` | 0.069505 | 115.1 | 1.213 | 1.531 | 1.441 | 0.00740 | 0.001313 |
| `mean64-direct` | 0.075475 | 24.8 | 1.044 | 1.397 | 1.020 | 0.01453 | **0.000344** |
| `mean64-wide` | 0.075477 | 24.6 | 1.018 | 1.270 | 1.017 | 0.01488 | 0.000347 |
| `resolved-moment16` | 0.075492 | 115.1 | 1.025 | 1.337 | 1.017 | 0.01405 | 0.000349 |
| `resolved-moment4` | 0.075495 | 115.3 | 1.034 | 1.292 | 1.017 | 0.01499 | 0.000345 |
| `spatial-grid2` | 0.075497 | 23.9 | 0.872 | 1.062 | **0.978** | 0.02608 | 0.001320 |
| `packed-query2` | 0.075500 | 101.3 | 1.046 | 1.316 | 1.020 | 0.01855 | 0.000338 |

The DCT candidates were the only arms to separate from the tightly clustered
mean-pooled controls after one epoch. `dct-detail14` improved validation loss by
9.2% relative to `mean64-direct`; `dct-detail4` improved it by 7.9%. The other
six learned transports lie within 0.000026 loss, a spread of only 0.035%.

### Promoted candidates: two observed epochs

The four promoted candidates resumed from epoch 1. Each completed epoch 2, but
the two-hour rung allocation expired before any completed the target epoch 3.
These epoch-2 records are comparable even though the harness correctly marked
all four rung results ineligible.

| Candidate | Epoch-2 validation loss | Improvement vs. `mean64-direct` | Window jump mean | Patch jump mean | Patch power mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dct-detail14` | **0.057458** | **23.8%** | **0.978** | 1.021 | 0.001405 |
| `dct-detail4` | 0.062409 | 17.2% | 1.201 | 1.431 | 0.001381 |
| `mean64-direct` | 0.075375 | -- | 1.040 | 1.018 | **0.000338** |
| `mean64-wide` | 0.075405 | -0.04% | 1.018 | **1.016** | 0.000347 |

`dct-detail14` improved rapidly from 0.068514 to 0.057458. Relative to the
epoch-2 baseline, its window-jump mean, p90, and maximum were 6.0%, 24.9%, and
38.3% lower. Its patch jump was essentially tied, but its patch-periodic power
mean was 4.15 times larger. `dct-detail4` combined better loss with clearly bad
patch boundaries: its patch-jump mean was 40.6% above the baseline and its p90
was 86.0% above it.

## Analysis and discussion

### Fixed phase-addressable transport is the strongest lead

The primary hypothesis is supported, but more specifically than anticipated.
Increasing generic width, packing ordered Perceiver queries into channels, and
adding learned anomaly moments did not improve early loss. In contrast, both
fixed DCT routes learned substantially faster. The full 3 x 5 basis was better
than four selected detail modes and avoided the latter's boundary-jump failure.
This points to information preservation and an explicit paired analysis/
synthesis contract—not raw latent count—as the important ingredient.

The result also refines Jesse's root-cause analysis. Mean pooling erases phase;
simply supplying more learned summaries did not recover useful phase quickly.
A complete, fixed, spatially indexed basis made that information immediately
available to the processor and decoder. The strong difference between 14 and
4 detail modes suggests that truncating local modes can create inconsistent
patch reconstructions even when aggregate forecast loss improves.

### Lower loss does not yet mean artifact-free

`dct-detail14` is the unambiguous loss winner and has excellent jump metrics,
but it concentrates 4.15 times more error power at the measured patch frequency
than `mean64-direct`. The metric is detecting periodic structure not expressed
as a literal boundary discontinuity. This is exactly why the search
pre-registered both jump and spectral guardrails. The DCT path should advance,
but only with matched images, spectra, and rollout trajectories; it is not yet
the validated v2 architecture.

`spatial-grid2` provides a different, useful signal. Its epoch-1 loss was tied
with baseline, while its window- and patch-jump means were 16.5% and 4.1% lower.
Its periodic-power metrics were worse and its maximum window tail remained
large. Preserving spatial axes therefore changes artifact shape before it
changes MSE. Because it ran as fast as the baseline, it remains a scalable
candidate worth training longer rather than eliminating from a one-epoch tie.

### Capacity and learned-summary hypotheses are not supported early

`mean64-wide` was indistinguishable from `mean64-direct` through epoch 2, so
decoder bridge width alone is not the bottleneck. The two resolved-moment arms
and packed-query arm were both slower and no better after one epoch. This does
not prove they can never cross over, but it makes them lower-priority than the
DCT and topology-preserving routes for the next budget.

### Search execution limited the inference

The promoted jobs each spent about 108 minutes on epoch 2, leaving insufficient
time for epoch 3 under a two-hour allocation. The failure status therefore
reflects time-budget underallocation, not model divergence. Rung-0 runtime also
varied from about 24 to 115 minutes by architecture. Future promotion wall times
must account for the number of *additional* epochs and the slowest promoted
architecture. Geometry divisibility should also be validated from the resolved
dataset shape before array submission.

## Conclusions

1. Promote a complete fixed mean/detail analysis-synthesis route as the leading
   Perceiver v2 direction. `dct-detail14` reduced epoch-2 loss by 23.8%.
2. Do not select it on loss alone: its patch-periodic error power was 4.15 times
   baseline and requires visual and rollout validation.
3. Do not pursue an incomplete four-detail basis in its current form. Its patch
   boundary metrics were substantially worse despite improved loss.
4. Keep `spatial-grid2` as the efficient topology-preserving alternative. It
   improved mean jump metrics at baseline runtime, but needs more training and
   spectral control.
5. Deprioritize wider decoder bridges, channel-packed query outputs, and
   learned moments until the two stronger representation routes are resolved.
6. Treat `spatial-grid4` as untested; its configured group geometry was invalid.

## Future work

1. Run a focused, matched-budget search with `mean64-direct`, `spatial-grid2`,
   and `dct-detail14` for at least 6 epochs. Add two DCT controls: a smaller
   *complete rectangular* basis and DCT-14 with cross-patch/global refinement.
2. Add a hybrid spatial-grid plus fixed-detail candidate. This directly tests
   whether topology-preserving transport can retain DCT's skill while reducing
   its periodic modes.
3. Make wet-mask-aware, conservative restriction and prolongation explicit.
   Evaluate equal-mean fronts and spectral reconstruction before forecast
   training to verify that useful phase survives the transport.
4. Run identical 1-, 5-, 10-, and 20-step evaluations with physical RMSE,
   gradient retention, seam tails, patch-frequency power, and matched-scale
   images. Reject candidates whose periodic mode grows autoregressively.
5. Repair `spatial-grid4` with a group shape that divides 90 x 180 (or pad with
   an explicitly tested geographic boundary contract), then compare it only if
   `spatial-grid2` improves with longer training.
6. Update search preflight to resolve dataset geometry and estimate
   architecture-specific epoch time. Allocate every rung for its incremental
   epochs plus validation and publication margin.
