<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2 2-degree structured latent-transport search

## Status

The immutable search was submitted on 2026-08-20. The `spatial-grid2` optimizer
probe is queued behind the account's four active GPUs; its release controller
will submit the candidate array only after verified optimizer progress. This
notebook recorded the design before inspecting forecast results.

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
- [ ] pass a real 2-degree Slurm optimizer probe through `spatial-grid2`;
- [ ] confirm online W&B registration and public worker lifecycle artifacts.

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
- Slurm probe and controller: `16063700` and `16063701`; the probe was initially
  pending on `QOSGrpGRES` behind the active pixel-dealiasing search.
- Slurm candidate array: Pending.

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

## Results and discussion

Pending.

## Conclusions and future work

Pending.
