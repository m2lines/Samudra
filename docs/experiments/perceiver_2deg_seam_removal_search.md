<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver 2-degree decoder seam-removal search

## Status

This lab notebook records the hypotheses and design for a focused follow-up to
the first Perceiver v2 architecture search. They were written before launching
the search or inspecting any candidate results. The experiment will train on
the public 2-degree OM4 dataset and publish W&B runs, scalar histories, worker
state, scheduler logs, resolved configs, and finalist checkpoints.

The search was submitted on 2026-08-18 as
`perceiver-seam-removal-2deg--20260818T210753.565365Z` from immutable code
revision
[`f3eaead6`](https://github.com/m2lines/Samudra/tree/f3eaead66a77ee0a05b5724c86f9dbd68d6f251b).
Its live and eventual final artifacts are published under the
[`m2lines-pubs` search directory](https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-seam-removal-2deg--20260818T210753.565365Z/).

The search is complete. All seven candidates produced eligible results without
worker failures. The five competing candidates were screened at one epoch;
`blend1-context1-residual`, `blend1-no-context`, and `blend1-context1` advanced
through the three- and six-epoch rungs. Both fixed anchors completed their
six-epoch budgets. The final public controller state, result tables, resolved
configs, logs, and checkpoints are available from the artifact directory
linked above.

## Motivation and prior observation

The first search selected `direct-no-context-lr4` by validation MSE, but its
one-step validation error maps contain conspicuous rectilinear discontinuities.
The same structure appears in the other finalists. In the 2-degree `zos` error
images, the strongest repeated jumps align with the decoder's 18 x 30 pixel
output tiles rather than arbitrary ocean features:

- a 6 x 10 degree encoder patch occupies 3 x 5 grid cells;
- each decoder call owns a 6 x 6 patch core; and
- each hard output tile therefore covers 18 x 30 cells, or approximately
  36 x 60 degrees.

The current decoder overlaps input context when `context_patches > 0`, but it
does not overlap outputs. Every tile is produced by a separate attention call
over a different token set and is copied directly into a disjoint output
slice. Continuous absolute-position queries cannot guarantee continuity when
the attention support changes abruptly at a fixed boundary.

This experiment tests that diagnosis directly. It does not reopen the broader
encoder-capacity and decoder-transport search.

## Research questions

The primary question is:

> Can overlapping output-query windows with smooth reconstruction eliminate
> decoder-window artifacts without sacrificing one-step validation skill?

Supporting questions are:

1. Are the visually apparent seams quantitatively concentrated at the known
   decoder-window boundaries?
2. Is input context alone sufficient, or must neighboring output predictions
   overlap and blend?
3. Does one patch of output overlap provide most of the benefit, or is a
   two-patch halo materially better?
4. Does context improve once duplicate output predictions are blended, despite
   the previous hard-window search favoring zero context?
5. Does predicting residual tendencies preserve spatial continuity better than
   predicting the full next state?
6. What validation-loss and runtime cost accompanies each reduction in seam
   strength?

## Hypotheses

These hypotheses are recorded before training results are available.

### H1: hard output assembly is the dominant large-grid artifact

The hard zero-context anchor will have a decoder-window jump ratio materially
above one. Adding one input-context ring without output blending may reduce the
ratio slightly, but will not remove it because adjacent query tiles still use
different attention supports and are never reconciled.

### H2: overlapping output queries plus smooth blending will reduce seams

Decoding a one-patch query halo around every core and combining duplicate
predictions with normalized cosine weights will lower both `zos` and
channel-mean decoder-window jump ratios. The intervention should act directly
on the 18 x 30-cell discontinuities while preserving differentiability.

### H3: one overlap ring will capture most of the attainable improvement

A two-patch output halo may be smoother, but its larger query footprint should
have diminishing quality returns and higher runtime. One overlap ring is the
expected best loss/seam/compute compromise.

### H4: context becomes useful when routing disagreement is blended

The prior zero-context winner suggested that anonymous neighboring tokens made
hard-window routing more difficult. With overlapping outputs, a one-ring input
halo may become beneficial because boundary pixels receive spatial information
from both sides and inconsistent predictions are combined rather than selected
abruptly.

### H5: residual prediction will further suppress discontinuous reconstruction

The residual candidate should inherit the continuous input field and ask the
decoder to predict a smaller tendency. It may therefore reduce visible seams,
although its short-budget loss ordering can differ because this changes the
prediction parameterization rather than window assembly alone.

### H6: global shared context is a useful 2-degree gold standard, not the
long-term scalable solution

When every query tile sees the same 1,080 processor tokens, query chunking
should no longer introduce support discontinuities. This anchor is expected to
have low decoder-window jump ratio but substantially higher attention cost. It
tests the diagnosis while overlap-add remains the candidate suitable for much
larger grids.

## Experimental design

### Common model and data

Every candidate uses the selected architecture from the preceding search:

- public 2-degree OM4 data and the same train/validation split;
- random seed 15;
- four-step autoregressive training and one-step validation;
- batch size 1 with 32 accumulated batches per optimizer update;
- normalized MSE loss and learning rate `4e-4`;
- patch-local Perceiver encoder with 256 internal latents;
- 6 x 10 degree patches and a 30 x 36 processor-token grid;
- the same ConvNeXt U-Net processor;
- direct output-query cross-attention with two 64-dimensional heads; and
- PyTorch scaled dot-product attention with automatic backend selection.

The common model is
[`perceiver_seam_search_2deg/model.yaml`](../../src/samudra/configs/perceiver_seam_search_2deg/model.yaml),
and the complete candidate and resource configuration is
[`perceiver_seam_search_2deg/search.yaml`](../../src/samudra/configs/perceiver_seam_search_2deg/search.yaml).

### Interventions

| Candidate | Input context | Output overlap | Prediction | Role |
| --- | ---: | ---: | --- | --- |
| `hard-no-context` | 0 rings | 0 rings | Absolute | Fixed six-epoch anchor reproducing the prior winner's hard assembly |
| `hard-context1` | 1 ring | 0 rings | Absolute | Tests whether an input halo alone removes seams |
| `full-context` | All 1,080 tokens | 0 rings | Absolute | Fixed six-epoch diagnostic gold standard with common attention support |
| `blend1-no-context` | 0 rings | 1 ring | Absolute | Isolates output overlap and cosine blending |
| `blend1-context1` | 1 ring | 1 ring | Absolute | Tests overlap plus additional surrounding input context |
| `blend2-context1` | 1 ring | 2 rings | Absolute | Tests diminishing returns from a wider output halo |
| `blend1-context1-residual` | 1 ring | 1 ring | Residual | Tests whether tendency prediction complements smooth assembly |

`hard-no-context` and `full-context` are fixed anchors and therefore receive the
full six-epoch budget independent of promotion. The other five candidates are
screened by successive halving at cumulative budgets of one, three, and six
epochs. Validation loss is the promotion objective; at least three competing
candidates advance at each boundary so that a slower-starting seam-removal
family is not eliminated by a small early loss difference.

### Overlap-add reconstruction

For overlap-enabled candidates, each six-patch core decodes an additional query
halo on both sides. Neighboring windows therefore predict the same boundary
pixels. A separable raised-cosine weight transitions across twice the halo
width; weighted predictions are accumulated and divided by their total weight.
Longitude wraps periodically. Latitude halos are clipped at the poles and the
remaining weights are renormalized. Setting overlap to zero follows the old
decoder path exactly.

Input context remains centered on the original six-patch core. This separates
the effect of output blending from the existing `context_patches` control.

### Seam diagnostics

For normalized one-step error `prediction - target`, the search records the
mean absolute adjacent-cell jump at designated boundaries and divides it by the
mean jump at all other adjacent cells. Land pairs are excluded through their
NaN wet mask, and the longitude wrap is included.

The metric is computed separately at:

- encoder-patch boundaries every 3 latitude and 5 longitude cells; and
- decoder-window boundaries every 18 latitude and 30 longitude cells.

A ratio of one means designated boundaries are no rougher than ordinary
interior locations. Values above one indicate grid-locked excess error. Both
`zos` and the mean across prognostic/derived channels are persisted. Validation
MSE remains the promotion objective because an unskilled spatially constant
prediction could obtain a deceptively low seam score.

The primary comparison is a Pareto analysis of validation loss,
decoder-window jump ratio, and epoch runtime. Patch-boundary ratios determine
whether a finer artifact remains after the large decoder-window grid is fixed.

### Compute and observability

Each worker requests one RTX 6000 GPU, four CPUs, and 32 GiB of memory. Up to
seven candidates may run concurrently. Longer walltime limits than the original
search accommodate the full-context and overlapping attention work. A real-data
optimizer-step probe gates release of the competing array.

W&B logs training curves, validation images, seam ratios, timing, resolved
configuration, and immutable code provenance. The search also publishes
Parquet result and epoch tables, lifecycle status, Slurm logs, resolved configs,
and final checkpoints under the public `m2lines-pubs` search prefix.

### Launch record

The immutable source is mounted through code overlay
`samudra-code-f3eaead66a77ee0a05b5724c86f9dbd68d6f251b.img`, whose SHA-256 is
`04fe2962a8d101f4b5bbe811bd2ad9db6bc8d471b3d145ab791bef15161f27f9`.
It uses the stable `physicsnemo-26.05-b73ca826.sif`; its dependency lockfiles
were byte-identical to the experiment commit.

Initial Slurm submissions are:

- fixed anchor array `15974992`;
- real-data optimizer-step probe `15974993`; and
- probe-release controller `15974996`.

Both anchors and the probe entered `RUNNING` state immediately. The probe loaded
the real staged dataset, processed 32 microbatches, and recorded one optimizer
step without error. The release controller then submitted first-rung array
`15975206` and promotion controller `15975207`. The first overlap-add worker
subsequently recorded a real first batch on an RTX6000, exercising the new
forward path under the production CUDA/container environment.

The initial W&B runs include:

- [`hard-no-context`](https://wandb.ai/ocean_emulators/default/runs/q1mtpbny);
- [`full-context`](https://wandb.ai/ocean_emulators/default/runs/wzl40byo); and
- [`blend1-no-context`](https://wandb.ai/ocean_emulators/default/runs/syy4iy9g).

Additional candidate links can be joined from `wandb_id` in the public result
tables as their workers start.

## Planned analysis

The conclusions will distinguish three outcomes:

1. **Seams fall with shared context and overlap:** confirms discontinuous
   attention support and hard assembly as the primary mechanism.
2. **Only shared global context works:** overlap is insufficient and local
   token routing or relative positional information needs redesign.
3. **Decoder-window seams fall but patch seams remain:** the next bottleneck is
   the encoder's one-token-per-patch compression or lack of a high-resolution
   reconstruction path.

No architecture will be recommended from seam score alone. A candidate must
retain competitive validation loss, improve the error maps, and have a compute
path that can plausibly scale beyond 2-degree data.

## Reproducing the results with DuckDB

These queries intentionally remain useful while the run is active.

<details>

<summary>Latest status and quality comparison</summary>

```sql
SELECT
    candidate,
    rung,
    epochs,
    eligible,
    validation_loss,
    "val/seam/window_jump_ratio/zos" AS zos_window_jump_ratio,
    "val/seam/window_jump_ratio/channel_mean" AS mean_window_jump_ratio,
    "val/seam/patch_jump_ratio/zos" AS zos_patch_jump_ratio,
    "val/seam/patch_jump_ratio/channel_mean" AS mean_patch_jump_ratio,
    optimizer_steps,
    round(train_seconds / 60, 2) AS train_minutes,
    worker_stage,
    worker_error
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-seam-removal-2deg--20260818T210753.565365Z/results.parquet'
)
ORDER BY rung DESC, eligible DESC, validation_loss ASC NULLS LAST;
```

</details>

<details>

<summary>Loss, seams, and runtime by epoch</summary>

```sql
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/zos" AS zos_window_jump_ratio,
    "val/seam/window_jump_ratio/channel_mean" AS mean_window_jump_ratio,
    "val/seam/patch_jump_ratio/zos" AS zos_patch_jump_ratio,
    round(epoch_train_seconds / 60, 2) AS train_minutes,
    round(epoch_validation_seconds / 60, 2) AS validation_minutes
FROM read_parquet(
    'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-seam-removal-2deg--20260818T210753.565365Z/epochs.parquet'
)
ORDER BY candidate, epoch;
```

</details>

<details>

<summary>Final-budget Pareto table</summary>

```sql
WITH latest AS (
    SELECT *, row_number() OVER (
        PARTITION BY candidate ORDER BY epoch DESC
    ) AS recency
    FROM read_parquet(
        'https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-seam-removal-2deg--20260818T210753.565365Z/epochs.parquet'
    )
)
SELECT
    candidate,
    epoch,
    validation_loss,
    "val/seam/window_jump_ratio/zos" AS zos_window_jump_ratio,
    "val/seam/window_jump_ratio/channel_mean" AS mean_window_jump_ratio,
    round(epoch_train_seconds / 60, 2) AS train_minutes
FROM latest
WHERE recency = 1
ORDER BY zos_window_jump_ratio, validation_loss;
```

</details>

## Results

### Final six-epoch ranking

All three promoted candidates and both fixed anchors completed six epochs. The
validation-loss winner is `blend1-context1-residual`, with a loss more than
three times lower than any absolute-prediction candidate. Ratios near one mean
that jumps at the known boundary are comparable to matched interior jumps.

| Candidate | Role | Validation loss | Train loss | `zos` window jump ratio | Channel-mean window jump ratio | `zos` patch jump ratio | Channel-mean patch jump ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blend1-context1-residual` | Promoted winner | **0.07291** | **0.54969** | **0.994** | **1.048** | 1.065 | **1.024** |
| `blend1-no-context` | Promoted | 0.23530 | 0.96908 | 1.281 | 1.126 | 1.135 | 1.064 |
| `blend1-context1` | Promoted | 0.23729 | 0.98057 | 1.264 | 1.132 | 1.131 | 1.065 |
| `hard-no-context` | Fixed anchor | 0.23835 | 0.97746 | 2.744 | 2.130 | 1.351 | 1.229 |
| `full-context` | Fixed anchor | 0.27227 | 1.10494 | 1.196 | 1.084 | 1.105 | 1.081 |

At the one-epoch screening boundary, the eliminated `hard-context1` candidate
had validation loss 0.37808, `zos` window ratio 3.958, and channel-mean window
ratio 2.763. The eliminated `blend2-context1` candidate had validation loss
0.38039, `zos` window ratio 1.219, and channel-mean window ratio 1.159. The
two-ring model was already smooth, but was not promoted by the
validation-loss objective; this experiment therefore does not measure its
longer-budget attainable skill.

The residual winner's validation loss continued to improve at every recorded
epoch, from 0.07549 after epoch one to 0.07291 after epoch six. Its aggregate
`zos` and channel-mean seam ratios remained close to one throughout training,
rather than improving only at the final checkpoint.

### W&B image and channel-level sanity check

The final W&B one-step error images were inspected for every finalist. The
`zos` images support the aggregate seam metrics: the hard decoder retains
conspicuous repeated rectilinear banding, both overlap-add absolute models are
substantially smoother, and the residual winner has no comparable `zos` grid.
The globally shared-context anchor is also smooth, supporting the diagnosis
that abruptly changing attention support causes the hard-window artifact.

Alex Merose's manual inspection identified an important limitation of that
aggregate result. Although `zos` is clean, the residual winner retains red and
blue block structure in
[`thetao_15`](https://wandb.ai/ocean_emulators/default/runs/jl31vvpv?nw=nwuseralxmrs&panelDisplayName=val%2Fsnapshot%2Fimage-error%2Fthetao_15&panelSectionName=val%2Fsnapshot%2Fimage-error)
and
[`so_14`](https://wandb.ai/ocean_emulators/default/runs/jl31vvpv?nw=nwuseralxmrs&panelDisplayName=val%2Fsnapshot%2Fimage-error%2Fso_14&panelSectionName=val%2Fsnapshot%2Fimage-error)
at W&B step 8463. The corresponding per-channel metrics confirm that this is
not merely visual interpretation:

| Channel | Window jump ratio | Encoder-patch jump ratio |
| --- | ---: | ---: |
| `thetao_15` | 1.515 | 1.021 |
| `so_14` | 1.518 | 1.082 |
| `so_10` | 1.250 | 1.017 |
| `zos` | 0.994 | 1.065 |

These values point more specifically to decoder output-window boundaries than
encoder patch boundaries. They also show that a channel mean can hide
scientifically relevant failures in individual deep-ocean channels.

The apparent visual severity needs a second qualification: W&B autoscales
each error image independently. At the final checkpoint, the residual model's
plotted ranges were approximately +/-0.261 degrees C for `thetao_15` and
+/-0.030 psu for `so_14`. For `blend1-no-context`, they were approximately
+/-6.86 degrees C and +/-1.80 psu. Residual prediction therefore has much
smaller physical errors even though autoscaling makes its remaining spatial
structure conspicuous. The +/-0.1263 range recorded during manual inspection
belongs to `so_10` at step 8463.

## Analysis and discussion

### Hypothesis assessment

- **H1 is strongly supported.** `hard-no-context` has a final `zos` window
  ratio of 2.744, and adding context without blending made the one-epoch ratio
  worse, at 3.958. Input halo alone does not reconcile independently decoded
  outputs.
- **H2 is strongly supported.** One-ring overlap-add reduced the final
  absolute-model `zos` ratio to 1.26--1.28. It addresses the intended decoder
  seam directly and does not require residual prediction.
- **H3 is only partially tested.** The two-ring candidate was smooth after one
  epoch, but its early validation loss prevented promotion. No conclusion about
  its final skill is justified.
- **H4 is not supported.** With one-ring blending, input context had no
  meaningful seam advantage and slightly worsened final validation loss
  relative to no context.
- **H5 is strongly supported for one-step skill, but not as a complete seam
  cure.** Residual prediction wins decisively on loss and aggregate seam
  metrics, while `thetao_15` and `so_14` retain elevated window jumps.
- **H6 is supported as a diagnostic.** Shared global context is smooth but has
  the worst final validation loss among six-epoch candidates and does not offer
  a scalable solution for finer grids.

### Interpretation

Hard output assembly, rather than insufficient input context, is the dominant
cause of the original large `zos` grid. Overlap-add should therefore be carried
forward as the scalable decoder assembly mechanism. The nearly identical
absolute-model results with and without context suggest omitting context unless
a later experiment demonstrates a channel- or rollout-specific benefit.

Residual prediction is not the only way to address patching artifacts: the
absolute overlap candidates already reduce the major `zos` window seam by more
than half. Instead, residual prediction changes the learning problem by giving
the model an identity-like baseline and asking it to predict a smaller ocean
tendency. This produces dramatically smaller physical errors, but it can make
low-amplitude structured errors easier to see under per-image autoscaling. The
current result supports residual prediction as the best one-step parameterization;
it does not establish that residual prediction removes every boundary mode.

The remaining deep-channel structure also exposes an observability weakness in
the search objective. Promotion used validation loss, while the headline seam
diagnostic used `zos` and a channel mean. Both favor the residual model, but
neither reports its worst channels. Future searches should retain per-channel
metrics and summarize their upper quantiles or maximum in addition to their
mean. Image comparisons should use matched physical color limits across
candidates so that visual amplitude is comparable.

Finally, one-step validation cannot answer whether a small systematic boundary
bias accumulates during autoregressive inference. The residual pathway may
remain stable because its errors are small, or coherent window errors may
compound with lead time. That is an open rollout question, not evidence against
the completed one-step ranking.

## Conclusions

1. Replace hard decoder-window assembly with one-ring overlap-add for the next
   scalable Perceiver generation.
2. Use residual prediction as the leading one-step model parameterization. Its
   final validation loss is 0.07291 versus 0.23530 for the best absolute
   overlap model.
3. Do not interpret the aggregate result as complete artifact removal.
   `thetao_15` and `so_14` retain measurable decoder-window structure.
4. Do not add input context by default: it did not improve the blended absolute
   model in this experiment.
5. Require rollout and channel-tail diagnostics before selecting the production
   multiscale decoder.

## Future work

The highest-value follow-up is a small factorial experiment that separates
prediction parameterization from output reconstruction:

- compare hard and overlap-add assembly under otherwise identical residual
  prediction, which was missing from this search;
- compare any new seam intervention under both absolute and residual
  prediction so its effect is not confounded with the identity baseline;
- evaluate one-step and short 5-, 10-, and 20-step rollouts using validation
  loss, physical-unit RMSE, and per-channel window and patch jump ratios;
- add worst-channel or upper-quantile seam summaries to the promotion report;
  and
- render matched-scale error maps and seam-aligned transects for finalists.

Alex proposed adapting the pixel-space de-aliasing intervention from a locally
supplied NVIDIA preprint. The manuscript is confidential and is intentionally
not committed or linked here. It attributes rollout checkerboards to an
unstable patch-scale mode and refines full-resolution outputs using continuous
upsampling followed by local convolution. It also argues that tendency
prediction can be less spectrally stable than state prediction, making the
residual model's long-rollout behavior particularly important to measure.

That intervention should not yet be assumed to solve the artifact found here.
The manuscript addresses tokenization-scale checkerboards, whereas this
experiment's strongest signal occurs at decoder output-window boundaries;
`thetao_15` and `so_14` have elevated window ratios but near-baseline encoder
patch ratios. Samudra's direct-query Perceiver decoder also does not use the
same unpatchify path. A useful adaptation would apply a lightweight
full-resolution spatial refinement after Perceiver decoding and compare it
under both absolute and residual prediction. Its success criteria should
separately measure encoder-patch, decoder-window, and rollout-amplified modes.
A public citation should replace this internal description before this
notebook is merged into a public branch.
