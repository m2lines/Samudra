<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver 2-degree residual assembly and pixel de-aliasing searches

## Status

This notebook pre-registers the questions, hypotheses, interventions, and
decision criteria for the follow-up to the
[decoder seam-removal search](perceiver_2deg_seam_removal_search.md). It was
written before implementing the new pixel-refinement pathway, submitting either
search, or inspecting any candidate results.

The program is deliberately split into two independently reported searches:

1. a prediction-parameterization and output-assembly factorial; and
2. a pixel-space de-aliasing factorial inspired by a locally supplied NVIDIA
   manuscript.

The searches may be submitted concurrently from the same immutable code
revision. Separate search runs preserve causal interpretability and independent
promotion or failure state while imposing negligible additional Slurm overhead.
Run identifiers, code revisions, W&B groups, and public artifact locations will
be recorded here before results are inspected.

Both searches were submitted on 2026-08-20 from immutable code revision
[`36ff2a2b`](https://github.com/m2lines/Samudra/tree/36ff2a2b34c4fa9c71708415f4d0573c43fb289f).
Their real-data Slurm probes completed 32 accumulated batches and one finite
optimizer update before releasing either candidate array. Search A's four
workers started immediately. Search B's eight workers were submitted and are
waiting behind the account's active GPU limit; this is scheduler backpressure,
not an experiment failure.

## Prior evidence

The first seam-removal search established five relevant facts on the public
2-degree OM4 dataset:

1. Hard output-window assembly produces severe decoder-window discontinuities.
   The six-epoch `hard-no-context` anchor had `zos` window jump ratio 2.744.
2. One-ring overlap-add is an effective scalable intervention. Absolute
   overlap models reduced that ratio to approximately 1.26--1.28.
3. Input context did not help once outputs were blended. With one-ring overlap,
   zero context slightly outperformed one-ring context by validation loss and
   had comparable seam metrics.
4. `blend1-context1-residual` was the clear one-step skill winner, reaching
   validation loss 0.07291 versus 0.23530 for the best absolute overlap model.
5. Aggregate metrics hid residual structure in individual channels. The
   residual winner's final decoder-window ratios were 1.515 for `thetao_15` and
   1.518 for `so_14`, despite a channel mean of 1.048 and `zos` ratio 0.994.

The residual candidate changed prediction parameterization and reconstruction
simultaneously relative to the original hard absolute model. The previous
search therefore did not determine whether residual prediction needs
overlap-add, whether the two interventions interact, or whether remaining
low-amplitude boundary modes grow during rollout.

A locally supplied confidential NVIDIA manuscript motivates a second line of
investigation. It attributes periodic rollout artifacts in a patch-tokenized
model to unstable patch-scale modes and uses full-resolution conditioning plus
local convolutional refinement to suppress them. Samudra's measured artifact
is not identical: its strongest scalar signal is aligned with decoder output
windows, and its direct-query Perceiver does not unpatchify semantic tokens.
The appropriate experiment is therefore an adaptation, not a direct port.

## Research questions

### Search A: residual prediction and output assembly

1. Does residual prediction remain substantially better when absolute and
   residual models use identical context and assembly?
2. Does residual prediction alone suppress seams, or does it still benefit
   from overlap-add?
3. Is there an interaction between prediction parameterization and assembly
   that is hidden by aggregate validation loss?
4. Do one-step decoder-window modes remain stable, decay, or grow over short
   autoregressive rollouts?

### Search B: pixel-space de-aliasing

1. Does a full-resolution spatial refinement after global output assembly
   reduce the remaining channel-specific window structure?
2. Is local convolutional refinement sufficient, or is smoothly upsampled
   processor conditioning also necessary?
3. Does the intervention improve both absolute and residual prediction, or
   only make an already-small residual error smoother?
4. Does it suppress decoder-window and tokenization-scale spectral modes during
   rollout without blurring physically meaningful gradients or degrading RMSE?

## Hypotheses

### H1: overlap-add and residual prediction provide distinct benefits

Residual prediction will produce the largest validation-loss reduction, while
overlap-add will produce the largest decoder-window seam reduction. The
residual hard-assembly candidate is expected to retain more boundary structure
than the residual overlap candidate even if its absolute errors remain small.

### H2: overlap-add remains necessary under residual prediction

The identity-like residual baseline reduces error amplitude but does not force
independently decoded windows to agree. One-ring overlap-add should therefore
reduce the residual model's worst-channel window ratios and their rollout
growth relative to hard residual assembly.

### H3: residual parameterization may trade one-step ease for rollout risk

Residual prediction should retain its one-step advantage, but a coherent
per-step boundary tendency may accumulate autoregressively. This hypothesis is
not a prediction that residual models will fail; it requires measuring seam
and spectral-mode trajectories rather than inferring stability from one step.

### H4: post-assembly pixel refinement will reduce remaining structured modes

A local full-resolution refinement block operating after overlap-add can
communicate across former decoder-window boundaries. It should reduce
worst-channel window ratios and localized spectral peaks beyond overlap-add
alone.

### H5: smooth processor conditioning and local refinement are complementary

Bilinearly upsampled processor features provide a continuous large-scale
conditioning field, while a depthwise local block repairs high-frequency
structure. Their combination should outperform either component alone if the
adapted mechanism matches Samudra's failure mode.

### H6: genuine de-aliasing will transfer across prediction parameterizations

If the refinement fixes reconstruction rather than merely hiding error through
an identity baseline, it will improve seam and spectral diagnostics under both
absolute and residual prediction. A benefit confined to residual prediction
would be weaker evidence for a general decoder intervention.

## Why these are two searches

Combining every candidate into one successive-halving pool would make two
different scientific questions compete under one validation-loss ranking.
Residual prediction's large early loss advantage could eliminate absolute
de-aliasing ablations before they reveal whether reconstruction improved.
Likewise, a larger refinement module may learn more slowly even when it has
better rollout behavior.

The searches will therefore have separate run IDs, result tables, and analyses.
They can still use the same Slurm submission resources and run concurrently.
The shared controls make their results comparable, while separate search state
makes failures and conclusions independently inspectable.

Because these are small causal factorials, every candidate should receive the
full six-epoch smoke budget. Successive halving is useful for broad architecture
exploration, but early elimination would leave missing cells in the factorial
and weaken the interaction analysis. If resource pressure requires screening,
all four Search A cells and the no-refinement controls in Search B must remain
fixed candidates.

## Common experimental setup

Both searches will use:

- the public 2-degree OM4 training and validation data;
- the same split, normalization, channel ordering, and seed used by the prior
  seam-removal search;
- the selected patch-local Perceiver encoder with 256 internal latents;
- the same ConvNeXt U-Net processor and direct output-query cross-attention;
- 6 x 10 degree encoder patches and the same decoder window geometry;
- normalized training loss and the prior learning-rate schedule;
- six training epochs for every causal comparison;
- PyTorch scaled dot-product attention with automatic backend selection;
- one-step validation at every epoch; and
- matched 5-, 10-, and 20-step validation rollouts from identical initial
  conditions at the final checkpoint.

The common training config is
[`perceiver_dealias_search_2deg/train.yaml`](../../src/samudra/configs/perceiver_dealias_search_2deg/train.yaml),
and the common model is
[`perceiver_dealias_search_2deg/model.yaml`](../../src/samudra/configs/perceiver_dealias_search_2deg/model.yaml).
Training uses batch size 1, 32 accumulated batches per optimizer update, two
data-loader workers, one RTX6000, four CPUs, and 32 GiB of host memory per
candidate. Each worker has a four-hour limit; experience with the preceding
search suggests that six epochs should finish well inside that allocation.

## Search A interventions

Search A is a 2 x 2 factorial. Input context is fixed at zero because it did not
improve the blended absolute model in the previous search.

| Candidate | Prediction | Output assembly | Purpose |
| --- | --- | --- | --- |
| `hard-absolute` | Absolute state | Hard windows | Reproduces the original seam-prone anchor |
| `blend1-absolute` | Absolute state | One-ring overlap-add | Measures assembly benefit under absolute prediction |
| `hard-residual` | Residual tendency | Hard windows | Isolates residual prediction without smooth assembly |
| `blend1-residual` | Residual tendency | One-ring overlap-add | Measures the combined intervention without input context |

The previous `blend1-context1-residual` result remains an external reference,
not a cell in this factorial. Comparing it with `blend1-residual` will provide
secondary evidence about context under residual prediction, but the runs are
not interchangeable unless all other resolved config and code fields match.

If budget permits, the complete factorial should be repeated with a second
seed. Paired seed differences are more informative than adding partially
trained architectures.

## Search B intervention architecture

The adaptation operates on full-resolution decoder features after all query
windows have been assembled with one-ring overlap-add. Applying refinement
inside each window would preserve the very support discontinuity being tested.

The proposed decoder path is:

1. Decode each query to a hidden feature of width `queries_dim`, rather than
   projecting immediately to physical output channels.
2. Assemble hidden features globally with the existing normalized overlap-add.
3. Optionally bilinearly upsample the processor patch grid to the output grid,
   project it to `queries_dim`, and use it as a smooth conditioning signal.
4. Optionally apply a lightweight full-resolution residual block consisting of
   normalization, a 3 x 3 depthwise spatial convolution, activation, and
   pointwise channel mixing. Longitude padding must be periodic and latitude
   must retain polar boundary semantics.
5. Project the refined hidden features to physical output channels.

Moving the linear output projection after overlap-add is mathematically
equivalent when refinement is disabled, so the no-refinement candidate remains
a meaningful control. The adaptation is intentionally smaller than the
manuscript's atmospheric decoder because Samudra already produces
full-resolution query features and does not have an unpatchify stage.

## Search B interventions

Search B is a 2 x 2 x 2 factorial over prediction parameterization, smooth
processor conditioning, and local pixel refinement. Output assembly is fixed
to one-ring overlap-add with zero input context.

| Factor | Values |
| --- | --- |
| Prediction | Absolute, residual |
| Smooth processor conditioning | Disabled, bilinear latent-grid conditioning |
| Pixel refinement | Disabled, depthwise residual refinement |

This produces eight candidates:

| Candidate | Prediction | Conditioning | Refinement |
| --- | --- | --- | --- |
| `absolute-none` | Absolute | No | No |
| `absolute-condition` | Absolute | Yes | No |
| `absolute-refine` | Absolute | No | Yes |
| `absolute-condition-refine` | Absolute | Yes | Yes |
| `residual-none` | Residual | No | No |
| `residual-condition` | Residual | Yes | No |
| `residual-refine` | Residual | No | Yes |
| `residual-condition-refine` | Residual | Yes | Yes |

`absolute-none` and `residual-none` are shared conceptual controls with Search
A's blended candidates. They should either reuse exactly the same immutable
checkpoints or be retrained and treated as independent runs; results from
different code revisions must not be silently combined.

## Measurements and observability

### Skill

- normalized validation loss and training loss by epoch;
- physical-unit RMSE, bias, and anomaly-correlation metrics by prognostic
  variable, depth, and rollout lead time; and
- best and final validation loss, with optimizer-step counts recorded so failed
  or empty training cannot rank.

### Artifact diagnostics

- decoder-window jump ratio for every channel, plus channel mean, upper
  quantile, and maximum;
- encoder-patch jump ratio with the same reductions;
- zonal and meridional error spectra, with energy summarized around the known
  encoder-patch and decoder-window frequencies;
- seam-aligned transects across representative ocean regions; and
- matched-physical-scale error maps across candidates, in addition to
  independently autoscaled diagnostic maps.

### Rollout behavior

At lead times 1, 5, 10, and 20, record:

- physical-unit skill and bias;
- per-channel window and patch jump ratios;
- patch/window-frequency spectral energy relative to neighboring frequencies;
  and
- the change from the candidate's one-step artifact amplitude.

The same initial states must be used for every candidate. A refinement is not
successful if it lowers seam ratios by broadly smoothing the forecast while
materially degrading gradients, spectra, or physical-unit skill.

### Systems measurements

- train, validation, and rollout seconds;
- peak allocated and reserved GPU memory;
- data-loader wait or throughput diagnostics;
- parameter count and refinement-specific parameter count;
- SDPA backend selected at runtime; and
- worker heartbeat, batches seen, optimizer steps, exit state, and error.

All scalar histories, resolved configs, checkpoints, logs, and image metadata
should be published to the public search artifact location when permitted. The
confidential manuscript itself must not be copied into public artifacts.

## Decision criteria

Search A will answer causal questions rather than select solely by validation
loss. Report the main effects of residual prediction and overlap-add and their
interaction for final loss, worst-channel seam ratio, and rollout seam growth.

Search B will recommend pixel refinement only if it:

1. improves channel-tail seam or spectral diagnostics relative to the matched
   no-refinement control;
2. retains the improvement at later rollout lead times;
3. does not materially degrade physical-unit skill or resolved gradients; and
4. provides evidence under both prediction parameterizations, or clearly
   documents why its effect is parameterization-specific.

The final architecture should be selected from a Pareto view of skill,
artifact suppression, rollout stability, memory, and runtime. Validation loss
remains important but is not an adequate single objective for this experiment.

## Preflight checklist

- [x] Implement hidden-feature overlap assembly with a no-refinement parity
      test.
- [x] Implement periodic-longitude, polar-safe processor upsampling.
- [x] Implement the optional full-resolution refinement block.
- [x] Add unit tests proving that disabled conditioning/refinement preserves
      the baseline decoder topology and checkpoint behavior.
- [x] Add synthetic boundary-jump and periodic-phase artifact tests.
- [x] Add one-step per-channel mean, upper-quantile, maximum, and periodic-mode
      summaries to search artifacts.
- [ ] Configure deferred 5-, 10-, and 20-step rollout jobs against every final
      checkpoint before training completes.
- [x] Create separate Search A and Search B manifests from one immutable code
      revision.
- [x] Run local shape and forward/backward checks and the full standard test
      suite.
- [x] Run one Slurm probe per search and require optimizer progress plus a
      finite training loss before launching the full arrays.
- [x] Record run IDs, W&B groups, artifact URIs, job IDs, and expected runtime
      below.

## Search records

### Search A

- Search run:
  `perceiver-residual-assembly-2deg--20260820T005445.253301Z`.
- Git revision: `36ff2a2b34c4fa9c71708415f4d0573c43fb289f`.
- W&B group:
  `perceiver-residual-assembly-2deg--20260820T005445.253301Z`.
- Public artifacts:
  [OSN search record](https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-residual-assembly-2deg--20260820T005445.253301Z/).
- Slurm probe: array `16057661`; passed with 32 batches and one optimizer
  update.
- Slurm candidate array: `16057724` (`0-3%4`); all four tasks began running
  immediately, registered online W&B runs, and made optimizer progress.
- Expected completion: approximately three to four hours after array start.
  Early first-epoch throughput ranges from roughly 0.6 to 1.4 seconds per
  batch across candidates; the four-hour allocation is therefore useful but
  fairly tight, and the final record should note any timeout explicitly.

### Search B

- Search run:
  `perceiver-pixel-dealiasing-2deg--20260820T005523.824824Z`.
- Git revision: `36ff2a2b34c4fa9c71708415f4d0573c43fb289f`.
- W&B group:
  `perceiver-pixel-dealiasing-2deg--20260820T005523.824824Z`.
- Public artifacts:
  [OSN search record](https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/experiments/searches/perceiver-pixel-dealiasing-2deg--20260820T005523.824824Z/).
- Slurm probe: array `16057701`; the full
  `residual-condition-refine` path passed with 32 batches and one optimizer
  update.
- Slurm candidate array: `16057734` (`0-7%8`); submitted successfully and
  initially pending on the account GPU-QOS limit.
- Expected completion: approximately six to eight hours after Search A began,
  if the four-GPU account limit continues to schedule Search B in two waves.
  This estimate will be revised from its first full-worker throughput.

## Query templates

The final result and epoch-history URLs will be inserted after submission. The
following queries define the intended first-pass analyses.

<details>

<summary>Final candidate health and skill</summary>

```sql
SELECT
    candidate,
    epochs,
    eligible,
    validation_loss,
    train_loss,
    optimizer_steps,
    round(train_seconds / 60, 2) AS train_minutes,
    round(validation_seconds / 60, 2) AS validation_minutes,
    worker_stage,
    error
FROM read_parquet('<SEARCH_ARTIFACT_URL>/results.parquet')
ORDER BY validation_loss ASC NULLS LAST;
```

</details>

<details>

<summary>Final aggregate seam comparison</summary>

```sql
SELECT
    candidate,
    validation_loss,
    "val/seam/window_jump_ratio/zos" AS zos_window_jump_ratio,
    "val/seam/window_jump_ratio/channel_mean" AS mean_window_jump_ratio,
    "val/seam/patch_jump_ratio/zos" AS zos_patch_jump_ratio,
    "val/seam/patch_jump_ratio/channel_mean" AS mean_patch_jump_ratio
FROM read_parquet('<SEARCH_ARTIFACT_URL>/results.parquet')
ORDER BY mean_window_jump_ratio, validation_loss;
```

</details>

Additional SQL for channel tails, spectra, and rollout trajectories will be
written against the implemented artifact schema before launch rather than
inventing columns that workers do not yet publish.

## Results

Pending.

## Analysis and discussion

Pending.

## Conclusions

Pending.

## Future work

Pending.
