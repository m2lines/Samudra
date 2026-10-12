<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Score-trained stochastic Samudra on 1-degree OM4

Status: first wave complete, 2026-10-12.
See the [comparison report](RESULTS.md) and [run record](RUNS.md). Persistence and training-only seasonal
climatology references, checkpoint sensitivities, continuous eight-year diagnostics and observation scoring
all completed. All 612 observation rows passed the coverage and aggregation audit. Observation changes are
small and mixed; sampled SST variability improves, while sampled upper-OHC variance-map error worsens.
The report distinguishes mean-field metrics from member metrics and the fixed final-EMA result from
checkpoint-policy sensitivity. All campaign callbacks are retired.
The user authorized B/C/D on Beta, Torch, or Engaging, using Rust loading and no `hfds_anomalies`.
Reuse A; do not retrain it.
Experiment record, not intended to merge. Later waves remain subject to approval.

## Question and scope

Can a convolutional Samudra forecast trained to generate ensembles improve probabilistic skill and preserve
realistic ocean variability at an acceptable cost relative to the deterministic model?

Start with full-state, forced autoregressive forecasting on 1-degree OM4. All members receive the same two
true initial ocean states and prescribed forcing. Predict temperature, salinity, horizontal velocities at
19 depths and SSH: 77 fields per state. This isolates stochastic forecasting before introducing uncertain
surface-only initialization, masked multitask training, or latent-state evolution.

The generated spread describes uncertainty conditional on this resolved state, forcing, training data, and
model. It is not a complete estimate of real-ocean uncertainty or uncertain initialization. An ensemble mean
can legitimately be smoother than its members. A visually sharper member alone is not evidence of skill.

## Source and baseline lock

The planning branch starts from freshly fetched `origin/main`, commit `6b88f2c02`
([source](https://github.com/m2lines/Samudra/commit/6b88f2c02)), rather than the older initializer report branch.

**Samudra 2 uses two input states, two output states per call, and four autoregressive training calls.**
The `samudra_om4_v2` preset has `steps: [4]` and inherits `input_steps: 2` and `output_steps: 2` from
`DataConfig`. The recent control has the same temporal setup: eight five-day outputs, through day 40 after
the last input. Time stepping is not a distinction between these configurations.

**Matched reference: the recent 1-degree deterministic control**, using the same Samudra 2-width
backbone and temporal setup. Its configuration differences from the named `samudra_om4_v2` preset concern
normalization and other settings: the recent control uses InstanceNorm and explicit bfloat16, whereas the
named v2 model specifies BatchNorm. The recent control also uses `tau_hfds` with Rust loading, whereas the
named v2 preset includes `hfds_anomalies` through its shared data configuration. Record these settings
explicitly for a matched comparison; they do not change the agreed two-in/two-out, four-call task.

Evidence for the recent control:

- [Saved control configuration](https://github.com/m2lines/Samudra/blob/a20c44484/experiments/current-stochastic-depth-1deg/control-train.yaml)
  and [experiment record](https://github.com/m2lines/Samudra/blob/a20c44484/experiments/current-stochastic-depth-1deg/README.md).
- [PR #915](https://github.com/m2lines/Samudra/pull/915) contains the later stochastic-depth comparison.
- Reuse seed-15 Beta job **210664**, W&B [uaei2vbr](https://wandb.ai/ocean_emulators/default/runs/uaei2vbr).
  Its final `ckpt_70.pt` and `ema_ckpt.pt` were verified present on 2026-10-11 under
  `/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006/control/output/current-1deg-control-ga2-210664/saved_nets`.
  The recorded source/config commit is `a20c44484`; A is a reused historical control, not a new run on this
  branch. B is the newly trained deterministic control. All final checkpoints will be rescored with the
  common evaluator. A's original validation-selected checkpoint uses a different selection policy.

Pin a resolved configuration for every arm, including defaults. The proposed common settings are:

| Item | Proposed setting |
| --- | --- |
| Backbone | ConvNeXt U-Net, widths 280/380/480/520, expansion 2, dilations 1/2/4/8 |
| Other architecture | InstanceNorm, capped GELU, zonally periodic upsampling, absolute-state prediction |
| State history/output | Explicitly set `input_steps: 2`, `output_steps: 2` |
| Training rollout | Four autoregressive calls, eight five-day outputs, through day 40 after the last input |
| Precision and regularization | bfloat16 network; FP32 score arithmetic; stochastic depth 0 |
| Forcing | `tau_hfds`: tauuo, tauvo, hfds; same supplied times in every arm |
| Loading | Rust; existing successful bounded-read/prefetch settings, adjusted only for measured resources |
| Normalization | Existing means/stds, identical across arms |
| Full-run recipe | 70 epochs, Adam optimizer family/settings from the resolved control, cosine schedule, initial LR 0.0006 |
| Effective context batch | 32 distinct forecast origins per optimizer update, excluding ensemble replicas |
| First training seed | 15; later confirmation seeds 16 and 17 |

The user selected `tau_hfds` with no `hfds_anomalies` for all new arms. Keep InstanceNorm across B/C/D;
BatchNorm over folded ensemble replicas would itself change batch statistics.

## Data and split

Use OM4 one-degree fields on the 180 x 360 grid at their actual five-day temporal support. On Beta,
`/mnt/home/jrusak/data/om4/v2026-09/om4_onedeg`, used by A, now resolves to project storage at
`/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg`. The new runs use this project path
directly, with `OM4.zarr`, `OM4_means.zarr` and `OM4_stds.zarr`. The symlink and all three stores were
checked on 2026-10-11. Each run exports its actual source bounds, variables, forcing, and fitting/validation
origin manifests. Retain depth-specific wet-cell masks.

Proposed periods follow the recent control: training from 1975-01-03 to the 2013-10-05 boundary, validation
from that boundary through 2014-10-05, and testing from 2014-10-10 through 2022-12-24. Resolve endpoints against
the actual archive timestamps. Export an origin/target manifest and assert disjoint training, validation,
and test **target timestamps**, including every autoregressive output. Earlier input history may cross a split
boundary; training targets may not. Common endpoint labels must not occur in both fitting and validation.

Atmospheric forcing is prescribed in this experiment, as in the control. Scores against OM4 are primary.
Existing observation metrics are secondary: five-day DUACS/OISST and monthly ARGO-IAP heat content, with
precisely matched timestamps and the established complete-month/complete-year averaging rules.
They do not constitute surface-observation initialization or instantaneous daily observational evaluation.

## First comparison

| Arm | Network and training | Purpose |
| --- | --- | --- |
| A: vanilla, reused | Existing seed-15 checkpoint; dynamic MSE with weight-ratio cap 20 | Practical baseline; no new training |
| B: fixed-weight MSE | Same backbone with the stochastic conditioning interface disabled; fixed normalized-field MSE | Control for changing channel weighting and adding the interface |
| C: stochastic marginal | Shared backbone with multiscale random conditioning; gridpoint almost-fair CRPS | Test direct ensemble generation |
| D: stochastic structured | Exactly C, plus scores on spatial averages and vertical/temporal differences | Test dependence and variability beyond gridpoint marginals |

B/C/D start from the same seeded backbone initialization and use paired context minibatch order. Use independent
random-number streams for data sampling, parameter initialization, training noise, and evaluation noise.
No frozen deterministic predictor is required. C/D produce the full two-state output jointly, not separate
independently sampled variables or times. Record added conditioning parameters and measured compute.

B/C/D use identical fixed positive channel weights after the existing standardization (initial proposal:
equal weight per prognostic channel, hence T/S/U/V each contribute 19 channels and SSH one). A retains its
actual dynamic weights. Report both channel-averaged and equal-variable-group metrics so this convention is
visible. A-to-C/D measures practical benefit; B-to-C isolates the objective/random-conditioning package more
closely; C-to-D isolates the added transformed-field losses. Dynamic training losses never rank forecast skill.

## Stochastic convolutional model

For member m and model call k, predict `x_next[m] = G(x_history[m], forcing, noise[m,k])`.
Each call emits the next two physical states. Each member then feeds back its own predicted history; never
feed back the ensemble mean or reset to true states within an evaluated rollout.

Inject a small number of independent spatial Gaussian noise channels at several U-Net resolutions through
learned conditioning projections. Draw independent low-resolution fields at each scale, use the same
geographic boundary conventions as the backbone, and apply projections after normalization so InstanceNorm
does not simply erase a spatially constant perturbation. The shared convolutions mix their effects across all
output variables and depths. Use small nonzero projection initialization so members differ from the first
update; exact equality can leave absolute-difference spread gradients uninformative.

Initially draw new noise per member and autoregressive call. The evolving member state carries previous
uncertainty; this construction alone does not prove realistic temporal covariance. Diagnose jumps at two-state
block boundaries. Temporally correlated noise is a targeted follow-up if those diagnostics justify it.

Use four members per training context and 16 per final evaluated origin. Ensemble replicas are not four
additional independent training examples. Qualification decides whether members fit locally, require
activation checkpointing, or require differentiable member sharding. Pairwise loss gradients must reach every
member; ordinary detached cross-GPU gathering is not sufficient.

One member requires one network call per two-state advance. Producing 16 members costs about 16 such calls;
four-member training is more expensive than deterministic training. No speedup over the small one-field
diffusion pilot is claimed until comparing matched full-state tasks and measured runtimes.

## Ensemble objective

For a scalar truth y and M generated values x, define empirical CRPS and fair CRPS as:

```text
empirical = mean_i |x_i - y| - sum_ij |x_i - x_j| / (2 M^2)
fair      = mean_i |x_i - y| - sum_i!=j |x_i - x_j| / (2 M (M - 1))
training  = 0.95 * fair + 0.05 * empirical
```

This almost-fair form follows [AIFS-CRPS](https://arxiv.org/abs/2412.15832); its small finite-ensemble bias is
deliberate to avoid a degeneracy of exactly fair CRPS in small training ensembles. Use the nonnegative
pairwise formulation and FP32 reductions. Report empirical and fair scores separately at evaluation, with the
same member count for probabilistic arms. A deterministic forecast's empirical CRPS equals its MAE.

D uses `L_D = L_grid + 0.02379038744 * L_spatial + 0.0002169552608 * L_differences`.
The initial proposal used 0.1 for both auxiliaries, but qualification found that this made the difference-term
output-gradient norm about 46 times the grid term. The frozen coefficients are
`min(0.1, 0.1 * ||grad_grid|| / ||grad_aux||)`, measured with respect to the predicted state at initialization
on the first training batch of each rank (16 contexts total, sampler epoch 0). Each weighted auxiliary then
has 10% of the grid term's initial output-gradient norm. No validation/test score selects these coefficients.
They are fixed throughout training; this is not dynamic reweighting or a claim of optimal weights.

Normalize each auxiliary group by its fixed training-only target RMS and reduce it to an average before
weighting. Exclude features with RMS at most `1e-6` instead of amplifying tiny denominators. Differences here
are differences of **standardized fields**, not physical-unit vertical gradients. Preserve equal terms and
contexts across C/D. The exact coefficients, measured norms, and selected training windows are retained.

- `L_spatial`: the same ensemble score on area- and wet-mask-weighted 4 x 4 and 16 x 16 cell averages.
  Empty blocks are excluded and latitude/longitude edges use a pinned boundary convention.
- `L_differences`: scores on selected adjacent-depth T/S/U/V differences and successive five-day changes,
  using common valid-cell intersections, including temporal changes across output-block boundaries.

These linear transformations probe dependence without pretending that marginal or transformed marginal scores
fully identify the joint ocean distribution. A later patch energy-score comparison is possible if needed.
[Multiscale AIFS work](https://arxiv.org/abs/2506.10868) and
[FourCastNet 3](https://arxiv.org/abs/2507.12144) motivate scale-aware probabilistic objectives; their reported
weather benefits do not establish benefits here.

## Evaluation and selection

Predeclare origins, seeds, masks, normalization, aggregation, and plot dates before evaluating test targets.

1. **Repeated forecasts:** approximately quarterly held-out origins through 2021, each with complete 360-day
   targets in the test interval. Score leads 5, 10, 20, 40, 90, 180, and 360 days. Store the exact common manifest.
   These are nominal days (source steps times five). The Julian-calendar source includes six-day intervals
   at the ends of leap years, so 72 forecast steps span 360 or 361 actual days. Preserve the actual origin
   and target timestamps and label plot axes accordingly; see the verified time-coordinate details in `RUNS.md`.
2. **Long-rollout diagnostic:** one continuously evolving trajectory from the first test origin to the final
   timestamp, without resets. Initially use four stochastic members to bound cost. Interpret drift, stability,
   temporal statistics and mean-state errors separately from repeated forecast skill.
3. **Checkpoint policy:** primary comparison uses final EMA at the same declared training exposure. Keep raw
   checkpoints as a sensitivity. Also report validation-selected EMA: deterministic arms by fixed rollout RMSE,
   stochastic arms by fixed normalized marginal CRPS on validation-only origins at 10/20/40/90 days, using a
   fixed eight-member draw. These are separately labeled objective-specific selections, not the primary result.
   The launched B/C/D runs use two 90-day validation rollouts per epoch, rather than the initially proposed
   360-day rollout. This implementation deviation was disclosed in the conversation; annual forecasts are
   part of the separate held-out evaluation, not training-time checkpoint selection.

Report, by variable/depth/lead and fixed regions:

- Ensemble-mean RMSE, bias and anomaly correlation; member error and anomaly amplitude separately.
- Empirical/fair CRPS, reliability/rank histograms, interval coverage, and spread/error with finite-member
  references. Randomize exact rank ties; never compare a four-member range with a 16-member reference.
- Spatial structure functions/spectra, vertical and cross-variable covariance, and temporal increments.
  Joint patch scores supplement marginal scores; nonlinear diagnostics are computed per member before
  summarizing their distribution.
- Heat/salt-content drift and resolved kinetic-energy behavior, using documented common geometry/masks.
- Fixed-date maps of truth, deterministic prediction, ensemble mean, members 0 and 1, spread and error.
  Use fixed color limits and integer cell-to-pixel scaling. Do not choose favorable members after viewing.
- Secondary, timestamp-matched observational surface/OHC metrics where available. Report members and mean
  distinctly and disclose observation support and uncertainty.

Include persistence and training-period climatology references. An inexpensive stochastic reference is
deterministic forecasts plus whole, jointly sampled residual fields from eligible training windows, selected
by season/lead. Use this only if out-of-fit residuals are available; in-sample fitting residuals would bias its
spread. Skip rather than silently substituting a poorly calibrated residual baseline.

Audit noise sensitivity, reproducibility, dry-cell values, member independence, nonfinite outputs, and member
identity across rollout. Summarize year-block uncertainty for paired metrics, separately from training-seed
uncertainty. Repeated forecasts within a year are not independent replications.

## Compute and execution stages

The user authorized this first wave on 2026-10-11. Beta has available B200 nodes, the same hardware used for A;
Torch has active jobs and CPU-limit queue pressure, while Engaging's batch-mode SSH authentication failed.
Use Beta's pinned ARM64 PhysicsNeMo container and existing compiled Rust loader. Project storage has 32 TiB
available at the live check. Production requests are full four-GPU nodes; qualification uses `beta_test`.

- **Qualification:** bounded synthetic correctness checks, then a short real-data run to establish peak memory,
  context updates/second, member-steps/second, nonzero noise response, and early spread/gradient behavior.
  Budget envelope: at most eight GPU-hours including real-data qualification.
- **First wave:** reuse A and train B/C/D at seed 15. Budget envelope: approximately 200 GPU-hours including evaluation,
  finalized from qualification. A previous four-GPU control reported about 3 h 40 min elapsed (~14.7 allocated
  GPU-hours), but that is historical hardware-specific evidence, not a measured estimate for the new code.
  Four-member training may approach four times deterministic compute before other overheads.
- Keep the same 70-epoch exposure as the intended comparison. If it does not fit, agree a common shorter
  exposure before launching the main arms. Report matched-exposure results and skill against accumulated GPU
  cost; do not label equal epochs as equal compute. Where needed, an equal-cost deterministic continuation is
  a later explicit control, not an inferred result.
- **Next wave, after review:** repeat the informative comparisons with seeds 16/17, or fix a diagnosed spread,
  structure or temporal failure. Only then add masked multitask training or surface-only initialization with a
  matched deterministic comparison. New training waves wait for approval.

For submitted jobs, check roughly hourly while active. The initial callback connectivity failure was resolved
on 2026-10-12 by migrating the receiver to HTTPS port 443. The already-running training scripts have no exit
hook and retain hourly reminder-based monitoring. New evaluation jobs use exit callbacks plus hourly fallback
reminders. Preserve job IDs, immutable source/container identifiers, complete configs,
dataset manifests, checkpoint lineage, runtime counters, raw metrics and selected field assets.

### Implemented first-wave details

- `common.yaml` pins the recipe; `prepare.py` exports complete resolved configs. `train_score.py` subclasses
  the main trainer, reusing Rust batch loading, DDP, Adam, cosine decay, gradient accumulation/clipping, EMA,
  W&B and atomic epoch checkpoints. Production source under `src/samudra` is unchanged.
  The inherited `loss: mse` field supplies B's loss; C/D override the model's training objective with the
  ensemble score identified by `model.arm`.
- `score_model.py` adds eight Gaussian channels at each of the four encoder resolutions, after the complete
  ConvNeXt block (including its normalization). The projection weights start at standard deviation
  `0.02/sqrt(8)`. Hooks execute inside the existing activation checkpoint, which preserves random draws on
  backward recomputation. B has no projections. Local tests compare common backbone weights exactly.
- Grid MSE/CRPS uses the control's **land-inclusive** spatial denominator after masking. Auxiliary scores and
  evaluation normalize over valid wet area. Thus equal normalized channel coefficients do not imply equal
  wet-ocean influence for channels with different masks. This choice preserves B's existing plain-MSE semantics.
- D's fixed RMS scales use 64 training origins selected by sampler seed 15, epoch 0, four batches per rank on
  four GPUs, including all eight target frames per origin. Save the sampled indices and scales before training.
  Spatial scores average the two block sizes; difference scores average vertical and temporal groups. For
  16-cell blocks, retain partial edge blocks using empty padding, with no duplicated geographic cells.
  Generate its production config with `prepare.py --auxiliary-calibration PATH --structured-scales REMOTE_PATH`
  after qualification; the generator omits the D training config if frozen coefficients are not supplied.
- First-wave validation uses eight members for C/D, one for B. Every epoch scores one-call validation windows
  and two validation-only 90-day rollouts, initialized at window indices 0 and 18. Select secondary checkpoints
  with the average score at 10/20/40/90 days (CRPS for C/D, RMSE for B). This replaces the proposed annual
  training-time validation to bound overhead. Annual and longer rollouts remain final evaluation tasks. The
  primary comparison remains the epoch-70 EMA; A's old best-validation checkpoint is not a matched selection.
- Epoch-boundary restarts restore optimizer, EMA, scheduler and progress from the ordinary checkpoint. The
  training-noise stream is deterministically seeded by epoch and rank; validation uses a separate fixed stream
  inside `fork_rng`. Sampler order has its own seeded stream. Mid-epoch resumes are not supported: a preempted
  partial epoch is replayed from the last completed epoch. Qualification checkpoints are explicitly non-resumable.
- Ten local numerical/model tests cover CRPS values and gradients, dry/edge masking, depth ordering, stochastic
  checkpoint recomputation, common initialization, member history, model-state round trips, exact B parity,
  and the distinction between ensemble-mean error, member error and spread.
- `evaluate_score.py` provides common quarterly-origin 360-day forecasts, empirical/fair CRPS, mean/member
  RMSE, bias, spread, quantile/range coverage, randomized-tie rank histograms and prespecified native-grid fields.
  It supports A's original checkpoint names as well as B/C/D. It requires an online W&B run and writes
  per-rank CSVs plus completion manifests. Separate continuous, reference, checkpoint-sensitivity and
  observation runners completed the reporting-stage evaluations; their audits and plots are linked in `RESULTS.md`.
  `evaluate_beta.sbatch` launches four independent origin workers; qualification uses four complete annual
  forecasts, while full evaluation uses the common complete-origin manifest. Final checkpoints must report
  epoch 70, and checkpoint checksums, source/container identity, config, and row counts are recorded.

## Decision and deliverables

A useful outcome is improved probabilistic skill with credible calibration and joint-field structure, without
an unacceptable regression in mean forecasts or long-run stability. A smooth ensemble mean is acceptable;
underdispersed, noisy, or physically inconsistent members are not evidence of success. Report tradeoffs rather
than choosing an undisclosed composite that can hide variable-specific failures.

Before launch, freeze: the resolved baseline configuration; data and forcing; optimizer/EMA settings; the target
manifest; stochastic projection specification; transformed-score scaling; execution budget and hardware.
These are configuration/qualification tasks, not grounds for changing scientific inputs silently.

The report should contain a matched comparison table, skill-versus-GPU-cost and lead curves, calibration and
dependence diagnostics, representative maps, limitations, and a proposed next wave. Keep machine-readable
per-origin/per-variable/per-lead metrics alongside the human report. Publish this campaign as a draft
experiment PR and run relevant local checks even though experiment PRs skip automatic CI.
