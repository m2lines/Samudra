<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Stabilize observation training, then replicate the earlier-OM4 family at longer budgets

**Draft for review; no new training or evaluation is launched by this PR.**
The user clarified that “U-multitask” means the **earlier-OM4 family**, not the
original regional-patch task. This plan proposes a small stabilization screen,
then a review of follow-up stabilization/modeling tests before paired fresh runs
through **16,000 total updates**, retaining checkpoints
at 4k/6k/8k/12k/16k along the same trajectory. It leaves the separately authorized
native-quarter-degree campaign unchanged.

Contents: [Motivation](#motivation) · [Models](#models-and-fixed-science) ·
[Diagnostics](#stage-1-measure-what-is-fluctuating) ·
[Stabilization](#stage-2-small-paired-stabilization-screen) ·
[Long training](#stage-3-longer-paired-reproductions) ·
[Evaluation](#evaluation-and-decision-rules) ·
[Execution](#implementation-and-execution) · [Review decisions](#review-decisions).

## Motivation

The [4k/6k RMSE curves](early-rmse-4k-reproduction-2026-10-11.md#all-u-global-seeds-and-early-variants-separated-by-budget)
show checkpoint fluctuations often larger than differences between recipes.
The [cooldown RMSE curves](early-fine-cooldown-2026-10-08.md#rmse-only-comparison)
suggest a more stable late trajectory: mean scores over the same final five
scheduled validations improve by 3.1–6.3% across the five models. Those points
are correlated, and resumed runs diverged before decay began; this is not an
independent-seed confidence interval or an isolated causal LR estimate.

The [older observation-only comparison](global-physical-day30-2026-09-30.md)
also supports testing longer training: day-30 SST RMSE improved from 0.694°C
at 8k observation updates to 0.572°C at 16k. Its different training/selection
recipe makes it motivation, not a matched learning-curve extrapolation.
**16k observation-only updates are not 16k mixed-task observation updates:**
the proposed observation-heavy 16k endpoint has **3.5k OM4 and 12.5k observation
updates**. Matching 16k observation updates would require 20k total updates
under the proposed schedule; examples per update must also match.

Questions:

1. How much variation comes from evaluation, individual training updates,
   task interference, numerical execution, and independent training seeds?
2. Can smaller terminal learning rates and weight averaging yield reliable
   endpoint quality rather than occasional favorable checkpoints?
3. Under one stabilized recipe, do earlier data, fine supervision, and recurrent
   latent channels improve learning speed, final skill, or annual rollouts?

## Models and fixed science

All processors evolve the full **180×360 global 1° state**. Preserve the existing
D-style approximately 63M-parameter ConvNeXt U-Net widths/depth, InstanceNorm,
physical losses, normalization, five-day step convention, history length,
forecast leads, geometry/task inputs, global masks and chronological splits.
Use the existing recent OM4 period (1975 onward within the training split),
1958–1974 early period, and observation products/ERA5 preparation from the
[6k methods](early-rmse-6k-2026-10-10.md#runs). Pin actual datasets, calendars,
normalization and source hashes before execution; no new data release is implied.

| Proposed literal run name | Recent OM4 updates at 16k | Earlier OM4 updates at 16k | Observation updates | Recurrent state / fine task |
|---|---:|---:|---:|---|
| U-global-stable16k | 3,500 at 1° | 0 | 12,500 at 1° | 77 physical channels |
| U-multitask-early-stable16k | 1,750 at 1° | 1,750 at 1° | 12,500 at 1° | Same 77 channels |
| U-global-latent-stable16k | 3,500 at 1° | 0 | 12,500 at 1° | 77 physical + 10 latent channels |
| U-multitask-early-latent-stable16k | 1,750 at 1° | 1,750 at 1° | 12,500 at 1° | Same 87-channel recurrent state |
| U-multitask-early-fine-stable16k | 1,750 at 1° | 1,750 at ¼° | 12,500 at 1° | 77 coarse channels; existing fine encoder/decoder |
| U-multitask-early-fine-latent-stable16k | 1,750 at 1° | 1,750 at ¼° | 12,500 at 1° | 87 coarse channels; existing fine encoder/decoder |

The fine task consumes global **720×1440** examples, encodes to 180×360, and
supervises decoded 720×1440 fields. It is neither a 128×128 regional patch task
nor a native-quarter global processor. The existing encoder uses strided
convolutions/GELU and the decoder uses convolutions/PixelShuffle. Latent slots
are initialized and autoregressed on every task, with no direct latent target.
Report exact parameter counts from instantiated models, including adapters.

The recent-only latent control completes the early-data × latent-state comparison;
the old 8k+8k memory experiment does not replace this matched control. No wider
backbone, attention, diffusion, auxiliary EKE loss, or LLC data are introduced
in the initial screen. The post-screen review can propose targeted modeling
changes, with matched controls, before freezing the longer-run recipe.

## Stage 1: measure what is fluctuating

Use U-global and U-multitask-early first, with retained optimizer checkpoints.
Confirm checkpoint inventory rather than assuming every plotted step was saved.
Use the retained total-update **2,667** fork when available; if necessary, replay
once to a common 3,000-update checkpoint, save it, and fork every treatment from
that same serialized model, optimizer, RNG and sampler state. Reuse no fitting
or diagnostic weights in final fresh-run comparisons.

- **Evaluation repeatability:** evaluate a fixed checkpoint five times on the
  identical nine validation origins, including two fresh-process reloads. Log
  component scores and prediction differences; also run a deterministic CUDA
  reference. Investigate changing inputs, evaluation masks, train/eval mode or
  state mutation before blaming optimization. Proposed practical target:
  repeated blend variation below 0.1% relative, an order below a 1% effect.
- **Training trajectory:** for 100 diagnostic updates, log task identity,
  sample IDs, recent task history, LR, loss components, pre-clip gradient norm,
  clipping frequency, and actual parameter-update norm. Probe fixed training
  examples from both sources before/after every update; run the full frozen
  observation validation every ten updates during this short segment.
- **Task interaction:** measure gradients on fixed training-only batches for
  observation/recent OM4/early OM4 separately, including initializer and
  processor norms and pairwise cosines. A negative cosine is local evidence,
  not proof that an entire task is harmful. Reuse existing gradient-diagnostic
  machinery where compatible with this runner.
- **Execution variation:** repeat the unchanged segment from the same state on
  the same runtime/GPU configuration, using the runtime chosen for both new continuation
branches rather than comparing a fresh resume to an old uninterrupted run.
Separate same-seed execution divergence
  from seed variation and from evaluation repeatability. Deterministic
  reference replay remains a correctness test, not a promise of identical
  production trajectories.

Deliverable: one compact diagnostic figure/table identifying the dominant
source(s) of variation. Pooling across locations cannot remove a model-wide
prediction shift, and the validation months are not independent spatial trials.

## Stage 2: small paired stabilization screen

For each of the two models, use the same common 3k fork, seed 1729, task order,
training examples, optimizer moments, precision and host/runtime. Continue to
4k with three treatments; also repeat the unchanged treatment once to quantify
execution variation. This is **eight 1,000-update continuations**, plus Stage 1,
not eight new full-length runs.

| Treatment | LR after the common fork | Purpose |
|---|---|---|
| S0: constant-rate control | 1e-4 | Reproduce the fluctuations; duplicate execution supplies a control |
| S1: lower constant LR | 3e-5 | Test reduced update size while preserving capacity to learn longer |
| S2: cooldown | Linear 1e-4 → 1e-6 over updates 3,001–4,000 | Test settling toward a stable endpoint |

Maintain raw and **EMA** weights in all branches; EMA has a fixed 100-update
half-life (decay = 2^(-1/100)), initialized from the fork weights. Update it once
per optimizer step, checkpoint it, and evaluate it separately. Explicitly define
buffer handling; these models use InstanceNorm, so BatchNorm recalibration is
not an assumed solution. EMA is a model to evaluate, not smoothing of the curve.
No half-life sweep is proposed in this first screen.

Choose one LR/weight policy for every long-run arm using only validation.
Primary screen criterion: the mean RMSE blend at the five scheduled updates
3,600/3,700/3,800/3,900/4,000, averaging equally across the two model families;
show each family separately. Secondary criteria: endpoint score, maximum late
score, adjacent-checkpoint changes, and progress over the segment. Do not favor
a flat but worse trajectory. If policies differ by less than 1% or their ordering
changes between the two families, carry the best two into a seed-1730 short
confirmation before freezing the policy. If still indistinguishable, prefer
S1 with raw weights unless EMA improves both families; report the ambiguity.
This is a screening rule, not a statistical significance threshold.

### Review after Stage 2: decide what to test next

Stage 2 is a decision point, **not an automatic commitment to the long-run
matrix**. Present the LR/raw/EMA curves, task-gradient diagnostics and duplicate
execution spread, then agree whether to proceed or run a small follow-up.
Improvement from LR/EMA does not rule out further improvements from batching or
modeling. Freeze the complete recipe only after this review.

The current runner already accumulates **eight examples per optimizer update**.
Candidate follow-ups, chosen from the measured failure mode, are:

| Evidence | Candidate experiment | Necessary control / interpretation |
|---|---|---|
| Large variation between examples or update directions | Accumulate 16 or 32 examples instead of eight | Match task-specific example exposure and report optimizer steps separately; at equal examples, these use half or a quarter as many steps. Start with unchanged LR, without automatic linear LR scaling. |
| Observation validation repeatedly worsens immediately after OM4 steps; task gradients disagree | Combine OM4 and observation gradients before an optimizer step | Keep effective batch and aggregate task weights fixed where possible. For the 7:1 tail, compare eight-example single-task updates with a batch of seven observation examples plus one OM4 example. This changes Adam's gradient/moment sequence, not the aggregate data mix. Balance recent/early OM4 examples across steps. |
| Initializer quality or its interaction with the processor dominates the fluctuations | Test separate initializer/processor learning rates, or briefly freeze the initializer | Change one item at a time from a common checkpoint; retain a jointly trained control and inspect own initialized persistence as well as evolved skill. |
| Short-lead improvement comes with worsening long-lead dynamics | Test a modest increase in training rollout horizon | Match source/example exposure and report extra compute; compare fixed 30-/365-day outcomes, with checkpoint selection still using the frozen validation metric. |
| Smooth, reproducible curves plateau with remaining training error | Test a targeted capacity change, such as width or latent-state capacity | Use fresh matched controls; a changed architecture is a new experiment, not a resume or a noise-reduction claim. Do not infer a capacity limit from validation loss alone. |

Select one or two tests with a clear diagnostic motivation, rather than a broad
hyperparameter grid. Confirm any adopted change on both U-global and non-fine
early-OM4, and distinguish improved mean skill from merely reduced fluctuation.
No automatic gradient surgery or loss-weight search. These are proposed options
for review, not additional authorized runs.

The provisional long-run LR rule is to reproduce the selected change at update
3,000 and hold its terminal LR thereafter; if EMA is selected, start it at 3,000.
Review learning progress before locking that rule: settling successfully at 4k
with a 1e-6 LR does not establish that it will learn effectively to 16k. Any revised
LR schedule must be common across matched arms. If accumulation is adopted,
restate all milestones in examples consumed, rescale EMA's half-life to preserve
its example horizon, and redefine optimizer-step counts explicitly before launch.

## Stage 3: longer paired reproductions

Start fresh with **paired seeds 1729 and 1730**, using the frozen stabilization
policy. First run U-global-stable16k and U-multitask-early-stable16k (four runs).
After successful bring-up and review of their trajectories, run the other four
recipes with the same two seeds (eight additional runs). This puts the user's
non-fine comparison first while retaining the full requested early-data family.
Use identical initial weights within architecture-matched pairs; shared physical
weights across latent/no-latent architectures do not make them identical models.

**One schedule with an observation-heavy, extendable prefix:** preserve the
historical 4k task schedule for updates 1–4,000 (2k OM4 + 2k observations).
That schedule uses cumulative observation count
`floor(2000 * (f/4 + 3*f*f/4))`, where `f = updates / 4000`;
its instantaneous observation share rises from 12.5% to 87.5%. A 50:50 extension
would therefore reverse the intended late emphasis on observations.

After 4k, repeat **seven observation updates followed by one OM4 update**
(87.5% observations). This carries forward the original schedule's terminal mix
while retaining some OM4 supervision. Retain task-specific sample counters and
shuffle streams; within each early arm, alternate recent/early OM4 slots.
This ratio is a concrete default, not an established optimum. Review any change
before launch and apply it to all matched arms. No endpoint-specific stretching
or restarted data epochs.

| Total updates | All OM4 | Observations | Recent / early OM4 in early arms |
|---:|---:|---:|---:|
| 4,000 | 2,000 | 2,000 | 1,000 / 1,000 |
| 6,000 | 2,250 | 3,750 | 1,125 / 1,125 |
| 8,000 | 2,500 | 5,500 | 1,250 / 1,250 |
| 12,000 | 3,000 | 9,000 | 1,500 / 1,500 |
| 16,000 | 3,500 | 12,500 | 1,750 / 1,750 |

The new 4k milestone matches the original 4k task exposures and ordering;
changed LR/EMA policy still makes it a new treatment. The new 6k milestone has
**more observation exposure and less OM4 exposure** than the old stretched 6k
schedule (3k of each), as well as different ordering. Label that historical
comparison accordingly; the new matched controls support the new comparison.
Stage 2 and the long runs share the same 4k schedule prefix. Report against both
total updates and observation examples consumed. The table assumes the existing
batch of eight examples per optimizer update; fine tasks cost more, so this is
update/exposure matching, not exact FLOP matching. If the post-screen review
adopts larger or mixed-task batches, replace the table with explicit optimizer
steps and per-task example counts before launching Stage 3.

Retain immutable raw/EMA checkpoints at these milestones, validation-selected
checkpoints separately, and restartable latest state. Save full optimizer/RNG/
sampler/EMA state. End at 16k for the first report. A possible **20k extension**
would reach 16k observation updates and 4k OM4 updates with this same prefix and
batch size; decide after the 16k report and explicit approval. Matching observation
exposure still does not make this the same model/training recipe as the older
observation-only experiment. Falling validation loss motivates checking, but
does not guarantee future improvement or annual forecast stability.

## Evaluation and decision rules

- Keep the existing **five-component full-field RMSE-only validation blend**:
  SST, SSH-derived geostrophic velocity, EKE, and OHC 0–700/700–2000 m, with
  frozen climatology denominators, masks and nine validation origins. No
  detrending/deseasonalization or spectral selection. Validate every 100 total
  updates and all retained milestones, with the same triggers across models.
- Show raw and EMA curves separately. Primary model comparison uses the frozen
  weight policy's fixed 16k endpoint across paired seeds. Also report fixed
  4k/6k/8k/12k checkpoints, predefined last-five validation means, and the
  validation-selected checkpoint; do not choose whichever summary favors an arm.
- After checkpoint/policy choices are frozen, score all endpoint milestones on
  the existing 96-origin short-rollout cohort. Run the existing three-origin
  365-day evaluation at 6k/8k/16k plus the validation-selected checkpoint,
  deduplicating identical weights. Annual evaluation uses its existing
  four-component blend and reports EKE separately. Annual results do not select
  checkpoints. These previously viewed test cohorts are exploratory development
  evidence, not an untouched final test.
- Report every paired-seed difference, its mean, physical components, own
  initialized persistence and seasonal climatology. Two seeds are an initial
  replication, not a reliable significance estimate. A third paired seed is a
  proposed tie-breaker requiring review. Date resampling assesses date variation;
  correlated checkpoints or ocean pixels cannot stand in for seed replication.
- Include matched day-30/day-365 maps at 1:1 or 2:2 pixels per grid cell, monthly
  temperature profiles, and initializer-versus-evolution diagnostics. Retain
  common masks/scales. Distinguish reduced RMSE from preserved anomaly amplitude.

A credible benefit should persist at fixed endpoints across both seeds, have
interpretable component changes, and not rely only on one validation minimum.
If stabilizing training removes earlier-data gains, report that result rather
than introducing new variants until a favorable minimum appears.

## Implementation and execution

This PR contains the plan and CI configuration; it launches no experiments.
Implementation will need: an explicit
prefix-preserving task schedule, EMA state/evaluation, a recent-only latent
arm, and diagnostic logging. Use the existing experiment runners and tests;
do not assume mainline EMA/config plumbing automatically applies to them.
Qualify schedule counts at 4k/6k/16k/20k, matched data streams, raw/EMA serialization
and resume, no validation-input mutation, and deterministic reference replay.
Pin tested source/data/runtime contracts before GPU submissions.

Prefer Torch RTX6000 availability for these 1° runs; Beta is also acceptable.
Keep matched pairs on the same runtime and GPU type, with equal batch/precision.
Use a short real-data throughput/utilization check to set allocations, split
long runs into exact-resume segments if needed, and report actual GPU-hours
including retries. Twelve 16k runs are the full proposed replication scope;
qualification, screen and evaluations are additional. Do not promise a one-day
campaign from a short synthetic timing. Enable and verify online W&B. Use the
updated async-job-followup quiet monitor with durable completion callbacks.

## Review decisions

Recommended first scope is the three-treatment diagnostic screen, followed by
a review of stabilization and modeling alternatives. The six literal models above,
two paired seeds and 16k total updates are the provisional longer-run matrix;
freeze or revise it at that review, including the observation-heavy schedule.
Approval is needed before implementation/compute for this new wave; this draft itself authorizes
none. The first long four-run pair is reviewed before releasing the remaining
models. A 20k extension, third seed, changed accumulation/mixing algorithm, modeling
change or changed loss weights would be separate decisions. The current
native-quarter runs are independent.
