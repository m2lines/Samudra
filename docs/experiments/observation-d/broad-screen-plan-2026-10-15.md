<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Proposed broad observation-model screen

Status: proposal for review; no jobs submitted. This supersedes the proposed
long-run matrix in the October 13 stabilization plan, not its completed results.
The common cooldown follows the [completed comparison](cooldown-standardization-2026-10-14.md).
The question is which data, task and capacity allocation gives the best
observation-initialized forecasting at a practical training budget.

## Evidence motivating the design

The newer [OE-generative adaptation report](https://github.com/m2lines/Samudra/blob/experiment/score-trained-samudra/experiments/score-trained-samudra-1deg/adaptation/RESULTS.md)
reports a 2.97% 360-day RMSE advantage for multitask-pretrained M over adapted F,
after 30 forecast-only adaptation epochs. Its 360-day temperature/salinity errors
still lag F by 12.51%/10.24%; velocity drives the aggregate gain. Completion RMSE
and coverage deteriorate substantially. This is one seed, synthetic OM4 targets,
and a different 84M-parameter stochastic model with two-input/two-output calls;
it is evidence for testing transfer, not a measured observation-model improvement.
The report was read at commit `39bfabcb07564eb2a576b0a111dc1529ee5c9fb8`.

M's training tasks were forecasting, surface-to-interior completion, missing-variable
completion, and spatial-gap completion. Its original task mixture was 50/25/12.5/12.5.
Total calls were matched, but forecast exposure was not. We will preserve that
distinction in the screen and add exposure-matched controls for promising results.

Our own cooldown comparison supports observation-only adaptation while retaining
initializer reconstruction/completion losses. Removing these losses changes the
inferred interior substantially for a 0.10% short-lead improvement. Annual OHC
regresses for early-OM4 even when SST improves. Neither a pooled score nor an
unchanged initializer alone establishes scientific quality.

## Fixed comparison protocol

- Global observational forecasting remains at 1°, 180×360, with the existing
  19-bin observation history, 77 physical channels, forcings, masks, normalization,
  splits and RMSE-blend denominators. Fine inputs/outputs, when used, are native
  ¼° OM4 patches through resolution-specific encoder/decoder layers; the global
  observation task remains unchanged. Reuse the earlier fine experiment's exact
  patch extent and mapping, recorded in each resolved configuration.
- Train fresh models, with paired seed 1729 initially. Do not initialize the new
  search from historically different selected checkpoints. Re-run U-global and
  early-OM4 controls under the new fixed schedules; old curves are references.
- Hold LR and source/task fractions constant throughout main training. Define
  seeded task streams independently of the intended stopping budget, with balanced
  blocks for discrete proportions. Extending 4k→8k→16k must preserve the prefix.
- Initially retain accumulation eight, optimizer settings, physical losses and
  EMA half-life 100 updates. Evaluate raw and EMA during main training. No
  automatic LR scaling when capacity changes.
- At 4k, 8k and 16k main-training milestones, fork a **separate** 1k cooldown:
  observations only, existing forecast/reconstruction/completion losses, linear
  LR decay from that parent's LR to 1% of it, EMA reset to raw at the fork,
  Adam state retained. Evaluate the fixed final EMA; raw is a sensitivity check.
  Continue longer main training from the untouched constant-LR branch, never
  from a cooled branch. Total budgets are therefore 5k/9k/17k including cooldown.
- Log source/task examples, optimizer updates, actual processor calls, parameter
  count and allocated GPU-hours. Equal updates are the initial engineering budget;
  different sizes, fine decoders or shorter gradient paths need not cost equally.

The chosen cooldown retains the existing observation auxiliary losses. It does
not retain newly added OM4 completion updates: those are part of main training.
Keep pre/post-cooldown completion evaluations so this choice cannot hide forgetting.

## Three independent mixture axes

1. **Observation fraction p:** fraction of optimizer updates on the existing
   observation task. The rest use OM4. Obs updates retain their existing losses.
2. **Early OM4 fraction e:** among OM4 updates, fraction from the pre-1975 archive;
   remaining OM4 updates use the existing recent archive. This is independent
   of whether that OM4 update is forecasting or completion.
3. **Completion fraction q:** among OM4 updates, fraction assigned to a conditional
   completion task. Remaining OM4 updates are forecasts. Within completion, start
   with 50% surface-to-interior, 25% missing-variable and 25% spatial-gap tasks.

For example p=.75, e=.5, q=.5 yields 75% observation updates, 12.5% OM4 forecasts,
6.25% surface completion, 3.125% variable completion and 3.125% gap completion.
Each OM4 task is split equally across early/recent dates. Adding completion does
not silently reduce the observation exposure at a fixed p. It does replace some
OM4 forecast exposure, which the later matched controls must address.

## What the new tasks actually train

The existing observation reconstruction loss primarily teaches the initializer.
Simply relabeling it as an OE-generative task would not test the same hypothesis.
The new tasks must also train the **shared evolution processor** used at inference.

Use a common, explicit supplied-value/availability/request interface, with the
same interface enabled in its forecast-only control. A small conditioning
projection may start at zero so the unmodified forecast is reproduced initially.
Keep the current processor's physical channels and forecast stepping initially;
do not silently import OE-generative's different history length, multi-output
architecture, ensemble objective or model size.

| OM4 task | Information supplied | Requested targets and loss |
|---|---|---|
| Forecast | Ordinary earlier state/history and allowed forcing | Future state; no future ocean fields |
| Surface → interior | Same-time SST and SSH, masks and allowed forcing; hidden historical full states | Other wet state entries, including subsurface T/S and velocities |
| Missing variable family | Same-time fields except one of u, v, T, S or SSH; hidden historical full states | The hidden family only |
| Spatial gap | Same-time fields outside a masked rectangle; hidden historical full states | All wet channels inside the rectangle |

For the first task-transfer experiment use deterministic physical losses, normalized
within requested variable groups, and record missingness explicitly. Never score
copied conditioning fields. Completion labels come from permitted OM4 training
windows; sparse observations do not supply fictitious dense subsurface labels.
At forecast validation and deployment, known future fields are always absent.

Qualification must verify no future-input leakage, requested-only gradients,
known-value copying, physical channel/mask ordering, conditioning parity for the
control, and exact task-RNG/checkpoint resume. The processor's completion interface
is a concrete implementation prerequisite, not functionality assumed to exist in
the present runner. Preserve global geometry and periodic longitude for gaps.

The stochastic almost-fair-CRPS model is a separate possible follow-up. First test
whether these tasks help the deterministic observation system. If we include a
stochastic arm later, pair it with the same-interface deterministic arm and score
ensemble means for RMSE plus CRPS, coverage, and temporal continuity separately;
a stochastic objective must not be confounded with task mix.

## Proposed screen and promotion sequence

| Stage | Configurations | Budget and purpose |
|---|---:|---|
| Core factorial | 24 | 2 LRs × 3 observation fractions × 2 early fractions × 2 task mixtures; 4k main + 1k cooldown |
| Anchors | 4 | Two obs-only LRs; two legacy-interface controls corresponding to recent/early baseline cells; same budget |
| Conditional alternatives | 12 new | Paired architecture/task alternatives around competitive core recipes; 4k + 1k |
| Medium training | 12 of the 40 | Continue uncooled parents 4k→8k, then a fresh 1k cooldown |
| Long training | 6 of those 12 | Continue 8k→16k, then a fresh 1k cooldown |
| Confirmation | Best 3 recipes × 2 extra seeds | Fresh runs to 16k + 1k; include the matched baseline among the three |

Core values: LR **3e-5 / 1e-4**; p **.50 / .75 / .875**;
e **0 / .50**; q **0 / .50**. Hold baseline width, zero extra latent channels,
and coarse OM4 in this factorial. Obs-only anchors use p=1 and have no meaningless
e/q choices. The legacy-interface controls use LR=1e-4, p=.75, e=0/.5, q=0;
compare them with the matching common-interface cells to detect interface effects.
For q=0 the common-interface model is forecast-only with respect to OM4; it still
has the existing observational reconstruction/completion losses.

Choose the 12 conditional alternatives in matched pairs after seeing the core:

- **Four fine/latent cells:** early coarse versus fine, each with 0 versus 10
  autoregressed latent channels, at one fixed competitive LR/data/task recipe.
  Use the same baseline backbone width in all four cells,
  so width is not bundled into the fine/latent claim. Latent channels are present
  on all tasks and autoregressed; fine encoding may populate them.
- **Two width cells:** baseline versus 1.5× backbone width at a second matched
  recipe. Record actual parameters and throughput rather than assuming scaling.
- **Two fine-skip cells:** a shared-scale fine-resolution skip versus the plain
  fine bottleneck at otherwise identical settings. This refers to the fine-scale
  path discussed in “Assess multiresolution U-Net sharing,” not recurrently passing
  a high-resolution physical state. Freeze the exact graph before implementation.
- **Four task-allocation cells:** q=.25 with the combined completion mixture,
  and three single-task ablations (surface, variable, gap) at the same q. Use the
  corresponding q=0 core cell as a reference; preserve p/e/LR/width.

These are 12 *new* runs even if some duplicate an earlier setting: duplicates are
useful matched controls. The within-four fine/latent comparison isolates factors
at the common baseline width; the separate width pair tests increased capacity. If a paired block's parent setting differs
from a core cell, explicitly include its q=0 control in the 12-run allocation by
replacing a lower-priority alternative, not by comparing unmatched settings.

Do not prune on raw pre-cooldown minima or stop a task family just because it learns
more slowly at 4k. Of 12 medium-budget slots, reserve four for diversity: the best
baseline, best task-augmented model, best early model, and best fine/latent model
(distinct slots, filling duplicates with the next relevant candidate). Fill eight
by validation performance and component tradeoffs. Carry at least one task-augmented
and one no-new-task model to 16k. This is a safeguard against the delayed crossover
seen in OE-generative, not a guarantee that 16k is sufficient.

The full proposed ladder totals about **416,000 optimizer updates**, counting
cooldown branches and six fresh confirmation runs, before qualification and any
exposure-matched follow-ups. This is not a GPU-hour promise: mixed tasks, width and
fine resolution alter cost. Profile each architecture/task type and publish a
compute cap before launching its stage. Approve stages separately; this proposal
does not authorize the full ladder or automatic replacement experiments.

## Selection and scientific controls

Primary ranking remains the frozen nine-origin short-lead observation RMSE blend
at final cooldown EMA. Also report its five components, individual leads, raw
endpoint, main-training curve, cooldown curve, and own initialized persistence.
Do not change the normalizing reference across trials or hide tradeoffs with a
new spectral/composite objective.

Run the same annual validation origin on all 4k finalists/alternatives, given its
low measured evaluation cost. Use annual SST, velocity and both OHC layers as
separate promotion diagnostics; retain Pareto alternatives rather than selecting
only the pooled-score winner. Pre-register a 5% annual component regression versus
the matched same-budget baseline as a review flag, not a statistically calibrated
rejection threshold. Flagged models cannot become the sole winner on short score.
Track initializer T/S profiles, OHC and OM4 completion retention before/after
cooldown. For stochastic models also track per-member cross-call increments.

The one annual validation origin is already inspected and can be overfit. Do not
use the previously reported 2015/2018/2021 test cohort for repeated promotions.
Before final confirmation, audit a disjoint time-held-out cohort not used to design
these choices; if none is available under the data split, state that limitation
and treat the final evaluation as reused-test evidence. More origins do not
substitute for independent training seeds.

At confirmation, compare paired seed differences, reporting each seed as well as
the average. Resampling forecast origins estimates evaluation uncertainty only;
it does not estimate training-seed uncertainty. Avoid treating overlapping windows
as independent replications. Report accuracy versus both allocated compute and
source/task exposure.

For a promising task or early-data effect, add two explicit causal comparisons
before claiming *why* it helps:

1. Same total update budget and architecture: completion replaces some OM4 forecast
   updates. This answers the practical compute-allocation question.
2. Same observation and OM4 forecast example streams/counts, with extra completion
   updates added in the treatment. This tests whether auxiliary supervision helps
   beyond that forecast exposure, but spends extra compute. Include a longer
   forecast-only control at the treatment's compute budget when interpreting it.

The broad search can find good recipes without proving an isolated task benefit.
These paired follow-ups and their budgets should be proposed once the candidate
and its actual task costs are known.

## Search tooling and decisions before launch

Use a small Optuna ask/tell controller around Slurm and the existing pinned
runner/qualification/callback machinery. Enqueue the factorial and matched controls
explicitly. Store resolved configurations, parents, task exposure and receipts as
trial metadata; a failed allocation is a failed attempt, not a bad scientific score.
Use a single writer to the study and keep checkpoints off its database.

A full Cartesian product over LR, source mix, tasks, fine adapters, latents, width
and budget is wasteful. After the fixed comparisons, conditional TPE can propose
local candidates in approved ranges (e.g. LR1e-5–3e-4, early fraction0–1, width,
latent0/10/32), with the same reporting contract. Its conditional-space support is
documented in [Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
Pin the installed API/version; no dependence on an unverified latest API is needed.
The first 40 configurations above are deliberately prescribed rather than handed
to a sampler that might confound task, data and architecture effects.

Decisions for review: approve the 28-run core or reduce it; confirm deterministic
processor task transfer first; and decide how much later fine/capacity exploration
to fund. Recommended starting scope is the core only, with the agreed cooldown
and the shared task interface qualified first. Keep LLC outside this screen until
its missing-source-data issue is independently resolved.
