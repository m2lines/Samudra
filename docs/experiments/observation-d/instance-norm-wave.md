<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# InstanceNorm follow-up: authorized execution plan

25 September 2026. User authorized the follow-up with **existing OM4 forcings,
ERA5 adapter and separate observation reconstruction retained**. No ERA5-for-OM4
substitution. One seed initially. Unconstrained velocity/deep-state channels are
allowed; intervention diagnostics examine their use, not physical accuracy.

## Qualification gates

Fresh affine InstanceNorm with no running statistics throughout initializer and
evolution; original BatchNorm checkpoints are not valid InstanceNorm warm starts.
Default BatchNorm paths remain available for reproducing earlier results.

1. CPU tests: no BatchNorm/running buffers remain in converted modules;
   train/eval agreement, per-sample batch independence and affine gradients.
2. Real observation fitting probe with fresh InstanceNorm weights: finite loss,
   gradient reach to initializer/evolution/adapter, decreasing training-only loss,
   strict save/reload and resume behavior. Existing nine-origin score is monitored.
3. Fresh OM4 joint-training qualification using original forcing channels and
   the existing wide D initializer/AR evolution architecture. No old pretrained
   BatchNorm dynamics may be loaded under the InstanceNorm label.
4. Qualify continuous-year data support and autoregression separately from
   model skill. Monthly prepared samples reset five-day phases; exact intervals
   must match before assembling annual forcing/targets. No nearest-time stitching.
5. Use measured throughput and stability to set production caps and pin source,
   data, normalization and scoring contracts before production submission.

## Main paths after qualification

- **InstanceNorm scratch:** random networks; 1,000 observation reconstruction
  updates, then initially 8,000 joint observation updates.
- **InstanceNorm OM4 → observations:** fresh OM4 pretraining using original OM4
  forcings; then the ERA5 adapter, the same observation reconstruction phase,
  and initially 8,000 joint observation updates.

Keep the auxiliary reconstruction term in joint training. Use effective batch
8, seed1729, equal small learning-rate search budgets and the existing history,
grid and data splits. Preserve each model's zero-update state and immutable
observation checkpoints at 10/25/50/100/250/500/1k/2k/4k/6k/8k, including early
reconstruction checkpoints. Separate scientific milestone weights from selected
best weights and resumable optimizer/RNG checkpoints. No source checkpoint is
selected with held-out observations.

Qualification initially changes normalization only and retains the existing
prepared data/scaling/operator definitions to isolate implementation behavior.
Before the main comparison, explicitly resolve the proposed common scaling and
corrected velocity-stencil reference as versioned contracts; do not silently
mix them with historical scores. Fresh training/qualification does not imply
that the original 100-hour wave's unused allocation authorizes arbitrary new
work: record a concrete bounded production budget from qualification throughput.

## Diagnostics and annual evaluation

Fixed maps/profiles, matched monthly reconstruction, inferred persistence and
30-day forecasts at retained checkpoints. At predetermined milestones, replace
or shuffle initial U/V and deep T/S to measure forecast sensitivity. Changes under
intervention demonstrate dependence, not physical identifiability.

One-year evaluation runs approximately 73 five-day steps, without gradients,
state resets or future surface corrections, with prescribed forcing. Track
lead-dependent integrated/spectral errors, amplitudes and heat-content drift;
include global maps with the supervised/scored domain marked. Use validation
origins for development; held-out cohorts remain separate. Annual diagnostic
failure must be reported, not hidden by selecting only stable examples.

Execution records and qualification results will be appended below. Production
is gated on evidence, not the presence of submitted jobs.


## Execution record: 25 September

The first fresh InstanceNorm probes passed under producer `b9f6e318`:

| Probe | Torch job | Evidence |
| --- | --- | --- |
| Observation fitting | 18528438 | Ten updates reduced training-only loss from 1.29126 to 0.24330; finite, nonzero gradient reached initializer, evolution and adapter. |
| Fresh OM4 joint training | 18528440 | Ten updates reduced validation T/S normalized MSE from 1.20469 to 0.48909; original OM4 forcings retained. |

These are implementation checks, not comparative skill results. Their allocated
GPU time was 282 seconds (0.0783 GPU-hours). The CPU code build was 18528423.

Producer `3dba9696` adds the final common observation-derived state scales,
version-3 observation scoring with complete observed ADT derivative-stencil
support, and dense checkpoints during both reconstruction and joint training.
OM4 forcing normalization is unchanged. The OM4 native/cache equivalence audit
remains in native coordinates; state conversion happens only at the model input
and target boundary. Eleven targeted CPU tests passed, including physical-unit
invertibility, dense checkpoint/resume behavior and the coastal-stencil regression.
Final-contract qualification build **18529768** was submitted; this is not yet
production or evidence of final-contract qualification success.

Continuous annual evaluation data were prepared on Empire AI in CPU job 101098
(87 seconds), following an import-path failure in 101097 (3 seconds). Four
365-day sequences start at validation 2013-11-01 and held-out 2015-01-01,
2018-01-01 and 2021-01-01. Each contains 19 history bins and 73 forecast bins,
with exact five-day intervals and monthly interior targets. Existing monthly
samples cannot be concatenated because their five-day phase resets. The new
bundle is 1,025,375,724 bytes; manifests record every payload hash. Its transfer
to Torch is pending explicit approval for OSN publication, following an automatic
approval-review rejection. No payload has been published.

Torch's user-scoped quota reports 3.84 TB used of 5 TB (about 1.16 TB headroom).
Reserve 100 GB for this wave's checkpoint copies, optimizer state, diagnostics
and annual data; preserve historical checkpoints. This is an estimate, to be
checked against actual usage before long runs.


A first calibration stage is queued behind final qualification: registration job
18529840 waits for build 18529768, then installs an `afterok` dependency on both
qualification jobs. Its execution also verifies their producer, successful
fitting gradients/loss and OM4 ten-update completion markers. Only then may it
submit two 500-update fresh OM4 learning-rate pilots (1e-4 and 3e-4, effective
batch eight, seed1729, 25-update warmup) and a three-update-per-phase observation
dense-checkpoint probe. Each has a one-GPU, one-hour scheduler cap: **three
GPU-hours maximum requested for this stage**, separate from the qualification
attempts. No main production is submitted by this gate. Observation LR pilots
and main production await its evidence; the two observation paths will receive
equal search budgets.
