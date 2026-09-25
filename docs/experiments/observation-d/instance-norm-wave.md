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


## Transfer and calibration update

The user approved OSN publication. Empire AI upload job **101147** completed in
seven seconds and its full download comparison found 421 matching files and no
differences. Torch DTN process **1623986** then downloaded the bundle to
`/scratch/jr7309/data/obs-d-annual-instance-v1`; another full read-back comparison
passed, followed by all **416 payload SHA256** checks across four origins.
`DATA_READY.json` records successful verification. The transfer blocker is resolved.

Final-contract qualification jobs **18530032/18530033** completed; observation
fitting loss fell to 0.24333 and OM4 completed ten updates with common scaling.
The dense observation probe **18530319** completed all three updates in both
phases, retaining phase-zero, update-one and update-three weights. Resume job
**18530614** completed without further training. The CPU resume regression also
checks retained milestone bytes.

| Calibration | Job | Completed updates | Selected validation result |
| --- | --- | --- | --- |
| Fresh OM4, LR 1e-4 | 18530317 | 500 | normalized subsurface T/S MSE 0.0205135 |
| Fresh OM4, LR 3e-4 | 18530318 | 500 | normalized subsurface T/S MSE 0.0156985 |

Both pilots used the same seed, effective batch eight and validation cohort.
**LR 3e-4 was selected.** Fresh OM4 main job **18530880** restarts from seed1729;
it does not continue the winning pilot. Its cap is 24,000 updates or eight
training hours, with six non-improving validations after at least two hours;
the scheduler allocation is capped at nine GPU-hours. Retain early OM4 weights
at 10/25/50/100/250/500/1k/2k/4k/8k/16k/24k when reached. This fresh joint
initializer/evolution pretraining is not a reproduction of historical D's
separate initializer and heavily pretrained evolution stages.

Scratch LR pilots **18530615/18530616** (1e-4/3e-4) are running with 1,000
reconstruction and 250 joint updates each. After successful completion, gate
**18531059** verifies exact update counts, producer and selected checkpoint
hashes, chooses the integrated-plus-spectral validation minimum, and submits a
fresh scratch main run at that rate. Main observation runs retain 1,000
reconstruction plus 8,000 joint updates, with separate two-hour/twelve-hour
training caps and a fifteen-GPU-hour allocation cap each.

After OM4 main completes, gate **18531062** verifies the selected source hash
and common-scaling statistics, copies immutable source weights, and launches a
transfer fitting qualification. Only its successful output permits two transfer
rate pilots (3e-5/1e-4), with the same 1,000 + 250 update search budget as scratch.
Their integrated-plus-spectral validation minimum chooses the transfer main rate.
Both main paths restart from their designated initial weights, not pilot weights.
The adapter remains present and learns during reconstruction/joint training;
there is no additional adapter-only phase in this wave.

Measured warmed OM4 throughput is approximately one optimizer update/second;
early observation reconstruction is roughly 4.5 seconds/update. The new wave
has a conservative **80 allocated GPU-hour all-attempt ceiling**, including
calibration, training, evaluation and recovery. This is a separate new-wave
ceiling, not a claim that the previous wave's allocation carries forward.
Requested primary allocations are substantially below it; do not use the
remainder for new scientific arms or seeds. Completed initial/final probes,
dense/resume tests and OM4 pilots consumed **0.6494 GPU-hours**, before active
scratch pilots and later jobs. Continue accounting for every actual allocation.

## One-year evaluator qualification

Evaluation-only producer `8056405ac6217fbe237859904889591a3b637e23` adds a
continuous 73-step evaluator; training stays pinned to `3dba9696`. Four CPU tests
cover calendar overlap, no future-surface leakage, single initialization with
state propagation, and named velocity dimensions. Real validation-only job
**18530921** is queued on the three-update qualification checkpoint; it tests
execution rather than scientific skill. Training is not selected on that result.

The evaluator exports full-grid initial states and forecast states at days
5/15/30/90/180/365. Scoring retains the fixed 60S–60N domain with observed
ADT derivative support for geostrophic velocity. Calendar-month OHC uses exact
bin overlap weights. Continuous-year EKE is explicitly centered on each year's
own mean and is not the fixed-lead, multi-origin EKE used for training selection.
Prescribed future ERA5 is supplied through the adapter; this is a conditional
ocean rollout, not an operational prediction of future atmospheric forcing.


Annual GPU qualification **18530921 completed in 33 seconds**, including all
73 autoregressive steps, twelve calendar-month OHC evaluations and map exports.
The checked completion marker identifies the exact evaluation producer, input
manifest and checkpoint. This validates execution only: the input was a
three-update fitting checkpoint. Fresh OM4 main **18530880 is now running**,
with finite structured training losses and roughly one second/update after
cache preparation. Successful completed GPU allocations through annual
qualification total **0.6586 hours**, excluding active jobs.

Dependent evaluation registration is installed for both main paths. Monthly
held-out evaluation requires both completed main selection and successful
immutable OM4 source export; annual validation/test evaluations require
completed main selection. Per arm, their combined scheduler caps are two
GPU-hours. Failures in preceding stages block their descendants. No held-out
result feeds the rate or checkpoint selection gates.


## Model and training definitions for the new comparison

| Literal model name | Meaning |
| --- | --- |
| InstanceNorm scratch | Fresh initializer, evolution and zero-output ERA5 adapter; observation reconstruction followed by joint observation training. No OM4 weights or OM4-derived state normalization. |
| InstanceNorm OM4 → observations | Fresh InstanceNorm initializer/evolution jointly pretrained on OM4, selected by OM4 validation; immutable selected weights then receive observation reconstruction and joint training with the ERA5 adapter. |
| Fresh InstanceNorm OM4 source | The selected OM4 checkpoint before observation adaptation; source of the preceding transfer model, not historical BatchNorm D. |

The initializer is the wide D ConvNeXt U-Net (widths 256/384/512/768,
approximately 121.7M parameters). The evolution U-Net uses widths 128/192/256/384
(approximately 31.6M parameters). The pointwise 8→32→3 ERA5 adapter has 387
parameters. Total is approximately 153.3M trainable parameters; replacing
BatchNorm with affine InstanceNorm preserves affine parameter counts and removes
running-statistic buffers. The initializer consumes 19 five-day surface/forcing
frames and geographic/seasonal context and emits two 77-channel states; the
surface values are copied as in D. Observational joint training forecasts the
roughly one-month target window; evaluation can autoregress 73 steps without
increasing the training backpropagation horizon.

Both observation paths use the same 180 × 360 Gaussian grid and shared static
model masks, conservatively remapped observational inputs, and the same
observation-training-derived state scaling,
effective batch eight, seed1729, AdamW with weight decay 0.01, gradient clipping 1,
BF16 and 25-update learning-rate warmup per observation phase. Core rates are
chosen independently from equal-budget pilots. Adapter rates remain 1e-3 during
reconstruction and 1e-4 during joint training. Joint loss retains 0.8 monthly
interior T/S, 0.1 SST and0.1 ADT normalized forecast terms, plus the 0.1 auxiliary
reconstruction term. Surface temperature is excluded from interior supervision;
velocity and unobserved deep T/S have no observational reconstruction targets.
Learning-rate differences are part of the calibrated training recipes, so the
comparison is not a strict weights-only intervention.

Training selection remains the fixed nine-origin v3 integrated-plus-spectral
observation score. Annual evaluation and initializer interventions do not select
checkpoints. Report both observation updates and total allocated GPU-hours,
including OM4 pretraining and calibration separately; update counts are not
claimed to be exact FLOP matches.

## Checkpoint state diagnostics

Producer `5ffcf10069bd25fb61eb6629f8e0cc8d12afe705` adds validation-only
initializer maps/profiles and controlled initial-state interventions. Real GPU
qualification **18532019 completed**, exporting both fixed validation map cases,
nine-origin depth-profile statistics and integrated/spectral changes under
zeroed initial U/V or mean-reset deep T/S. Two CPU tests check that interventions
preserve the other channels and that the unmodified diagnostic rollout matches
the native model rollout exactly. The qualification checkpoint is only a
plumbing test, not evidence about either production model.

After each main run, registered diagnostics cover reconstruction updates
0/10/25/50/100/250/500/1k and joint updates 0/10/25/50/100/250/500/1k/2k/4k/6k/8k.
Maps contain the full grid; profile reductions and scores use the fixed scored
domain. Monthly reconstruction uses target-month surfaces and is labeled
separately from forecast skill. Additional continuous-year validation diagnostics
are registered at joint 0/100/1k/4k/8k. These predetermined diagnostics run after
selection and do not tune either model.


## Reconstruction handoff correction — 25 September, 14:50 ET

The completed scratch pilots revealed a handoff defect: each performed 1,000
reconstruction updates, but its frozen random evolution model made the
forecast-selection score worse throughout reconstruction. The phase's best
forecast score remained the untrained initial value **5.410358**. Consequently,
the old handoff loaded the initial initializer and discarded all reconstruction
learning before joint training. The earlier scratch LR scores therefore measure
that old recipe and are **superseded for selecting this wave's rate**.

Scratch main **18536182 was stopped during reconstruction**, with all files kept.
Its downstream evaluations and the old transfer-export gate 18531062 were retired.
OM4 main 18530880 continues unchanged: this defect concerns only the observation
phase boundary. No reconstruction or checkpoint files were deleted.

The correction explicitly uses `--reconstruction-handoff last` for both new
observation paths. The state after the full 1,000 reconstruction updates seeds
joint training, including when resuming after a completed reconstruction phase.
Final checkpoint selection still uses the same integrated-plus-spectral minimum;
this fix changes the training handoff, not the selection metric. Legacy `best`
handoff remains available for reproducing previous runs. A regression test
forces non-improving forecast validation and proves the learned final state is
handed onward while best-score weights and retained milestones remain unchanged.
All 26 relevant tests passed.

Replacement observation runs will use a new pinned producer and fresh directories
under `handoff-v2`, with new fitting/dense/resume qualification and equal-budget
rate pilots for both paths. No old phase state or optimizer is silently resumed
under the new contract. The existing successful OM4 pretraining and immutable
source export remain reusable; only observation training/calibration is replaced.
All superseded allocations remain in the 80 GPU-hour all-attempt ceiling.


Recovery producer **`d186758213bf556456de319bba9e311c604bacd2`** is now pinned.
Fresh fitting 18537338, dense phase 18537397 and resume 18537405 completed in
42/101/24 GPU-seconds. CPU audit 18537484 then loaded the actual full-model
checkpoints and verified every tensor: joint update zero equals reconstruction
last exactly and differs from reconstruction best. Only after this proof did it
submit replacement rate pilots **18537519/18537520**, each with the full
1,000 + 250 update budget and the corrected handoff. No old phase state or
optimizer was imported. Their main-training, source-transfer and evaluation
registrations now live under `handoff-v2`; historical records are retained.

The canceled old scratch main consumed 267 GPU-seconds. An initial attempt to
register the audit against an expired Slurm build dependency was rejected
before allocation; accounting-proven completion resolved it. The successful
audit preserves its qualification dependency proof. Reserve approximately 200 GB
for all main and calibration checkpoint copies, diagnostics and retained
superseded attempts; the previously measured 1.16 TB quota headroom is ample.
