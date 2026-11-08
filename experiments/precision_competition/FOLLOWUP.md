<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Learning rate, weight averaging, and introduced competition

This follow-up tests whether the first wave's unstable copying error can be reduced
by smaller late updates, and whether a useful second task then removes acquired
precision. It uses the same manufactured fields and transformer as the
[first wave](README.md), at width 128, depth 4, and two heads of width 64.
FP32 with TF32 disabled and L40S GPUs are fixed throughout, including continuations.
The first wave used six GPU types; the follow-up fixes GPU type across comparisons.

In the first wave, copying precision fluctuated after its initial improvement;
final squared bias explained 81–99% of solo-copy MSE. T, S, u and v are independent
zero-mean periodic random Fourier fields. T and S have identical distributions;
u and v have stronger fine-scale content. Each output query includes the current
field values and position. Both tasks share the input embeddings and transformer
blocks, with two attention heads per block and separate scalar output heads.
Copying targets T. Transport is `g(A,u) = -P_4 d(A*u)/dx / 4`, where `P_4` retains
Fourier modes through 4. The evolved-velocity target is
`roll(v,-1) + 0.1*g(v,v)`. All fields are supplied to every arm.

Effective RMS bits are `-log2(RMSE / 0.1292985189)`, using the fixed temperature
standard deviation. RMSE averages over random field realizations and uniformly
sampled spatial query positions. Raw means the current, unaveraged model weights.
Auxiliary normalized RMSE divides by target RMS on the same evaluation set;
predicting zero therefore scores exactly one. Loss-normalization constants instead
come from the separate calibration set. A bias correction is one scalar: subtract
the mean validation prediction error from every test prediction.

## Qualification: 36 runs

| Task | Learning rate | Schedule | Loss | Seeds | Runs |
| --- | --- | --- | --- | --- | ---: |
| Solo copying | 3e-4, 3e-5 | Constant; cosine | Original MSE | 11, 22, 33 | 12 |
| Solo transport auxiliary | 3e-4 | Constant; cosine | Original MSE; MSE / target mean square | 11, 22, 33 | 12 |
| Solo evolved-velocity auxiliary | 3e-4 | Constant; cosine | Original MSE; MSE / target mean square | 11, 22, 33 | 12 |

Each run has 12,000 updates, batch 64, and a 900-second measured-training cap.
The cosine schedule warms up for 100 updates, reaches 3e-7 at update 3,000, then
holds that rate through update 12,000. The constant schedule uses the same warmup.
Report both 3,000 and 12,000 updates, plus the full curves. Normalization constants
are fixed from a separate 8,192-example calibration stream (seed 660001); they are
stored in the manifest's scales file. These zero-mean targets make mean square a
variance estimate. Normalizing a loss is an explicit objective intervention.
Raw physical-unit error metrics do not change.

All runs maintain weight EMAs with decay 0.99 and 0.999, corresponding to roughly
100 and 1,000 updates of averaging. Each EMA starts from the same initialized
weights and updates after every optimizer step. It does not affect training.
Report both EMAs alongside raw weights at every evaluation; do not select the best
EMA on test data. Checkpoints preserve both EMAs. The raw model determines recipe
selection and all subsequent forks. For a readout diagnostic, fit an additive bias
correction on the final validation set and evaluate its error on the independent
test set. The uncorrected errors remain primary.

Select one common copying recipe across seeds by the largest worst-seed minimum
raw precision over the last three validation evaluations. Select one transport
recipe by the smallest worst-seed maximum normalized RMSE over those evaluations.
Neither test metrics nor EMA metrics enter selection. Qualification requires at
least six raw copying bits in every selected seed and transport normalized RMSE
below 0.8 in every selected seed. One normalized RMSE is the error of predicting
zero. If either requirement fails, report the failure and stop before competition.
The evolved-velocity qualification is diagnostic; it does not replace transport
in the copy-dependent comparison.

## Introduce competition, then remove it

The selected copying runs retain checkpoints at the first three consecutive
validation evaluations meeting 4, 6, or 8 raw RMS bits. Save the third checkpoint;
use only precision levels attained by all three seeds. These are sequential
validation-based starting-point choices, not test-selected results. Report each
checkpoint's actual precision and update count. A level can be reached earlier
than the final qualification window, so report subsequent solo stability as well.

At each eligible level, fork the exact same model weights, Adam moments, weight
EMAs, and RNG states into seven arms: main-only; independent auxiliary with weights
0.1, 1, 10; dependent auxiliary with weights 0.1, 1, 10. Independent transport is
`g(S,u)` and dependent transport is `g(T,u)` while copying T. The filtered product g
is defined in the first-wave protocol. All inputs and query counts remain matched.
The auxiliary head has not been trained during copying; its learning transient
is part of this intervention and must be reported.

Use the auxiliary loss scale selected during qualification. When normalized
transport wins, divide main MSE by the fixed temperature variance too; otherwise
use both physical-unit MSEs. Thus the weight sweep compares either dimensionless
losses or physical-unit losses, explicitly recorded in the decision file. All
forks, including main-only, receive identical scale changes and LR schedules.

Each fork runs 3,000 additional updates with a fresh 100-update warmup to 3e-5,
then cosine decay to 3e-7. Adam moments and step counters remain inherited; the LR
schedule and stage-duration counters restart. The main-only fork controls for
that intervention. Each arm has a 300-second measured-training cap. Save raw and
EMA trajectories, task accuracy, weighted shared-gradient norms and cosine, and
paired bits lost at equal stage updates. Bits lost are
`log2(RMSE_joint / RMSE_main_only)` within each seed and starting checkpoint;
positive means worse precision. Report this separately from each arm's change
relative to its starting precision. Shared-gradient measurements exclude both
output heads and use a fixed, separate batch (seed 880002), including the actual
loss scale and coefficient of each task. Report a competition effect alongside
whether the auxiliary actually became useful (normalized RMSE below 0.8
for three consecutive evaluations). This threshold is an operational skill test;
it does not prove transport is easier than improving copying precision. Regressing raw precision alone
does not establish an easier task outcompeting it.

For the weight-1 independent and dependent endpoints, fork another 3,000 updates
with auxiliary on versus off. Continue the corresponding main-only endpoint for
the same duration. All five continuations use the same LR schedule restart and
exact-state inheritance. Compare recovery from the same joint endpoint first;
compare against the continued solo model as a separate reference.

This creates at most 63 competition arms and 45 recovery arms, depending on the
attained precision levels. Together with qualification, the maximum is 144 runs
and 18 measured-training GPU-hours, excluding validation and startup. Concurrency
is at most four GPUs. Qualification gates and missing precision levels reduce the
count. Allow at most two infrastructure retry submissions and two extra allocated
GPU-hours; preserve failed attempts. Do not expand model size, output bins, tasks,
or training budgets automatically.

## Validation and execution

Validate every 100 updates, so the final selection window is updates 11,800,
11,900 and 12,000. Use the existing fixed 1,024-example validation set and independent 8,192-example
test set. Test is evaluated only at each stage's final endpoint; the 3,000-update
qualification comparison uses validation. Exact common-step comparisons are
primary. If a time cap prevents full updates, flag the arm as incomplete and do
not silently use it to qualify or compare final endpoints. Report seed medians
and ranges; these three-seed ranges are not confidence intervals.

`followup_suite.py` creates qualification, competition, and recovery manifests.
Competition generation writes the validation-only selection decision, including
an empty manifest if qualification fails. Parent checkpoint SHA256 values are
checked before loading. Output directories are immutable with respect to config;
preemption resumes the latest checkpoint. The checkpoint retains raw weights,
EMA weights, optimizer, RNG, and stage history. Tests cover exact resume with EMA,
EMA's lack of effect on raw training, fork state inheritance and hash rejection,
loss scaling, schedule endpoints, and validation-only recipe selection.
