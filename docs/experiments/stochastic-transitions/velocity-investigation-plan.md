<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Velocity learning investigation

This investigation follows the [matched 2k loss experiments](loss-ablation-partial.md).
Its question is why the stochastic initializer learns velocity poorly under fair
CRPS, and whether a small change can improve reconstruction while retaining
useful ensemble variation. The user authorized diagnostic work and successive
small experimental waves on 2026-10-29. No new long production run is planned.

## Wave 1: distinguish the explanations

Evaluate the seven existing model/loss combinations on several fixed validation
dates, using identical surface histories, ERA5 inputs, normalization, masks and
sampling seeds. Save the input and checkpoint fingerprints. Preserve the earlier
single-case result, and report any date excluded because exact OM4 history is
unavailable. Do not silently resample a mismatched calendar.

Compare velocity error by depth, reference speed and region; ensemble mean and
individual members; the absolute-error and spread terms of fair CRPS; kinetic
energy, spectra and spatial correlation. Include zero velocity and a genuinely
training-derived climatology where available. A baseline computed from validation
targets is not a training-derived climatology. Use common map limits and retain
unclipped errors. Examine saved intermediate checkpoints for the main loss
contrasts. Deterministic duplicated members do not supply calibration evidence.

Audit gradients on fixed training batches, separating OM4 initializer velocity,
other initialization fields, forecast terms, and observation terms. Report the
normalization and loss coefficients, output gradients and shared-parameter
gradients, and distinguish gradients before clipping from actual optimizer
updates. A negative gradient cosine alone does not prove harmful interference.

A separate CPU toy checks the finite-ensemble loss derivatives and whether a
simple conditional regression problem reproduces the loss-dependent learning
pattern. It supplies a mechanism test, not evidence of ocean forecast quality.

The initial diagnostic allocations are bounded to approximately four GPU-hours
for multi-date evaluation and three for the gradient audit. Actual allocated
time, including failed startup overhead, must be recorded.

## Wave 2: test an intervention supported by the evidence

Choose the intervention after inspecting wave 1. Candidate mechanisms include
insufficient emphasis on large velocity residuals, weak relative velocity
supervision, finite-member gradient behavior, and competing task gradients.
Do not launch all candidate interventions without evidence.

Use a matched continuation control, identical starting model and optimizer
states, data order, update count and learning-rate schedule. Keep an explicit
baseline when a reset or learning-rate change is necessary. Start with short
continuations and intermediate evaluations; extend only if the comparison is
informative. Record whether any treatment changes loss scale and clipping.

Judge success on mean and member error, pointwise probabilistic scores,
calibration, and spatial structure across dates. A near-deterministic ensemble
with lower mean RMSE is not a demonstrated resolution of probabilistic learning.
Check the unchanged observation forecast blend for collateral degradation.
Validation dates used to choose an intervention cannot become an independent
test set. Report the pilot as exploratory unless a separate cohort is evaluated.

## Deliverables

Produce a report with learning curves, matched maps, depth/current-strength
breakdowns, CRPS terms, gradient evidence, and any controlled intervention.
Separate observed effects from causal hypotheses and unresolved questions.
Retain per-experiment machine-readable scores and source/checkpoint/job receipts.
Use durable callbacks and monitoring subagents for work that outlives a turn.
