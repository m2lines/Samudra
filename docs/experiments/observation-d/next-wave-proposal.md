<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Proposed next experiments: does OM4 pretraining help?

23 September 2026. Proposal for review; no runs submitted. Each later wave needs
its own compute approval. Keep the deterministic D architecture, one-degree grid,
five-day step and observational training sources fixed.

## Questions and current evidence

Separate three claims: (1) better forecasts at a fixed observational training
budget, (2) faster learning or fewer observational examples to reach a fixed skill,
and (3) better skill after a substantially larger training budget. A short run
cannot establish the third claim or convergence.

The pilot is encouraging, but transfer and scratch used different learning rates,
normalization, BatchNorm handling and phase lengths. There was one seed. Transfer
validation score improved at every saved joint checkpoint, from 0.7424 at update
100 to 0.6141 at 1000. Scratch improved from 1.6767 to 0.9128 at update 900, then
0.9196 at 1000. Neither curve establishes a plateau. These numbers come from the
[archived validation events](artifacts/2026-09-23/evidence.json.gz).

Two comparisons are needed. Equal-budget independent tuning asks which approach
produces the better usable model. A separate matched-recipe initialization ablation
asks which part of pretraining supplies the gain. Do not describe the first as an
isolated effect of learned weights.

## Fixed evaluation protocol

Continue selecting and reporting the integrated-plus-spatial-spectral observation
score, with fixed validation-climatology denominators. RMSE remains a component
and training diagnostic, never a replacement selection criterion. Freeze score
weights, regions, leads and masks before this wave. Preserve the pilot score for
comparability; report complete-ADT-stencil velocity/EKE diagnostics alongside it.
Do not change the primary metric after seeing which model wins.

The current nine-month validation period is weak for tuning. Before the search,
reserve two complete seasonal cycles from the previously available development
period and retrain both arms on the same reduced training partition. Choose exact
month boundaries from actual product coverage, with a purge at least as long as
the 95-day input support plus 35-day forecast support, enlarged for documented
provider temporal smoothing. Recompute training normalization, climatology and
validation reference denominators. New scores must be labeled as a new protocol;
old selected observational checkpoints cannot initialize these experiments if they
saw the new validation months. Original OM4 D remains the transfer starting point.

The already inspected 2015–2022 cohort remains a historical comparison, not a
pristine final test. Audit a common, unused 2023-onward OISST/DUACS/IAP/ERA5 interval
before promising fresh confirmation; availability and sufficient duration are not
yet verified. Lock that interval and inspect it only after recipes and checkpoints
are fixed. If unavailable, report the results as exploratory and obtain new data
before making a confirmatory claim. Also audit the dates used for OM4 dynamics
training, initializer training, normalization and checkpoint selection: common
forcing/calendar years can weaken a claim of temporal generalization even without
using observational targets.

## Wave A: improve both approaches with equal search budgets

Use the same 153.3M-parameter architecture, adapter, data order, effective batch of
eight, optimizer, loss scales, training examples and validation cadence. Pair the
search seed between arms. Start each trial afresh, including all adaptation phases;
do not reuse the observationally fine-tuned pilot weights.

| Hyperparameter | OM4-initialized candidates | Observation-only candidates | Purpose |
|---|---|---|---|
| Core peak learning rate | 3e-6, 1e-5, 3e-5 | 3e-5, 1e-4, 3e-4 | Avoid imposing the transfer learning rate on scratch |
| Added spectral loss coefficient | 0 or 0.05 | 0 or 0.05 | Test whether power loss can be reduced without degrading prediction |
| Core schedule | 200-update warm-up, then cosine decay to 10% of peak | Same | Replace the current constant learning rate |
| Adapter joint peak learning rate | 1e-4 | 1e-4 | Hold forcing adaptation policy fixed |
| Weight decay / clipping | 0.01 / norm 1 | Same | Retain known working settings |
| Reconstruction auxiliary weight | 0.1 | 0.1 | Retain current joint-stage balance initially |

This is six configurations per arm, twelve trials total. Keep the original forecast
loss weights (T/S/SST/ADT = 0.4/0.4/0.1/0.1) in this screen. The proposed spectral
term is a differentiable, dimensionless log-power error on training SST/ADT at the
same broad resolved scales, masks and regions as evaluation; average over fields,
regions and scored leads before multiplying by 0.05. No velocity/EKE spectral loss
in the first wave because the ADT-gradient support effect remains unresolved.
Use a training-derived positive power floor, fixed taper/mask treatment, and check
finite gradients and relative loss/gradient magnitudes before production. The
coefficient is a hypothesis, not a demonstrated optimum. Always retain the
zero-spectral-loss controls. Evaluate EKE in the unchanged selection score.

Give every trial 1000 reconstruction updates and 2000 joint updates. No separate
transfer-only warm-up in the matched schedule. Use the same freshly initialized
zero-output adapter. Keep the pilot's native normalization/BatchNorm policy for
this practical comparison and explicitly report that it compares training recipes:
source normalization/frozen running statistics for transfer, observation-only
normalization/updating statistics for scratch. Loss normalization is observation-
derived for both. Report all candidates, including divergence and non-improvement.

Validate every 250 joint updates. Do not terminate scratch just because it starts
worse; no score-based early stopping before the common screening budget. Stop on
nonfinite states or losses and count failed trials against the search budget. Use
the same compute cap per arm; record examples, updates and GPU-hours separately.
A spectral-loss trial may be more expensive, so equal updates alone are insufficient.

A scheduler must support these staged comparisons without an artificial learning-
rate reset at promotion. Use a predeclared 8000-update cosine horizon for screening
and extensions, and preserve optimizer/scheduler/RNG state when extending.

## Wave B: longer training and final repeated comparisons

Promote the best two configurations per arm on validation only and extend each to
8000 joint updates. Preserve equal aggregate search budgets. Retain best checkpoints
by the observation score, but also show learning curves at 1000, 2000, 4000 and 8000
updates and common GPU-hour budgets, including reconstruction cost. If scratch is
still improving, report the budget-limited result rather than declaring convergence.

Lock the best recipe for each arm, then run three new paired seeds per arm from
original OM4 weights or random weights, respectively (six full runs). Do not select
the luckiest seed. Report every seed and the mean/spread. Pair initial adapter weights
and sample order where possible. These repetitions measure observational training
variation conditional on one source D checkpoint, not variation across OM4
pretraining seeds. A general claim about the pretraining procedure eventually needs
independent source checkpoints too.

If Wave A/B exposes a specific bottleneck, a separately approved small follow-up can
vary initializer:evolution learning-rate ratio (0.3, 1, 3), reconstruction weight
(0, 0.1), surface/interior loss balance (0.1/0.1/0.4/0.4 versus 0.25 each), or BatchNorm
training-only recalibration versus online updates. Give both approaches the same
number of extra trials; do not grant transfer a larger tuning budget. Do not expand
all these into a large Cartesian search before the first screen informs it.

## Wave C: identify what transfers

Run a 2 × 2 initialization ablation with a common recipe and three paired seeds.

| Literal model name | Initializer | Evolution | Meaning |
|---|---|---|---|
| OM4 initializer + OM4 evolution | OM4 | OM4 | Full transfer |
| OM4 initializer + random evolution | OM4 | Random | Test transferred reconstruction without transferred dynamics |
| Random initializer + OM4 evolution | Random | OM4 | Test transferred dynamics without transferred reconstruction |
| Random initializer + random evolution | Random | Random | No learned OM4 weights |

For this controlled ablation, use common source-coordinate normalization for all
four cells, identical learning rates, phase lengths and trainable parameters, and
reset/recalibrate BatchNorm running statistics on training observations with the
same policy in all cells. All cores then train; all adapters start identically.
This deliberately separates initialization from recipe changes. The random/random
cell uses OM4 normalization information and MUST be labeled that way; it is not
the strict observation-only baseline. Keep the pure observation-only model from
Waves A/B in the report. The ablation estimates the weight-transfer effect under
this shared coordinate system. It does not eliminate every dependence on OM4.

Choose the common recipe by a prespecified symmetric validation rule (e.g. lowest
mean score across all four initialization cells under equal small tuning budgets),
not the best transfer-only recipe. Mixed initializations require finite-forward,
small-batch fitting and physical-interface checks; mismatched initializer/evolution
states can make their interaction scientifically meaningful but hard to optimize.
Report all four outcomes and their interaction, not an additive attribution assumed
in advance. Budget these as new runs; the independently tuned Wave B runs are not
substitutes for this controlled factorial.

For every trained model report its own inferred-state persistence, interior-anomaly
persistence and seasonal-climatology comparisons. Also compare forecast-minus-
matched-persistence skill per metric. If transfer only improves OHC persistence,
the defensible conclusion is improved state reconstruction, not improved dynamics.
If desired, a common supplied-state evolution diagnostic can probe dynamics more
directly, but label it separately from each model's end-to-end forecast.

## Evidence to report and optional follow-ups

Report integrated-plus-spectral score, all five integrated components, spectra by
field/region/lead, anomaly correlation and amplitude, deep-OHC anomalies, and maps
at fixed origins. Include signed log-power ratios so damping and excess power are
not hidden by unsigned dex errors. Winning the score alone does not establish
better phase skill or physical realism. Spectral improvement is useful only when
viewed alongside the integrated and anomaly diagnostics.

For final comparisons, compute paired differences on common origins. Resample
contiguous temporal blocks and recompute complete metrics, including EKE, rather
than treating grid cells or monthly starts as independent samples. Predeclare block
length/sensitivity using development data; report seed spread separately from
finite-test-period uncertainty. Three seeds and a short fresh test support limited,
conditional claims, not precise population confidence statements.

A later sample-efficiency study can use nested 25%, 50%, 100% training subsets with
seasonal coverage, chosen before scoring, and all preprocessing fit within each
subset. Use identical subsets across arms and show both fixed-example and fixed-
compute comparisons. Another later study should run 90/180/365-day forecasts and
then continuous eight-year trajectories for stability and temporal spectra. Those
need contiguous forcing and long-rollout evaluation changes; present one-month
results cannot establish those capabilities. Test long runs from fixed checkpoints,
not select again using the final test.

## Resources and implementation before launch

Plan at most four independent one-GPU trials concurrently. On beta this means one
host with four GPUs; first verify four independent trials fit host memory and I/O.
Avoid assuming the pilot harness already supports DDP. Torch remains an alternative
subject to live quota/capacity checks. Reuse the compact data design, stage only the
needed revised split/new coverage, retain selected/last checkpoints and scalar
histories, and export large prediction arrays only for finalists. Measure actual
checkpoint size before multiplying by trial count and check scratch quota.

The original runs cost about 4–5 allocated GPU-hours each for roughly 1600–1900
retained core-stage updates plus qualification/restarts. A rough Wave A envelope is
80–120 GPU-hours, not a reservation or measured new-run estimate. Benchmark validation
and spectral-loss overhead first and revise the budget before submission. Twelve
8000-update factorial runs and seed repetitions make the complete study several
hundred GPU-hours; approve later waves separately. Report downstream training and
search cost both excluding and, where available, including amortized OM4 pretraining
cost. Fast fine-tuning alone is not an end-to-end compute saving.

Required code changes: configurable initializer/evolution/adapter rates and schedule;
explicit initialization, normalization and BatchNorm switches; differentiable masked
spectral loss; fixed-budget promotion/resume and compute accounting; split/provenance
checks; paired-seed/factorial manifests and report generation. Validate unchanged
zero-spectral-loss behavior, checkpoint loading, scheduler resume, masks/gradients,
and no validation/test examples in preprocessing. These are planned changes, not
features claimed to be available today. No architecture or OM4 pretraining rerun is
required for the primary next wave.
