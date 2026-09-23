<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Proposed next wave: allocate additional training between OM4 and observations

23 September 2026. Revised after reviewing the other thread's source training
artifacts. Supersedes the earlier hyperparameter-screening proposal in this file.
The user accepts rough matching by updates; detailed FLOP profiling is not a
prerequisite. **Proposal only: no training or evaluation jobs submitted.** Later waves require
separate approval. Keep deterministic D's architecture and the observational task.

## What the other thread actually trained

The observational pilot's source is primary D, SHA-256 beginning `22e629f4`, not
joint D (`a0c42c23`). It combines a newly trained wide history initializer with
previously pretrained wave-1 autoregressive dynamics. These are separate training
histories, not a single short pretraining run.

| Component | Observed training history | Interpretation for the next wave |
|---|---|---|
| D initializer: 121.7M parameters | AdamW, constant LR 3e-4, global batch 8; 95,919 optimizer updates, epoch counter 274, 3.669 training hours on two RTX PRO 6000 GPUs. Best reconstruction validation at update 43,439; stopped after six non-improving validations. | Reconstruction had plateaued under this recipe. Do not spend the main OM4 allocation repeating it unchanged. |
| Inherited evolution: 31.6M parameters | Wave-1 AR pretraining, constant LR 3e-4 and one/three/six-step curriculum. Selected update 1,260,598; production ran about 31h25m. Seven subsequent validation checks failed to improve after more than 12 hours at six-step horizon. | Substantial dynamics pretraining already occurred. This is a marginal-compute experiment from an established model, not early pretraining versus scratch. |
| Joint D: same initializer and evolution | Constant LR 1e-4, six-step full-variable forecast loss + 0.1 reconstruction loss; 20,268 updates in three hours on two RTX GPUs. Best forecast validation at update 18,041; the run ended at its time cap with one non-improving check. | Joint adaptation was not shown to have converged. This is the most relevant OM4 continuation objective. |
| Observational fine-tuning | Constant core LR 1e-5, adapter LR 1e-4 in joint training; 1,000 joint updates. Validation improved at every saved joint check through the last one. | Additional observational training is also plausible; allocation needs a direct test. |

D reconstruction-validation normalized T/S RMSE was 0.05519 at the selected
checkpoint and 0.05528 at the final checkpoint. The final training-probe RMSE was
about 0.04048, so continuing the same reconstruction recipe was not obviously
limited by too few optimizer updates. The source learning rates were constant:
there was no exhausted cosine/decay schedule to restart.

On the other thread's OM4 held-out cohort, joint D improved day-30 T/S RMSE from
0.06918 to 0.06653 (3.8%), while initialization RMSE slightly worsened. This is
existing model-world evidence for joint training, not evidence that joint D
transfers better to observations. The observational pilot deliberately used
pre-joint D, so it does not answer that question.

Primary evidence:

- [Other thread's wave-3 report](../surface-wave3-results/index.md),
  [training protocol](../surface-wave3-plan.md), and
  [learning curves](../surface-wave3-results/figures/optimization/learning-curves.csv).
- [D training completion](../surface-wave3-results/artifacts/training/primary/D/TRAIN_COMPLETE.json),
  [joint D completion](../surface-wave3-results/artifacts/training/joint/D/TRAIN_COMPLETE.json),
  and [checkpoint lineage](../surface-wave3-results/artifacts/lineage-audit.json).
- [Wave-1 report](../surface-wave1-results/index.md) and
  [AR pretraining early-stop record](../surface-wave1-results/ar-pretrain-early-stop.json).
- [Observation pilot report](snapshot-2026-09-23.md).

These are checked-in historical execution records, not a fresh cluster inventory.
The source code increments the wave-3 step counter at optimizer steps after
accumulation; nevertheless, those counts are not FLOP-equivalent to monthly
observational updates with multiple reconstruction/forecast forwards.

## Question and fixed starting point

**From the exact same primary D checkpoint, where should the next unit of training
compute go: joint OM4 training or observational adaptation?** Hold architecture,
source checkpoint, observation splits, data operators and evaluation fixed. No
capacity, history-length, resolution, spectral-loss or forcing-adapter expansion
search in this first allocation wave. All observation stages restart from their
assigned OM4 endpoint, not the already observation-adapted pilot checkpoint.

Interpret results conditional on the already spent OM4 pretraining. A separate
end-to-end comparison would charge the original dynamics and initializer training,
including curriculum-dependent costs, to transfer. Neither the original million
updates nor the entire multi-arm 122-GPU-hour wave is the correct direct training
cost of D without reconstructing its specific lineage.

## Qualification and small learning-rate calibration

Before production, verify source checkpoint hashes, data availability and space;
strict-load both original D and archived joint D for reference. Reuse existing
normalization stores. The transfer allocation arms all retain the same source
normalization, adapter design and observation-only loss scaling. Do not introduce
a new split or normalization change inside the allocation comparison.

Use **optimizer updates at effective batch eight** as the primary budget unit.
Both joint paths roll out roughly a month: six five-day OM4 steps versus six/seven
observational steps. Observation training also performs auxiliary monthly
reconstruction, so equal updates are explicitly not equal FLOPs. Count actual
examples, forecast steps and reconstruction forwards, and report GPU-hours and
wall time alongside score. A short timing check of roughly 20 steady-state updates
per path is sufficient for resource planning; no full FLOP profiler is required.
If one path costs dramatically more, show that directly rather than calling this
compute-equivalent. A later hardware-time-matched sensitivity check can address it.

Run two short, equal-update learning-rate trials per domain (200 joint updates each):

- OM4 joint core LR: **3e-5 versus 1e-4**.
- Observation joint core LR: **1e-5 versus 3e-5**, adapter LR fixed at **1e-4**.
- AdamW weight decay 0.01, gradient clipping 1, effective batch 8; preserve phase-
  specific BatchNorm policies and log them. Adapter starts with zero outputs.
- Select OM4 rate on its forecast validation score; observation rate on the frozen
  integrated-plus-spectral observation score. Qualification updates are discarded.
  Observation calibration uses a common fresh reconstruction warm-up for both rates.

Use a short warm-up (5% of each phase's update budget), then a constant selected LR
for the first wave. This deliberately avoids conflating allocation with different
positions in a cosine schedule and permits exact-prefix continuation. Fix optimizer
reset policy: fresh optimizer from D for OM4 continuation, then fresh observation
optimizer after switching domains. Retain optimizer state for budget extensions
within a phase. A second LR check at the two allocation extremes is a follow-up
if the ranking is small or unstable, not an unbounded tuning search.

## First production wave

Every model gets **1,000 observation reconstruction updates plus a pool of
4,000 joint updates**, at effective batch eight. Allocate that joint-update pool
between OM4 and observations. Run OM4 first, then the common observation
reconstruction stage, then observation joint training. The common reconstruction
stage is performed separately for each arm, because its starting weights differ.

| Literal result name | Initial weights | OM4 joint updates | Observation reconstruction updates | Observation joint updates |
|---|---|---:|---:|---:|
| D → observations | Original D | 0 | 1,000 | 4,000 |
| D → 25% OM4 → observations | Original D | 1,000 | 1,000 | 3,000 |
| D → 50% OM4 → observations | Original D | 2,000 | 1,000 | 2,000 |
| D → 75% OM4 → observations | Original D | 3,000 | 1,000 | 1,000 |
| Observation-only random | Random initializer/evolution | 0 | 1,000 | 4,000 |

The percentages apply to the 4,000 joint updates, not all 5,000 updates or FLOPs.
The most OM4-heavy arm still gets the pilot's 1,000 observation joint updates.
This is a deliberately rough **update-matched** comparison, not a FLOP-matched
one. The different supervision, six/seven-step windows and auxiliary reconstruction
cost must accompany interpretation. A shared OM4 prefix can supply snapshots at
1,000/2,000/3,000 updates if its schedule and seed are identical; report both each
model's training pathway and the smaller actual campaign cost when reusing it.

Observation-only random retains observation-derived normalization and learns
BatchNorm statistics as in the pilot. It has no learned OM4 weights or OM4
normalization. Its core LR is separately calibrated at **3e-5 versus 1e-4** using
the same 200-update trial budget. It is an equal-additional-update practical baseline;
it is not equal-total-lifetime-compute to pretrained D. Retained source grid and
architecture are common structural priors and disclosed as before.

Use one paired observational sampling seed for this screen. Match adapter seed
and the prefix of the observation sample sequence across transfer arms. Keep the
original observational forecast loss and auxiliary reconstruction weight 0.1.
During OM4 training use native OM4 forcings and full-state labels, with six-step
joint forecast loss + 0.1 reconstruction, as in the successful joint-D experiment;
the ERA5 adapter is trained only on observations. This compares two data/objective
packages, not data provenance under an identical loss.

Take the terminal checkpoint of each finite, fixed-update OM4 phase into adaptation;
do not add per-arm OM4 best-checkpoint searches. Monitor its validation as a
diagnostic. Select final observational checkpoints using the agreed observation
score, at the same prespecified five fractional observation-joint checkpoints
(including the phase start). Also report terminal-checkpoint scores so early
selection cannot conceal wasted allocated compute. Keep all costs, including
failed/divergent trials; do not silently restart only unfavorable arms.

## Reporting, confirmation and scope

Primary output: observation score versus fraction of joint updates allocated to
OM4, with total updates, examples, forecast/reconstruction work counts and
wall/GPU-hours. Also plot score against actual training GPU-hours. Report all integrated components,
spectral error by region/field/lead, anomaly correlation/amplitude, and each
model's own inferred-state and interior-anomaly persistence. Persistence gains
without forecast-over-persistence gains support better initialization, not better
dynamics. Preserve the complete-stencil velocity/EKE diagnostic because the
existing ADT-boundary issue can influence the composite.

The main decision uses validation only. Freeze the pilot score and masks; do not
add a spectral training loss or change the velocity metric during this comparison.
The nine-month validation record is limited, and 2015–2022 results have already
been inspected. Report those as historical benchmarks, not new blind evidence.
Before a confirmatory claim, audit and reserve a previously unused common product
interval (possibly 2023 onward, not yet verified). A broader blocked validation
scheme is a separate protocol change, requiring symmetric reruns; do not quietly
mix it into the existing-score comparison. Audit source OM4 normalization dates
and forcing/year overlap when making temporal-generalization claims.

After reviewing the screen, repeat the observation-heavy baseline and the best
nonzero OM4 allocation with three fresh paired seeds at the same 4,000-joint-update budget. Retain random
observation-only repetitions if making a pretraining-versus-no-pretraining claim.
If the screen cannot separate allocations, report that and choose a prespecified
contrast (0% versus 50%) rather than declaring the small numerical winner optimal.
A separately approved second budget of 8,000 joint updates tests whether the preferred allocation
changes with scale. Show paired seed differences and temporal-block uncertainty;
do not treat grid cells or monthly starts as independent replicates. Claims remain
conditional on the one shared source checkpoint unless source seeds are replicated.

Archived joint D is a useful cheap historical transfer check, but not a free
member of the update-matched curve: it already consumed 20,268 joint OM4
updates (selected at 18,041) and three training hours. If adapted, plot it
separately with that historical work explicitly charged. Do not replace one production arm with it silently.

This wave answers marginal allocation for one-month observation prediction.
It does not establish eight-year stability, a universal optimal pretraining ratio,
or the cost-effectiveness of OM4 pretraining from random initialization.

## Resource envelope and launch gates

Propose **a first-wave ceiling of 100 allocated GPU-hours**, including calibration,
short timing checks, five screening runs, validation and recovery; this is a ceiling for
review, not a throughput-backed promise. Benchmark first. If the 5,000-update-per-model
matrix does not fit, report measured costs and revise the entire budget before
production rather than shortening selected arms. Later seed/budget waves are
separate decisions. No runs are launched by this proposal.

On beta use at most one host/four GPUs. Start from the already working OM4 two-GPU
recipe (two concurrent jobs) unless one-GPU memory/throughput qualification supports
four independent trials. Keep observational jobs single-GPU unless distribution is
explicitly qualified; do not assume their harness supports DDP. Torch is an allowed
alternative after live storage/capacity checks. Check space for OM4, the compact
observational package, optimizer checkpoints and exports; no full-resolution
observation transfer is required. No cluster/quota check was performed for this
proposal because nothing is being submitted.

Implementation needed: explicit per-phase update caps and work/time logging; configurable
learning rates and phase optimizer policies; common checkpoint loading between OM4
and observation harnesses; phase-normalized validation cadence; reproducible
allocation/seed manifests and reports showing both update counts and cost. Verify loading/equivalence,
optimizer resume, finite fitting, exact masks and budget accounting before production.
The current architecture is adequate; rebuilding D or repeating its original
pretraining is not a prerequisite for this wave.
