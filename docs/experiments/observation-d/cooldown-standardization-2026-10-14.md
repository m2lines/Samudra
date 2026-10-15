<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Standardizing cooldown from existing 4k runs

Compare data mixture, loss and EMA initialization using the completed S0
checkpoints from the [paired stability screen](stability-screen-2026-10-13.md).
The user approved eight 1,000-update continuations: four branches each for
**U-global** (recent 1° OM4 + observations) and **U-multitask-early** (the same
model, splitting OM4 exposure between recent and earlier 1° data). No architecture,
latent-channel or fine-resolution changes are introduced.

| Branch | Cooldown data | Objective | EMA at start |
|---|---|---|---|
| A | Seven observation updates then one OM4 update, repeated | Existing forecast + reconstruction/completion losses | Reset to raw weights |
| B | Observations only | Existing losses | Reset to raw weights |
| C | Observations only | Forecast loss only | Reset to raw weights |
| D | Observations only | Existing losses | Retain S0 EMA |

All branches start from identical raw model/Adam/RNG state within each family at
4,000 updates, preserve the per-task sample streams, and decay LR linearly from
1e-4 to 1e-6 over updates 4,001–5,000. Accumulation remains eight examples and EMA
half-life remains 100 optimizer updates. C preserves B's input corruption and
mask seeds; it removes loss terms only. D retains the existing 1,000-update EMA
history; all branches save EMA atomically with model/optimizer/RNG state.

A receives 875 observation and 125 OM4 updates. In the early-data family these
125 OM4 updates comprise 63 recent and 62 early examples at the optimizer-update
level. B/C/D receive 1,000 observation updates. Their first 875 observation
updates use the same sample/mask sequence as A. Different exposure is part of
the mixture treatment, not an accidental mismatch.

Compare raw and EMA RMSE validation at 4,000 and every 100 subsequent updates.
Primary screening summary is the mean over 4,600/4,700/4,800/4,900/5,000, weighted
equally across the two families; report endpoints and all components alongside
it. A vs B isolates data mixture, B vs C isolates auxiliary loss removal, and
B vs D isolates EMA history. The unchanged S0 checkpoint is a zero-extra-training
reference, not a matched-compute control for cooldown versus continued constant LR.

Before choosing a standard, add annual rollout and initializer temperature/OHC
diagnostics. The short validation cohort is November 2013–July 2014; observation
training ends July 2013. Audit an annual rollout beginning **2013-11-01** through
November 2014, including OM4 training-date exclusion, target coverage and forcing,
before using it as annual validation. This is one validation case, not a robust
annual test or the previously examined 2015/2018/2021 cohort. If its full payload
cannot be verified, report the blocker rather than substitute a test cohort into
selection. No final cooldown policy is chosen from short-lead scores alone.

## Qualification and execution

New producer source audit checks both completed S0 model/optimizer/EMA checkpoints.
Fresh fitting, I/O and deterministic restart checks gate branch-specific
qualification. Each branch qualification replays 16 updates twice under the
deterministic reference, covering both recent and early OM4 updates in A, and
requires exact model/optimizer/RNG/EMA equality and raw restoration after EMA
validation. Qualification copies are never promoted. Production uses normal
numerical settings and the original immutable S0 source checkpoints.

Implementation: [runner](../../../scripts/run_observation_cooldown_comparison.py),
[schedule, EMA migration and loss-isolation tests](../../../tests/test_observation_cooldown_comparison.py).
The shared trainer now exposes a schedule factory; existing runs retain their
original TaskSchedule. Local validation: 54 targeted tests passed.

The prior stability campaign consumed 10.5378 GPU-hours; this comparison is
accounted separately, including qualification and retries. Existing unrelated
forecast-adaptation jobs are left unchanged. No hyperparameter-search campaign
or longer main-training runs are launched by this approval.

Pinned producer: `5b49d2133eb73a83b0a6f47e06146d322ec332c6`.
Transferred archive SHA-256:
`8d3f123b5c22f9fdc53133379c35b6a2aaa6e8a1b4f32469f2eb1151e250699b`.
Scratch usage before staging was 4.74/5.00 TB; datasets and image are reused.
Root: `/scratch/jr7309/runs/2026-10-14-observation-cooldown-standardization`.

Submitted CPU audit **19850076**, base qualification **19850077**, and eight
branch qualifications **19850078**, with successful-completion dependencies.
Initial state was queued. Callback **19850080** and a quiet 30-minute startup
backstop cover the qualification chain. The eight production continuations
require verified qualification outputs; they are not yet submitted.

### Replay qualification recovery

The source audit and both base qualifications passed. All eight branch replay
checks failed before production. A regression with populated Adam state reproduced
the cause: `optimizer.load_state_dict` can retain references to CPU optimizer
state tensors, allowing replay one to mutate the in-memory baseline for replay
two. The restore helper now clones the saved state before loading it. The test
also verifies that the original snapshot stays unchanged after each replay.
All 54 targeted tests pass, including the previously failing populated-optimizer
case. The acceptance requirement remains bitwise equality; no tolerance was
relaxed. Production and original source checkpoints were not modified.

Failed branch outputs and the passing original checks remain at the original
root. Their combined allocated cost is **1.5364 GPU-hours**. Recovery uses a new
producer and a separate `recovery-snapshot-copy` directory with fresh qualification.
