<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Matched learning-rate cooldown

The follow-up tests whether the [completed earlier-OM4 comparison](early-fine-wave-2026-10-07.md)
changes when its final training phase uses a decreasing learning rate. It includes
U-global and all four new models, using the same seed and total exposure.

## Protocol

- Fork each retained `joint-01000.pt`: **2,667 total updates**, including 1,000
  observation and 1,667 OM4 updates. Filenames count observation updates.
- Continue at `1e-4` through update 3,000. There is no retained checkpoint exactly
  at 3,000, so these 333 updates are replayed using the original task/sample schedule.
- For updates 3,001–4,000, linearly decrease **both task learning rates** from
  `1e-4` to `1e-6`. At update 3,500 the rate is `5.05e-5`.
- Preserve optimizer moments, CPU/CUDA/NumPy RNG state, counters and the existing
  task/sample sequence. The original checkpoints and reports stay intact.
- End at the same **4,000 updates: 2,000 OM4 and 2,000 observations**. The last
  1,000 updates contain 782 observation and 218 OM4 updates; this is not an
  observation-only finetune.
- Select on the same frozen integrated-plus-spectral observation validation
  composite. Reevaluate the fork checkpoint and select along its continuation;
  never import a later selected checkpoint from the original constant-rate run.
- Evaluate the selected models and their own initialized persistence on the same
  96 monthly origins in 2015–2022, with day-5/15/30 surface and monthly OHC metrics.

| Report model name | Original model/checkpoint | OM4 training | Recurrent extra channels |
|---|---|---|---:|
| U-global-cooldown | U-global at update 2,667 | 2,000 recent 1° updates | 0 |
| U-multitask-early-cooldown | U-multitask-early at update 2,667 | 1,000 recent + 1,000 earlier 1° | 0 |
| U-multitask-early-latent-cooldown | U-multitask-early-latent at update 2,667 | 1,000 recent + 1,000 earlier 1° | 10 |
| U-multitask-early-fine-cooldown | U-multitask-early-fine at update 2,667 | 1,000 recent 1° + 1,000 earlier ¼° | 0 |
| U-multitask-early-fine-latent-cooldown | U-multitask-early-fine-latent at update 2,667 | 1,000 recent 1° + 1,000 earlier ¼° | 10 |

Architecture, input fields, losses, normalization and parameter counts are
unchanged; see the original report's methods. Each directory retains its original
model name inside the separate cooldown root; result tables add `-cooldown`.
Compare each cooldown against its own constant-rate parent before interpreting
differences between data/architecture arms. One seed cannot establish significance.

U-global moves from Beta to Torch at the fork. Its model/optimizer state and
observation data fingerprints are checked exactly, but cross-hardware numerical
variation remains a caveat when attributing its difference entirely to cooldown.
The four other arms stay on the same Torch runtime. Production CUDA numerical
flags are unchanged; deterministic flags are confined to qualification replay.

## Execution

Root: `torch:/scratch/jr7309/runs/2026-10-08-early-fine-cooldown`.
One RTX, 16 CPUs and 192 GiB per arm; up to five arms, subject to the scheduler.
The same data caches and observation preprocessing are reused. Qualification uses
fresh disposable weights. A CPU migration stage then checks all five qualifications
and exactly audits all non-manifest checkpoint state before production may start.
Fitting/probe weights never enter production. Reports will include every attempt's
GPU time separately from the already completed original wave's 25.325 GPU-hours.

Implementation: [shared trainer](../../../src/samudra/experiments/observation_joint.py),
[runner](../../../scripts/run_early_fine_cooldown.py), and
[checkpoint migration audit](../../../scripts/prepare_early_fine_cooldown.py).
The new recent-only route uses the original OM4 objective for every U-global OM4
update while sharing the tested CPU observation cache. **48 targeted tests pass**,
including an identical prefix before cooldown, exact optimizer resume across a
task boundary, and recent-only control routing.

Status: implementation tested; checkpoint transfer and submission in progress.
