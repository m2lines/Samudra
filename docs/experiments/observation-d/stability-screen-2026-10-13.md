<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation training stability screen

Stages 1–2 of the [longer-training plan](stabilization-long-training-plan-2026-10-13.md)
are authorized. The longer-run matrix and any additional batching/modeling tests
remain behind the post-screen review.

The first stage uses **U-global** and **U-multitask-early** (earlier 1° OM4;
no fine task or extra latent channels). Retained seed-1729 checkpoints contain
2,667 total updates: 1,667 OM4 and 1,000 observation updates. After a new runtime,
fit, I/O and deterministic-replay qualification, copy their model, optimizer,
RNG and counters exactly, then replay the unchanged schedule once to a shared
3,000-update fork per model. All treatment and diagnostic copies descend from
these forks; diagnostic weights are never promoted.

Stage 1 evaluates each fixed fork five times (three in one process and two
fresh-process reloads), plus two deterministic-reference evaluations. It also
runs two independent 100-update diagnostic continuations per model, recording
fixed training-example losses before/after each update, sample/mask seeds,
gradient norms, actual parameter-update norms, and observation loss components.
Full fixed-nine-origin RMSE validation runs every ten updates; shared-parameter
gradient cosines use four fixed training examples per source at the start/end.
The recent-only model's early-data gradient is a diagnostic probe, not training.

Stage 2 will use the same shared forks for the eight planned 1,000-update
continuations: constant 1e-4 (two executions), constant 3e-5, and linear
1e-4 → 1e-6, for each model. Raw and EMA results will be compared with the frozen
RMSE-only reference. Its EMA/resume implementation must be tested and qualified
before submitting those continuations. No Stage-2 or longer-run jobs are claimed
as submitted in this initial implementation record.

## Execution

Torch preflight confirmed authenticated SSH, registry credentials and online
W&B credentials. Dry-run Slurm requests estimated H100 preemptible capacity
sooner than RTX; these are advisory estimates. Matched jobs use one GPU of the
same family and the same pinned image/runtime, with 16 CPUs and 192 GiB RAM.
Preemption warnings save resumable state; diagnostic copies checkpoint every
ten updates. Interrupted qualification attempts are preserved and restarted
fresh, so a partial probe is never accepted as a passed qualification.

Scratch was 4.78/5.00 TB used at preflight. Existing datasets and the cached
container are reused in place; this stage adds checkpoints and diagnostic
artifacts, with a planning allowance below 20 GB. No source data is deleted.

Source checkpoint lineage comes from the audited cooldown source manifest;
U-global's historical checkpoint was originally produced on Beta. This does not
make comparisons against its old uninterrupted trajectory a pure LR experiment.
The new unchanged-control repetitions use the same new runtime as the treatments.

Implementation: [diagnostic runner](../../../scripts/run_observation_stability.py),
[Slurm harness](../../../scripts/slurm_observation_stability.sbatch), and
[isolation/migration tests](../../../tests/test_observation_stability.py).

## Submitted first stage

Pinned producer: `72c06e2e92d007315542c5c255bc3ad1937b3927`.
Torch root: `/scratch/jr7309/runs/2026-10-13-observation-stability`.
The source-only archive was verified after transfer. The source audit passed:
both checkpoints have the expected 2,667-update optimizer state, observation
metadata match their original manifests, and the container/runtime hashes match.

| Job | Work | Dependency |
|---|---|---|
| 19725599 | CPU source/checkpoint/runtime audit | None; audit marker passed |
| 19725600, array 0–1 | Fresh U-global / earlier-OM4 fit and replay qualification | Successful source audit |
| 19725601, array 0–1 | Exact-state migration and common 3,000-update forks | Both qualifications successful |
| 19725618 | CPU-only qualification completion callback | Qualification termination, including failure |

Slurm confirmed H100 routing, the preemption comment and requeue enabled.
These submissions do not establish that qualification or training has succeeded.
The separate repeatability/trace jobs follow inspection of the common forks.
A quiet callback monitor uses a two-hour queue/audit backstop; notifications
report meaningful completion or blockers. Local validation: 45 targeted tests
and all repository pre-commit checks passed.
