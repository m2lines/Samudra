<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Wave 2 execution status

September 20, 2026, resumed after 20:14 UTC. The [study plan](surface-wave2-plan.md)
is active again, with the original September 23 16:51:29 UTC target. Storage was
restored, the container and code overlay rebuilt, and qualification attempt 2
completed successfully. Production and the pre-registered rate control are now submitted.

## Qualification attempt 1

Code: `c4ab5e344ca4c3b73cbbbcd0e289d98fc88751ad`, with the existing Rust-loader SIF.
Remote root: `/scratch/jr7309/runs/2026-09-20-surface-wave2-qualification`.

| Arm | Slurm job | Outcome | Allocated seconds | GPUs |
| --- | --- | --- | --- | --- |
| A, initializer | 18079867 | Failed during training-cache preparation; detailed exception absent from captured streams | 308 | 4 |
| B, evolution | 18079868 | 30 training steps completed; failed during held-out cache preparation | 278 | 4 |
| C, joint | 18079869 | Cancelled after dependency failed; never started | 0 | 0 |

Actual allocation consumed: **0.651111 GPU-hours**. A and B did run simultaneously,
confirming eight-GPU execution was available after the other shared-account job ended.
B's frozen initializer verification passed and its best validation T/S MSE improved
from 0.004401835 to 0.004341907 over 30 steps. This is qualification only, not a
production result. A reached the same starting validation score. Neither arm has a
successful final evaluation marker.

## Storage blocker and recovery

A subsequent `mkdir` under `/scratch/jr7309` failed with **Disk quota exceeded**.
The login node's `/tmp` was also full (6 GiB used of 6 GiB), preventing a new code-layer
fetch. Aggregate scratch filesystem free space was about 519 TiB; that does not
establish free space within the user's quota. `quota -s` could not query the home
quota and did not provide usable scratch quota details.

The experiment cache directory occupies only 6 KiB, code layers 104 MiB, wave-1
outputs 4.7 GiB and qualification outputs 951 MiB. There is no large disposable task
cache to clear. Existing source and qualification checkpoints were retained. The
user was asked to free/add at least 10 GiB and received the authorized Slack blocker
notification successfully. Cancellation was issued for A/C as a safeguard; accounting
subsequently showed both had already terminated. Do not describe the qualification
failure's exact cause as proven: stream logs stopped before the exception, and the
storage failure is independently confirmed.

Commit `b46eb446f153a967869bbfc7f9e93783de986579` adds persistent rank-local Python
exception and fatal-signal diagnostics. It is pushed but its overlay is **not built**:
first attempt failed because login `/tmp` was full; scratch temporary-directory
creation then failed because of quota. The original overlay is intact.

After storage is restored, build the new overlay and requalify all three gradient
paths under the same producer commit before production. Use a new qualification
attempt directory and retain/account for attempt 1. Validate original checkpoint
hashes and qualification checkpoint integrity before using any saved state. Once
qualification passes, launch the four approved production arms using the launcher.
The production gate requires matching code, four ranks and 30 qualification steps.
Continue hourly monitoring and keep all attempts in final compute accounting.

The final recheck at 19:13:58 UTC still returned `OSError(122, Disk quota exceeded)`
for a one-MiB scratch write, and the user queue was empty. Automatic continuation
is stopped until the user restores storage and resumes the goal. No additional
Slack notification was sent because the same blocker was already reported.

The paired-comparison analysis and its four passing tests were added in commit
`ac923a4f`. They require complete A/B/C/D production outputs, check common inputs
and fixed references, and compute paired calendar-year uncertainty intervals.
There are still no completed production results or scientific answers from wave 2.

## Storage recovery and qualification attempt 2

After cache cleanup, `myquota` reported 4.52 TB used of 5 TB (90.31%). A full 10 GiB
write/fsync succeeded **after** restoring the 12,176,011,264-byte SIF and updated
code overlay; all probe files were removed. CPU-only restoration job **18094385**
completed in 29 seconds. Download/unpack temporary paths were configured on its
node-local storage; the login node's full `/tmp` was avoided. The original AR
pretraining, selected wave-1 joint, and shared initializer SHA-256 hashes still match
the checked-in wave-1 records.

Qualification producer: `b46eb446f153a967869bbfc7f9e93783de986579`.
Remote root: `/scratch/jr7309/runs/2026-09-20-surface-wave2-qualification-v2`.
Jobs: A **18096965**, B **18096966**, C **18096967** (after A). All use fresh
qualification outputs and the original shared inputs; attempt 1 is retained.
A matched 1e-4 joint control E is now pre-registered to isolate learning rate from
the changed checkpoint-selection rule. It will follow production, within the
user's authorization for targeted follow-ups.

## Qualification passed; production submitted

All attempt-2 jobs completed: A in 222 seconds, B in 517 seconds, C in 217 seconds.
Each completed 30 optimizer steps, eight-origin evaluation, 8,316 unique finite
channel rows and 5,184 paired origin/variable rows. Starting model fingerprints
matched; fixed pretraining, wave-1 joint and climatology references matched within
numeric tolerance. Runtime frozen-state verification passed for A/B. No rank-local
error files contained an exception. Attempt 2 used **1.062222 GPU-hours**; cumulative
qualification attempts used **1.713333 GPU-hours**.

| Arm | Job | Dependency | Max GPUs | Allocation limit |
| --- | --- | --- | --- | --- |
| A: initializer | 18097624 | none | 4 | 4 hours |
| B: evolution | 18097625 | none | 4 | 4 hours |
| C: joint | 18097626 | A succeeds | 4 | 4 hours |
| D: joint, alternate data order | 18097627 | B succeeds | 4 | 4 hours |
| E: matched 1e-4 joint control | 18097633 | C and D succeed | 4 | 4 hours |

A-D outputs: `/scratch/jr7309/runs/2026-09-20-surface-wave2/{A,B,C,D}`.
E outputs: `/scratch/jr7309/runs/2026-09-20-surface-wave2-rate-control/E`.
All use producer `b46eb446f153a967869bbfc7f9e93783de986579` and the same qualified
initial checkpoints. The host submission/analysis update is commit `d5ce271f`.
The initial four-arm allocation limit is 64 GPU-hours; E adds at most 16.
These are allocation limits, not completed compute or scientific results.
