<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Wave 2 execution status

**Completed September 21, 2026.** The [final report](surface-wave2-results/index.md)
answers the four primary questions and the matched learning-rate follow-up. All six
production evaluations and the CPU checkpoint audit passed. All 14 recorded jobs
are terminal; this wave has no remaining GPU or monitor jobs.

| Arm | Slurm job | Treatment | Result |
| --- | --- | --- | --- |
| A | 18097624 | Initializer only, 1e-5 | Complete |
| B | 18097625 | Evolution only, 1e-5 | Complete |
| C | 18097626 | Joint, 1e-5, data-order seed 1729 | Complete |
| D | 18097627 | Joint, 1e-5, data-order seed 1730 | Complete |
| E | 18097633 | Joint, 1e-4, data-order seed 1729 | Complete |
| F | 18160437 | Joint, 1e-4, data-order seed 1730 | Complete |

Global day-30 combined T/S normalized RMSE is 0.075645 for the untuned pair,
0.072176 for the selected wave-1 joint model, and 0.073620 / 0.074012 / 0.071116 /
0.071049 / 0.070841 / 0.070957 for A-F. Joint adaptation improves about 6% over
the untuned pair at both tested rates. The lower rate is not necessary to retain
the gain; differences between rates are small and vary by region, lead and order.
These are OM4 short hindcasts, not observational forecasts.

Total allocation was **86.463333 GPU-hours**, including 1.713333 for both
qualification attempts; peak concurrent allocation was **eight GPUs**. CPU
checkpoint audit **18160463** completed at September 21 08:29:35 UTC, before the
September 23 16:51:29 UTC deadline. No next wave is submitted.

## Recovery and provenance

Initial qualification A/B (18079867/18079868) failed and dependent C (18079869)
was cancelled without starting. A scratch write independently confirmed quota
exhaustion, although missing exception output prevents assigning the precise job
failure cause. After user cache cleanup, the SIF and producer overlay were restored
with CPU job 18094385. A 10-GiB write/fsync passed after restoration; quota showed
4.52 TB used of 5 TB. Probe files were removed and original checkpoint hashes matched.

Fresh qualification A/B/C (18096965/18096966/18096967) passed all three gradient
paths under producer `b46eb446f153a967869bbfc7f9e93783de986579`, then production
used that same immutable code. Runtime and CPU checkpoint checks verified frozen
weights and batch-normalization buffers in A/B. E/F were registered before their
held-out outcomes and implement the matched 2×2 rate/data-order comparison. F's
first submission was rejected due to expired, completed dependency handles; the
launcher was repaired to verify accounting and omit satisfied dependencies. No
GPU allocation resulted from the rejected submission.

The [report](surface-wave2-results/index.md) includes exact manifests, input and
selected checkpoint fingerprints, all-attempt accounting, validation trajectories,
paired year-block uncertainty, surfaces/velocity diagnostics, limitations and a
suggested next wave. All 166 packed source files preserve their original bytes;
all five derived analysis files reproduce exactly. Twenty-six focused tests pass.
The report distinguishes the training producer from later submission, analysis and
publication commits. Checkpoint files remain on Torch at the documented run roots.

## Initialization diagnosis follow-up

The [initialization diagnosis](surface-wave2-results/initialization/index.md) adds
lead-zero, true-state persistence, spatial-scale and temporal error decomposition
diagnostics at the same 99 origins. Inference-only job 18195421 completed in
163 seconds on one GPU; a one-second launcher failure (18195300) is retained in
its separate accounting. The follow-up used 0.04556 GPU-hours and no retraining.
The main skill graph now includes true-initialized evolution and inferred-state
persistence; the follow-up shows both no-evolution controls and the initial error.
