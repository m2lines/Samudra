<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global diffusion: cooldown within the 16k run

Authorized October 4 after the "Unet limitations" study. Apply a cosine LR cooldown
**over the final 2,000 updates of the existing 16,000-update run**. Total exposure
remains exactly 8,000 OM4 + 8,000 observation updates at effective batch eight.
There are no additional training updates or material training-time increase.
Compare against earlier checkpoints, including a new checkpoint immediately
before the taper. No cooldown updates have run yet.

## Motivation and interpretation

Thread `01a0ed26-4726-7430-8163-555020c74b07` reports paired toy continuations
from identical weights, optimizer state and subsequent examples. A late cosine
cooldown reduced held-out grain RMS by about 54% and 40% across two seeds.
The source is `experiments/noise_study/cooldown_branch.py` and the completed
`experiments/noise_study/REPORT.html` in the study checkout at
`/home/jder/Ocean_Emulator`. That study explicitly did not retrain the ocean model.
Its cosine schedule ends at one percent of peak LR; this run uses the same relative
reduction, with a shorter cooldown chosen to fit the existing ocean budget.

The objective here is the best result within the authorized run. Comparing earlier
checkpoints with the cooled endpoint includes more training as well as lower LR.
It does not isolate LR causally; a constant-LR continuation at equal exposure would
be needed for that. Appending extra training would not itself solve that problem.
The user selected this within-budget approach; no additional control arm or
post-16k extension is planned.

## Fixed recipe

- Resume the current run's real model, AdamW moments and RNG state at a normal job
  boundary, before step 14,000. Retain the original qualified scientific contract
  and record the new producer and LR schedule explicitly.
- Keep LR at `1e-4` through global update 14,000. Save immutable weights and a full
  optimizer checkpoint immediately before cooldown.
- Cosine LR from `1e-4` on additional update 1 (global update 14,001) to `1e-6` on
  global update 16,000. LR depends on absolute update number, so preemption does
  not restart the taper.
- Keep the **original quadratic task schedule exactly**, with no changed task
  mix or reset of sample/noise counters. Step 14,000 has 7,657 OM4 / 6,343 observation
  updates; the last 2,000 updates add 343 OM4 / 1,657 observation updates.
- Keep the same trainable parameters, data, normalization, global masks, noise,
  losses, clipping, weight decay, effective batch and 32 sampling steps.
- Continue rolling optimizer saves every ten updates and existing observation
  milestones. Also save weights at global steps 14,000, 15,000 and 16,000.

The existing 900-GPU-hour allowance still applies, including restarts/evaluation.
The cooldown adds only checkpoint and handoff overhead, not a separate training
budget. The final fixed-exposure result will use the tapered LR; a constant-LR
16k endpoint is no longer planned.

## Execution and validation

`diffusion_global_train --cooldown-parent ...` validates the original contract
against its successful qualification and rejects a weights-only checkpoint, a
parent after cooldown should have started, changed data/loss settings, or reuse of
the parent output directory. Later resumes require exact continuation contract
and exposure matches. The original running allocation keeps its immutable producer;
apply the revised trainer at its next job boundary. Model and loss modules are
unchanged.

For auditability, resumed output goes to `train-cooldown/` and references the frozen
handoff `train/last.pt`. This is one continued run, not an additional arm. The
Slurm launcher supports `MODE=cooldown`, with `CODE_COMMIT` set to the new producer
and `PARENT_COMMIT` set to the original qualified producer. Keep both directories
on any cluster migration, checksum-verify optimizer transfers, and allow only one
active writer. Inspect the first native and observation updates and verify a
checkpoint/resume before continuing. GPU handoff is pending the next job boundary.

CPU tests check unchanged task/exposure sequences at every global update, LR
endpoints, monotonic decay, rejection of invalid parents, and bitwise optimizer/RNG
continuation across an interruption. Preserve
`train-cooldown/cooldown-start-optimizer.pt` at global step 14,000 for any later
controlled follow-up.

Evaluate earlier milestones, pre-cooldown, midpoint and final checkpoints on the
same nine validation origins with eight members, identical evaluation seeds and
32 inference steps. Compare individual-member and ensemble-mean spectra separately,
grain and spatial increment errors, RMSE, fair CRPS, spread and rank histograms,
and the frozen integrated/spectral composite. Use matched monthly targets for
interior accuracy; an instantaneous interior picture is not a monthly error
measurement. Smoother maps alone do not count as success. Retain the fixed-exposure
endpoint separately from validation selection and final held-out test results.
