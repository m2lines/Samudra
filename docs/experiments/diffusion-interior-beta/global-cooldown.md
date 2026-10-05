<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global diffusion: endpoint learning-rate cooldown

Authorized October 4 after the "Unet limitations" study. Preserve the original
16,000-update endpoint and append a separately recorded 2,000-update continuation.
The original milestone reports and fixed-exposure comparison are not delayed by
this follow-up. No cooldown updates have run yet.

## Motivation and interpretation

Thread `01a0ed26-4726-7430-8163-555020c74b07` reports paired toy continuations
from identical weights, optimizer state and subsequent examples. A late cosine
cooldown reduced held-out grain RMS by about 54% and 40% across two seeds.
The source is `experiments/noise_study/cooldown_branch.py` and the completed
`experiments/noise_study/REPORT.html` in the study checkout at
`/home/jder/Ocean_Emulator`. That study explicitly did not retrain the ocean model.
Its cosine schedule ends at one percent of peak LR; this continuation uses the
same relative reduction, with a shorter update budget chosen for ocean cost.

This is a test of whether an inexpensive continuation improves the ocean result.
Endpoint versus cooldown includes extra optimization and training exposure as well
as lower LR. It is not a causal isolation of LR: that would require a matching
constant-LR continuation. No additional control arm is included in this request.

## Fixed continuation recipe

- Start from the **completed** global step 16,000 `train/last.pt`, containing the
  real AdamW moments and RNG state, at 8,000 OM4 / 8,000 observation updates.
  Milestone weights alone are rejected. Preserve the original output directory.
- Train all the same parameters with effective batch eight, existing data,
  normalization, masks, noise, losses, clipping, weight decay and 32 sampling steps.
- Cosine LR from `1e-4` on the first additional update to `1e-6` on the last.
  LR depends on absolute update number, so preemption does not restart the decay.
- Hold the original quadratic schedule's terminal task proportion at **seven
  observation updates per OM4 update**. This adds 1,750 observation and 250 OM4
  updates, reaching **9,750 observation / 8,250 OM4** at step 18,000. Continue
  existing task sample and noise counters; do not replay the beginning of training.
- Write to `cooldown/`, record parent checkpoint checksum, original and new producer,
  recipe and exact exposure in the contract. Save optimizer checkpoints every ten
  updates, plus immutable weights at the start, midpoint and endpoint.

On the current four-H200 throughput this adds roughly **32–34 hours / 130–135
GPU-hours**, excluding queue delays and preemption. Eight RTX GPUs would take
roughly 22 hours at the previously observed throughput. Count the follow-up within
the existing 900-GPU-hour allowance and recheck actual allocation before launch.
Keep the original endpoint report available while this continuation runs.

## Execution and validation

`diffusion_global_train --cooldown-parent ...` validates the complete original
contract against its successful qualification and refuses an incomplete endpoint,
weights-only checkpoint, changed data/loss settings, or reuse of the parent output
directory. Later resumes require exact continuation contract and exposure matches.
The original 16k producer remains pinned; stage the continuation under its own
immutable Git producer. The original model and loss modules are unchanged.

The Slurm launcher supports `MODE=cooldown`, with `CODE_COMMIT` set to the new
producer and `PARENT_COMMIT` set to the original qualified producer. Use the same
cluster root as the endpoint, or migrate and checksum-verify the full optimizer
first. Launch only after the original run completes, never concurrently with a
writer to `train/last.pt`. Inspect the first native and observation updates and
perform a checkpoint/resume check before continuing through the tail.

CPU tests check exact exposure, LR endpoints, monotonic decay, refusal of invalid
parents, and bitwise optimizer/RNG continuation across an interruption. GPU
continuation qualification is pending the original endpoint.

Evaluate the original, midpoint and cooled endpoints on the same nine validation
origins with eight members, identical evaluation seeds and 32 inference steps.
Compare individual-member and ensemble-mean spectra separately, grain and spatial
increment errors, RMSE, fair CRPS, spread and rank histograms, and the frozen
integrated/spectral composite. Use matched monthly targets for interior accuracy;
an instantaneous interior picture is not a monthly error measurement. Smoother
maps alone do not count as success. Preserve the original fixed-exposure result
separately from any validation-selected cooldown result and final held-out test.
