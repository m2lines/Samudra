<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Persistent latent state with diffusion readouts: active wave

Report deadline: **2026-09-30 01:17 UTC** (September 29, 9:17 p.m. Eastern).
The user authorized this wave after the diagonal-artifact investigation. Training
has not yet passed the real-grid qualification gate; this page describes the
experiment and will be replaced with measured execution and results.

## Scientific comparison

Surface history and past forcing initialize two latent memory slots. A learned
processor advances those slots using coarse forcing and geographic/seasonal
context. A conditional diffusion decoder reads physical fields from the latent
state at initialization and future leads. Decoded fields, interior targets and
readout noise never enter recurrence. OM4 initial and future denoising losses
train the encoder, processor and decoder jointly from scratch.

This directly tests whether a learned recurrent representation avoids requiring
information passed between steps to masquerade as a physical ocean state. It
also replaces the previous frozen physical dynamics with a trainable processor,
so a difference in skill cannot be attributed solely to where information is
stored.

The initial configuration uses the existing wide surface-history encoder,
19 five-day input frames, two 128-channel latent slots on a 45×90 grid, four
residual processor blocks, and a width-192 diffusion decoder. Training and
readout targets are the original 77 physical fields at 180×360. The processor
receives the existing three native OM4 forcings during pretraining and the
existing learned ERA5 adapter during observation adaptation. There is no new
forcing, data resolution or observation target in this wave.

Diffusion samples readouts independently at each lead on a **deterministic latent
trajectory**. This is not diffusion in latent space and does not propagate a
sampled latent initialization. Monthly sample variance therefore depends on a
different temporal covariance assumption from physical-state diffusion. We will
report that distinction explicitly and inspect temporal consistency rather than
call these coherent ensemble trajectories.

## Execution gates and budget

- Focused tests: target-independent recurrence, future-observation exclusion,
  gradient flow, anchoring, checkpointed denoising RNG/gradient parity, and
  optimizer-boundary resume.
- Full-grid qualification: native initial/future denoising backward, observational
  monthly/surface-loss backward, finite nonzero encoder/processor/decoder
  gradients, memory/throughput measurements, actual checkpoint resume and strict
  weight reload.
- Two independent seeds (1729 and 1730) if qualification confirms feasibility.
  The production update/time caps will be set from measured throughput with time
  reserved for observation fine-tuning, evaluation and the report.
- Use single-GPU Engaging jobs, existing Rust loading and optimizer-boundary
  checkpoints. Count allocations, preemptions and evaluations against the
  remaining 576 GPU-hour campaign authorization. Prior completed campaigns and
  artifact diagnostics account for approximately 62.16 GPU-hours before this wave.
- Stage completion at a wall-time cap does not imply completion of the requested
  update ceiling; both actual updates and time will be reported.

## Evaluation

Compare to the selected unchanged deterministic baseline and the completed
physical-state diffusion wave using the same splits, normalization, frozen score
reference and held-out cohorts. Include monthly interior point and probabilistic
metrics, surface scores, calibration, instantaneous salinity/velocity fields,
diagonal correlations, member versus ensemble-mean structure, temporal
consistency, and the existing annual rollout diagnostics as budget permits.
Distinguish OM4 model-world interior/velocity controls from observational skill.
Five-day binned surface observations and monthly IAP fields are the available
measurements; this wave does not establish daily or independent Argo-profile skill.

The final report will state incomplete evaluations, partial training or negative
results rather than substituting nominal configuration for observed execution.
