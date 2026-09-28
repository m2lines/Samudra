<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Persistent latent state with diffusion readouts: active wave

Report deadline: **2026-09-30 01:17 UTC** (September 29, 9:17 p.m. Eastern).
The user authorized this wave after the diagonal-artifact investigation. Real-grid qualification passed on L40S and H200. The first two-seed launch was stopped after repeated preemption and slow
random-data loading. Host-resident caching passed qualification on L40S; replacement pretraining
is submitted and observation adaptation is queued behind it. Comparative results remain pending.

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
  Each seed has a 24,000-update / 16-hour pretraining cap, followed by a
  6,000-update / 20-hour observation cap, whichever comes first. These caps leave
  time for evaluation and the report; matching earlier update counts is not promised.
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
Export the same three observation-input dates from pretrained and adapted
checkpoints, so the timing of any reappearing artifact can be checked.
Distinguish OM4 model-world interior/velocity controls from observational skill.
Five-day binned surface observations and monthly IAP fields are the available
measurements; this wave does not establish daily or independent Argo-profile skill.

The final report will state incomplete evaluations, partial training or negative
results rather than substituting nominal configuration for observed execution.

## Observed launch state

Source `0ef9d79a6` passed real-grid qualification (`24144434`) on an H200.
The earlier L40S gate (`24144026`, source `db7ece865`) also passed. Initial
measurements were 2.89 s for an OM4 forward/backward/update and 21.37 s for an
observation update on H200; peak observed allocation was 20.99 GiB. These single
qualification measurements exclude sustained checkpoint/validation overhead and
will be replaced with production throughput.

Pretraining array **24144680** started two single-H200 tasks (seeds 1729/1730),
but both experienced preemption and production updates took 11–22 seconds with
random reads. It and the dependent observation array **24144688** were stopped;
outputs are preserved. GPU-cache qualification **24145089** was also preempted
during warm-up. These are operational bring-up attempts, not completed training
results. Jobs support optimizer-boundary resume. Qualification exercised actual save/resume and
strict fixed-noise loss reload. Training provenance binds data, observation
normalization and the frozen validation reference. No held-out test data enter
training or checkpoint selection.

### Cache and scheduling recovery

Host-cache qualification **24146352** passed on a single L40S, using source
`8291df4dc`, four allocated CPUs, 16 Rust read threads and 96 GiB host RAM.
It caches the same float32 prepared frames in host memory and moves selected
windows to the GPU. Real native-versus-cache comparisons check masks, forcing,
labels, edge windows and repeated/shuffled requests before training.

Standard GPU queue estimates were too late for the deadline, including the
available advanced GPU account/QoS, so production is not waiting on that queue.
A tested per-invocation time limit is available for shorter allocations; it
preserves the global stage budget and does not mislabel an interrupted stage as
complete. The first completed/terminated latent bring-up allocations consumed
**0.4472 GPU-hours**, including all preempted attempts (Slurm duplicate accounting
records); the active host-cache qualification is additional.


### Qualified replacement launch

Source `8291df4dc` passed exact prepared-frame equivalence, finite nonzero gradients
in encoder/processor/decoder, optimizer resume, and identical fixed-noise loss
following strict reload. Qualification measured 3.33 s for a native update and
32.11 s for an observation update; peak host RAM was 62.9 GiB and GPU allocation
20.95 GiB. These timings are individual measurements, not sustained throughput.

Replacement pretraining array **24147018** and dependent observation array
**24147020** each contain seeds 1729 and 1730, using one L40S and 96 GiB host RAM
per task. Run group: `latent-d192-host-v1`. The update/time ceilings above remain
unchanged. The initial report smoke used too few validation origins to compute
the required spectral comparison; the corrected full-validation smoke is
**24147149** and passed point-score, eight-member calibration and structure
exports using qualification weights solely to test reporting. It is not a
scientific evaluation of trained skill.


Both seeds passed 100 updates and wrote resumable checkpoints. The initial
steady native update interval was approximately 0.7–0.8 s after cache preparation;
this is faster than the cold qualification measurement, but does not yet include
a full production validation interval.

Evaluation source `bd6044a2b` is staged separately from the immutable training
source. Dependent arrays are native OM4 controls before adaptation (**24147533**),
fixed observation-input maps before adaptation (**24147534**), native controls
after adaptation (**24147537**), and full observation/annual reports (**24147538**).
Each follows its corresponding seed's completed stage and verifies the selected
checkpoint digest. These are queued evaluations, not completed results.
