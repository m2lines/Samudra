<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Persistent latent state with diffusion readouts: active wave

Report deadline: **2026-09-30 01:17 UTC** (September 29, 9:17 p.m. Eastern).
The user authorized this wave after the diagonal-artifact investigation. Real-grid qualification passed on L40S and H200. The first two-seed launch was stopped after repeated preemption and slow
random-data loading. Host-resident caching passed qualification on L40S. Both replacement seeds
completed 24,000 OM4 updates and pretrained diagnostics. Observation adaptation
was last verified running at 01:04 UTC on September 29; SSH access subsequently
failed. Current remote state is unknown. Comparative results remain pending.

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

The existing composite scores the ensemble-mean forecast. Separately compare
individual-member spatial spectra, their average power and range, and the
spectrum of the ensemble mean against observations on the three saved example
dates. These are distinct quantities: averaging fields can remove power that is
present in each member. Export both mean member spectral error and the error of
mean member power; neither replaces the fixed selection criterion. Report the
three-date scope and unavailable regions explicitly. Monthly latent samples
average independent readouts, so examine instantaneous surface spectra as well.
The reproducible entry point is `samudra.experiments.diffusion_member_spectra`.
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
checkpoint digest. The two pre-adaptation arrays completed; the post-adaptation
arrays were still waiting on training at the last successful remote check.


## Pretraining completed; observation adaptation active

Both seeds completed the full 24,000-update ceiling without production preemption.
Slurm allocation durations were 5:24:00 and 5:20:48, including data-cache preparation.
Recorded fitting time was 19,036 and 18,844 seconds. Selected native denoising
validation losses were 0.07114 and 0.08855; these are training diagnostics, not
observational forecast scores. Adaptation array **24147020** started both seeds
on L40S after verifying their completed pretraining checkpoints. Both seeds completed their 24-origin native controls and three-date observation
map exports; all 80 output files were transferred and checksum-verified.
Scientific comparisons after adaptation remain pending.


Annual evaluation smoke **24168783**, using the selected seed-1730 pretrained
checkpoint, completed one 73-step year with eight readouts per step in 484 seconds
and 23.46 GiB peak GPU allocation. This verifies runtime and memory feasibility;
it is not the final observation-adapted comparison. Fine-tuning is measured at
roughly 33–37 seconds per update, so its 20-hour cap is expected to bind before
6,000 updates. Actual completed updates, wall time and validation curves will
accompany the final report.


## Interim validation and access interruption — September 29

These are the fixed nine-origin validation composites, calculated from eight-member
ensemble means using the same frozen criterion as the deterministic baseline.
They are not held-out report scores, member spectral scores, or CRPS.

| Observation updates | Seed 1729 | Seed 1730 |
| --- | ---: | ---: |
| 0 | 1.98253 | 2.02898 |
| 500 | 1.36134 | 1.30082 |
| 1,000 | 1.21656 | 1.17132 |
| 1,500 | 1.17556 | 1.03414 |

Lower is better. Both remain substantially worse than the selected deterministic
baseline's validation composite of 0.53419. Improvement is not uniform across
components: seed 1729's SST RMSE increased from 0.893 to 1.071 °C between updates
1,000 and 1,500 despite improvement in the composite. The comparison is currently
limited by partial optimization: throughput projects roughly 2,000 adaptation
updates under the 20-hour cap, versus the baseline's selected update 6,400.
This does not establish convergence or an architectural performance ceiling.

At 01:04 UTC, training tasks `24147020_0` and `24147020_1` were running at updates
1,763 and 1,782. Host memory remained about 81.64 and 80.80 GiB, below the 96 GiB
allocations. Native/report arrays `24147537` and `24147538` were waiting on their
training dependencies. The next hourly SSH poll was rejected by authentication;
subsequent fresh connections were refused. Remote job state after the last
successful check is unknown. No jobs were restarted or cancelled because of this
monitoring failure. The existing checkpoint/resume and dependent evaluation
jobs remain the recovery path once access returns.
