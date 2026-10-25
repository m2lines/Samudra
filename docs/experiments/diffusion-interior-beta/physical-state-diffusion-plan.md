<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Physical-state initializer and processor diffusion

Status: the user approved shared ERA5 with all eight channels supplied directly,
and a small GPU qualification. Beta job `217301` completed successfully in
2m52s, exercising OM4 and observation objectives at full resolution on one GPU.
The batch-one OM4 update took 6.43s (3.08 GiB peak); the observation update took
6.03s (8.17 GiB peak). Both diffusers received finite nonzero gradients on both
tasks. These are single-update measurements, not warmed-up production benchmarks.
Production budget and schedule remain subject to the resulting profile.
Outputs: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/physical-state-era5-v1`.
The completion callback was verified on the compute node and handled.

Local verification: 27 targeted CPU tests pass, including real 77-channel loss
backpropagation, per-member monthly averaging, causal state shifting, OM4 window
and ERA5 alignment, optimizer/RNG restoration, both-source replay, and
microbatch loss equivalence with unequal target support. Member vectorization
matches serial execution under identical explicit noise. Checkpointing on/off
preserves outputs and gradients; compilation caches preserve state-dict names,
bypass evaluation, and are removed when copying EMA models.

Beta job `217306` is queued for a two-hour, one-GPU `beta_test` qualification.
It will exercise six effective-batch-eight updates and checkpoint resume, then
run eight isolated profile variants. Production has not launched. The new replay
and optimization paths have CPU coverage but no measured GPU performance yet.

For the provisional 100 OM4 warmup + 500 updates per source + 100 OBS cooldown,
there are 600 OM4 and 600 OBS updates. The earlier batch-one measurements project
serial accumulation to `600 * 8 * (6.43 + 6.03) / 3600 = 16.6` GPU-hours before
replay, evaluation, checkpoint overhead, and steady-state corrections. This is
not an estimate for the optimized replay run; that requires the pending profile.

## Scientific question

Can conditional diffusion over a persistent physical ocean state produce useful,
coherent ensemble trajectories when trained on full OM4 fields and partial
observations? Each ensemble member carries its own sampled history forward.
This replaces the previous deterministic latent processor and stochastic readout;
it is not a frozen OM4 autoencoder or a latent-target diffusion model.

## Proposed architecture

The shared state has 77 normalized physical channels on the existing 180 × 360
grid: nineteen levels each of zonal velocity, meridional velocity, temperature,
and salinity, plus SSH. Observation examples use the same physical meanings and
normalization. Unobserved fields are sampled, not fixed to zero or repurposed as
arbitrary memory. The implemented pair contains 42,439,451 parameters with no forcing adapter
(initializer 26,554,148; processor 15,885,303), verified by the CPU `--describe`
path. The U-Nets use explicit
InstanceNorm. Both denoisers use the repository's Samudra ConvNeXt U-Net
backbone, including down/up sampling and skip connections; neither is a
convolution-only stack. They have independent weights.

- The initializer conditions on nineteen historical five-day bins (95 days),
  field availability, atmospheric forcing, and geographic/calendar context. It
  jointly samples the two physical history states needed by the processor.
- The processor conditions on those two states, the next interval's forcing,
  and context, and samples the next full physical state. It predicts the next
  state directly, not a physical residual added to the latest state. Its EDM
  denoising preconditioning is a separate numerical choice from autoregressive
  residual prediction.
- After each transition, shift the history and retain the sampled new state.
  Never feed back the ensemble mean. There is no learned physical decoder:
  conversion to physical units and observational averaging are deterministic.
- Keep geography/wet support distinct from observation availability. Missing
  observed channels must not prevent the model from sampling physical interiors.

The full initializer history is conditioning information. The two sampled
physical slots are outputs, not two latent channels or two independent ensemble
members. No future ocean observations enter either initializer or processor.
Prescribed future atmospheric forcing is permitted, as in the preceding work.

## What is inherited from the latent-diffusion work

The inspected reference is the `5873/Ocean_Emulator` checkout at source commit
`4150c5b491e229c6bdf9389b281251dc2aff0895`, specifically
`monthly_latent.py`, `monthly_pilot_data.py`, `monthly_model_data.py`,
`shared_era5.py`, and `scripts/run_monthly_replay.py`. The newest forcing contract
is documented in `docs/experiments/observation-d/shared-era5-2026-10-23.md`.

Retain the existing observation samples and splits: train May 1993–July 2013,
validation November 2013–July 2014, and test January 2015–December 2022. Retain
full-latitude area weighting, channel-specific wet masks, existing physical
mean/std statistics, zero-filled missing inputs with explicit availability,
and the exact monthly overlap weights. Physical fields use the fixed OM4
mean/std scales on both tasks, matching the latest latent thread's `Samples`
construction without `use_observation_normalization`; the older pilot's additional
observation renormalization is not inherited. Atmospheric ERA5 inputs retain
the existing observation atmosphere statistics. There is no normalization refit.

OM4 retains its existing recent-data sample pool: 1975 onward through the July
2013 training cutoff. Native interval bounds are authoritative. The source has
six-day jumps at some leap-year boundaries; eligible windows exclude gaps rather
than silently relabeling intervals. The shared-ERA5 reference audits 409 model
windows and 2,702 corresponding intervals without changing the sample pool.

**Approved forcing interface:** both tasks receive all eight normalized ERA5
channels directly, without a learned 8-to-3 adapter. This deliberately removes
the latent thread's three-channel bottleneck. Historical ERA5 preserves the
pre-1993 OM4 windows, and cache contracts must be verified on the execution host.
The latest latent reference uses the same ERA5 inputs but compresses them through
a shared learned 8-to-3 adapter; older versions instead supply native OM4
`tauuo/tauvo/hfds`. Neither older interface is used here. No observed SSH/SST is
used as a future forcing.

## Objectives and required departure from the reference

The latent reference trains a single denoising call against either a five-day
field or a monthly field. Its monthly decoder conditions on an average of latent
states, and its partially observed denoising task only supplies observed target
channels. Those directly sampled monthly outputs need not equal the average of
five-day samples.

That shortcut cannot define this model's full physical transition: observation
examples do not provide clean instantaneous interior fields to corrupt, and a
monthly interior target is not an instantaneous initial-state label.

**OM4:** ordinary conditional denoising supervises the initializer's clean
physical history targets and the processor's clean next-state target. The
implemented objective is initializer denoising plus the mean teacher-forced
processor denoising loss over all six or seven forecast intervals in the month,
with equal weights on those two terms. Training samples corruption levels for
each call. Teacher forcing does not by itself supervise recovery from generated
states. The implemented OM4 replay objective supplies sampled detached past
states as conditioning while retaining gold next-state denoising targets.

**Observations:** sample complete trajectories through initializer and processor,
then score observed physical outputs. Compute each member's calendar-month
interior average from its own five-day physical predictions, before computing
ensemble scores. Use the proposed objective
`0.8 interior + 0.1 SST + 0.1 SSH + 0.1 initial surface completion`, with fair
ensemble CRPS and the existing masks/area conventions. The reused
`spatial_crps` also includes a 0.5-weighted CRPS term on wet-neighbor spatial
increments; this starter therefore does have a spatial-gradient objective. The separate SSH/SST
weights make the intended balancing explicit; this is not a claim of identical
loss values to the older combined 0.2 surface denoising term. Initial completion
uses historical observed surfaces only. Do not supervise an instantaneous initial
interior against the next or preceding month's IAP mean. No direct observation
velocity loss is invented; mixed OM4 training constrains those channels.

Sampling gradients must reach both diffusers. Freezing weights, detaching sampler
steps, or replacing missing interiors with labels would change this objective.
Two members are the minimum for fair CRPS; eight members remain the proposed
validation readout. Report ensemble-mean errors and individual-member spectra,
roughness, spread, and temporal increments separately.

## Replay, performance, and launch gates

**Both observation and OM4 replay are implemented.** Each source has a separate
physical-state buffer with two entries per lead-month, a maximum origin age of
250 optimizer updates, and eleven replay months. Half the effective-batch slots
attempt replay; an empty or expired buffer falls back to fresh examples. Sampling
is uniform across occupied lead-month buckets. Buffers retain each member's two
physical history states, never their ensemble mean. Checkpoints persist buffers,
source-specific draw RNG states, and fresh/replay counters.

Observation replay continues cached members using aligned eight-channel ERA5
and calendar context, then scores the exact future monthly observation window.
It does not reinitialize from later surface observations. OM4 replay conditions
next-state denoising on sampled detached histories and advances those histories
with sampled states. Original initialization age is retained through successive
continuations; entries cannot silently refresh their age. Training-split,
calendar-contiguity, and one-year horizon checks guard both sources. Replay
provides long-lead conditions but does not backpropagate through the detached
cached past. Its effectiveness is not yet established for this architecture.

The runner uses one GPU, effective batch eight, BF16 with FP32 master weights,
and AdamW. Configurable microbatches group compatible temporal lengths; the
observation objective scores each example with its own monthly weights and
missing-target normalization before averaging. Ensemble members are vectorized
along the batch axis with independent noise and InstanceNorm. The supported
optimization switches are training-only U-Net compilation, fused AdamW, and
optional denoiser activation checkpointing. Evaluation remains eager; there is
no distributed runner.

EMA has a half-life of 100 updates. The provisional schedule is 100 OM4-only
warmup updates, alternating mixed training with 500 updates per task, then 100
observation-only cooldown updates. Cooldown resets EMA to raw weights and retains
Adam moments, with LR decreasing to 1e-6. These lengths remain provisional until
the measured replay-inclusive cost is available.

The starter uses eight Heun noise levels: fifteen denoiser evaluations per
sample. With two members and an initializer plus six or seven monthly processor
steps, observation training requires 210 or 240 denoiser evaluations per example
before backward/recomputation. These are full physical-grid U-Nets. This is
substantially different from the latent thread's single-call denoising objective;
its previous speed measurements do not transfer. The pending effective-batch-eight qualification
and profiling must measure memory and update time before a production budget or
ETA can be justified.

The eight profile variants are the baseline, microbatch two/four/eight,
checkpointing off, training-only compilation, fused AdamW, and microbatch two
with checkpointing off. They repeat cached examples to isolate network compute
from data loading; they are not representative sampling or scientific training
comparisons. Different microbatch shapes assign different stochastic draws.
Same-shape parity checks and deterministic fixed-input denoiser checks assess
numerical agreement separately. Timing gains, compile warmup, and memory costs
remain unmeasured until the profile completes; no speedup is assumed here.

Required qualification before any production: finite gradients into both U-Nets
on OM4 and observation objectives; no future-observation leakage; reproducible
member histories; exact monthly physical averaging; masked-land handling;
checkpoint/optimizer/RNG restart; sample support independent of target
availability; and estimated runtime/memory for an actual observation update.
Compare short- and long-lead surface and interior metrics at fixed checkpoints.
Check single-member maps and temporal coherence, not only ensemble averages.

## Questions to resolve before launch

1. Resolved: use shared ERA5 directly on both sources, with eight channels and
   no forcing adapter.
2. What initial/transition sampling budget is acceptable for differentiable
   observation training? A short sampler needs quality and gradient checks;
   inference can use more steps but that difference must be reported.
3. How large should this first pilot be, and should it start with OM4-only
   denoising warmup before mixed observation training? The new architecture cannot
   resume a latent-readout checkpoint as an equivalent continuation.
4. Should the initial comparison keep all three historical initialization tasks
   (surface-only, missing-variable family, spatial gap), or start surface-only?
   Surface-only gives the simplest target-task comparison; extra tasks must not
   leak interior observation labels into initialization.
5. Both-source replay is now implemented and included in the pending
   qualification. Its measured overhead and long-rollout behavior remain open;
   implementation tests do not establish annual stability.

The implementation summary must name the actual resolved settings, parameter
counts, CPU tests, outstanding GPU qualification, and proposed cost. It must not
claim this plan is already a trained or scientifically validated model.
