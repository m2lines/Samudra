<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Physical-state initializer and processor diffusion

Status: the user approved shared ERA5 with all eight channels supplied directly,
and a small GPU qualification. Beta job `217301` is submitted to `beta_test`
for two updates exercising OM4 and observation objectives at full resolution,
using one GPU and a one-hour limit. Submission does not establish a pass.
Production budget and schedule remain subject to the resulting profile.
Outputs: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/physical-state-era5-v1`.
The job exit callback and a two-hour fallback reminder are installed.

Local verification: 16 targeted CPU tests pass, including real 77-channel loss
backpropagation, per-member monthly averaging, causal state shifting, OM4 window
and ERA5 alignment, and optimizer/RNG restoration. Required pre-commit checks
pass. No full-resolution GPU execution or throughput measurement has occurred.

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
states. Replay is not implemented in this starter.

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

**This starter does not implement replay.** Both-source replay was requested for
the preceding pilot, but the current architectural reset is not represented as
having inherited it. It remains an explicit extension requiring source-specific
physical-state buffers, correct forcing/calendar offsets, no future observation
reinitialization, age limits, and restart tests. The newest shared-ERA5 latent
reference has no replay; its older replay branch only replays observations.

The implemented starter uses one GPU, microbatch one with accumulation to
effective batch eight, BF16 with FP32 master weights, and AdamW. It has EMA with
half-life 100 updates, no compilation, and no distributed runner. The provisional
configurable schedule is 100 OM4-only warmup updates, alternating mixed training
with 500 updates per task, then 100 observation-only cooldown updates. Cooldown
resets EMA to the raw weights and retains Adam moments, with LR decreasing to
1e-6. This follows the latent thread's cooldown mechanism but does not assume
that its training lengths are appropriate for this substantially costlier model.
The qualification can set warmup/cooldown lengths to zero to exercise both tasks
within two updates. Production lengths will be chosen after profiling.

The starter uses eight Heun noise levels: fifteen denoiser evaluations per
sample. With two members and an initializer plus six or seven monthly processor
steps, observation training requires 210 or 240 denoiser evaluations per example
before backward/recomputation. These are full physical-grid U-Nets. This is
substantially different from the latent thread's single-call denoising objective;
its previous speed measurements do not transfer. The first GPU qualification
must measure memory and update time before promising a run budget or ETA.

Latest latent-thread optimizations (training-only compilation, fused AdamW,
larger microbatches, and DDP) are possible later additions, not implemented or
qualified features of this starter.

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
5. Both-source replay remains outstanding before a substantive stability
   comparison. The immediate approved qualification tests the no-replay starter;
   its results do not establish annual stability.

The implementation summary must name the actual resolved settings, parameter
counts, CPU tests, outstanding GPU qualification, and proposed cost. It must not
claim this plan is already a trained or scientifically validated model.
