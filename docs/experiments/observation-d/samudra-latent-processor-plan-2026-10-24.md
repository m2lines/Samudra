<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra-style latent processor: proposed next experiments

## Authorized sequence (2026-10-26)

The user approved a sequential, improvement-gated campaign after reviewing
shared-ERA5. This section supersedes the older exploratory menu below; it does
not authorize launching that entire menu.

0. Reweight OBS training loss from 0.8 monthly interior / 0.2 SST–SSH forecast /
   0.1 initializer to **0.5 / 0.5 / 0.1**, in both main training and OBS-only
   cooldown. Keep the completed shared-ERA5 architecture and forcing adapter.
   OM4 loss remains 0.5 monthly / 0.5 native forecast / 0.1 initializer.
   Compare against the completed shared-ERA5 4k + 1k baseline.
1. If step 0 improves the existing validation RMSE blend, replace the small processor with the Samudra-style latent U-Net; retain
   shared ERA5 and its existing 8→32→3 adapter.
2. If step 1 improves the existing validation RMSE blend, remove the three-channel
   forcing bottleneck: pass all eight normalized ERA5 channels to both
   initializer and processor, retaining step 1's U-Net.
3. If step 2 improves the same score, increase latent spatial resolution on
   top of step 2. Retain 128 channels for a resolution-only comparison if
   memory qualification permits; a reduction to 64 is a separate capacity
   tradeoff and must be identified explicitly before launch.

Each arm starts fresh with 4k main updates (50% OM4 / 50% OBS) followed by
1k OBS-only cooldown, matching the completed shared-ERA5 4k comparison.
Retain effective batch eight on four GPUs, training targets (with the OBS
coefficients changed only as specified in step 0),
main LR, cooldown and EMA-reset protocol, validation cohort and ensemble
settings. Save checkpoints every 500 updates and evaluate raw/EMA curves after training,
parallelized over checkpoints with the existing full validation precision. These
architecture/interface changes are not exact resumes of the old processor.
Use compiled training denoising and eager evaluation, as qualified in the
shared-ERA5 study. Qualify each changed architecture before production.

**Selection unchanged:** the user retained the existing RMSE blend and withdrew
v2. Average the five fixed-climatology-normalized RMSEs equally: SST,
SSH-derived velocity, EKE, upper OHC and deep OHC. Salinity remains supervised
in monthly OBS training but is not directly part of checkpoint selection.
Compare each candidate with its immediate predecessor using this same rule;
report every component and raw/EMA curves. Carry forward the OBS reweighting
only if experiment 0 improves the blend. If a gate fails, report it and discuss
the next step rather than automatically launching the remaining sequence.
Treat one-seed improvements as screening evidence, not a replicated conclusion.

Experiment 0 passed qualification in `218269`; logging-only recovery is Beta job
`218327`; see the [run report](shared-era5-obs-reweight-2026-10-26.md).
Later stages have not been submitted.

## Baseline change

Replace the four-block, single-scale `PersistentLatentProcessor` with the
repository's ConvNeXt `UNetBackboneConfig`, using explicit configuration rather
than an independently implemented U-Net. Start with widths64/128/256,
dilations1/2/4 and one block per scale, a projection to latent width and a
small initialized residual output head. This is a Samudra-style **latent**
evolver, not a pretrained physical-state Samudra checkpoint.

Preserve the processor interface: concatenate two latent time slots, forcing
and five geographic/calendar channels; predict an increment to the latest
slot, then shift the memory. Keep the existing initializer, diffusion decoder,
128-channel45×90 latent grid, data, objectives, effective batch and training
schedule fixed initially. Use InstanceNorm without running statistics as in
the existing experimental U-Nets; document that this also changes normalization
from the small processor's GroupNorm. This first contrast is the entire
processor replacement, not a pure parameter-count or receptive-field ablation.

Qualify odd-grid down/up sampling (45×90), boundary handling, exact output
shape, forcing gradients, full optimizer/EMA/RNG resume, memory and warm
throughput. Record processor and total parameter counts and wall time; do not
infer speed from total model size. Train from scratch: old convolutional
processor weights are not interchangeable. Keep old runs as context rather
than claiming seed-matched architecture controls.

## Small first comparison

Choose the forcing convention after reviewing shared-ERA5 results. If useful,
use the same ERA5 adapter for OM4 and OBS in every new mixed-data arm; otherwise
state the alternative explicitly before launch. Do not choose separate forcing
conventions for different new arms.

| Proposed model | Main updates | Main data/task mix | Purpose |
|---|---|---|---|
| U-latent-E | 2k | 100% OBS, surface initialization | New OBS-only baseline |
| U-latent-A | 2k | 50% OBS, 50% OM4; surface initialization | Does model-data supervision help the larger processor? |

Both receive a separate1k OBS-only cooldown, with EMA reset and the existing
1e-4→1e-6 linear decay. Keep main LR1e-4 and one common seed initially, no replay
or latent bound. Save main checkpoints and raw/EMA validation every250 updates.
Distinguish model-data mixing (A) from masked-initialization multitasking (B).

## Experiments on the new baseline, staged rather than one large sweep

1. **Longer training and marginal data value:** continue the uncooled U-latent-A
   checkpoint with equal numbers of extra updates, one OBS-only and one50/50
   OM4/OBS, then identical cooldowns. This directly tests whether extra model
   data help from the same starting state. Preserve the main trajectory for
   possible4k/8k extensions after inspecting curves. Total-update and OBS-update
   axes both matter; equal total updates are not equal OBS exposure or FLOPs.
2. **Forcing reliance:** repeat both new A and E baselines with processor forcing
   zeroed, retaining historical forcing in initialization. Compare the change
   relative to each model's own forcing-enabled control.
3. **Initialization multitasking:** U-latent-B keeps A's data mix but uses, on
   OM4 updates,50% surface initialization,25% missing-variable initialization,
   and25% spatial-gap initialization. Same initializer, processor and decoder;
   cooldown remains surface-only. This isolates the extra initialization tasks.
4. **Latent resolution:** compare A against native1° latents with128 channels
   to isolate spatial resolution. A64-channel native variant is a separate
   memory-saving tradeoff, not a clean resolution-only test. Qualify cost first.
5. **Later, if warranted:** replay for longer OBS leads and CM4 diversity.
   Keep these separate initially; historical ERA5 cannot simply be date-matched
   to an independently evolving coupled CM4 simulation.

## Evidence to report

Use the established RMSE-only validation blend and show its individual surface
and upper/deep OHC components, raw and EMA weights, and cooldown boundaries.
Report both30-day and365-day behavior on fixed cohorts; do not select on the
held-out annual test. Include initialized persistence, matched maps, interior
profiles, single draws and ensemble means. For OM4 and OBS separately inspect
supervised denoiser RMSE across diffusion steps, so a better pooled score does
not hide a still-failing full-field decoder. OHC skill alone is not proof of
accurate interior structure.

Treat one-seed differences as screening evidence. Replicate the promising
contrast before claiming a model-data benefit. The completed shared-ERA5 study
will determine whether decoder/forcing issues need fixing before this launch.

## Performance defaults and distributed batch comparison

Use activation checkpointing **off**, compiled denoising, and fused AdamW as
the agreed baseline. Current compilation is partial because of deterministic
padding graph breaks. Checkpointing is a memory fallback. Recheck memory and
throughput when the processor, resolution or numerical implementation changes.
The [single-GPU screen](diffuser-performance-2026-10-24.md#microbatch-results)
favored microbatch eight (~54 GB allocated), but its effective batch was eight
on one GPU; that does not imply microbatch eight on each of four GPUs.

The user now authorizes migrating shared-ERA5 to **four data-parallel GPUs,
microbatch two per rank, accumulation one: effective batch eight**. Qualify
identical global examples, per-example targets/noise, gradient weighting,
optimizer/EMA/RNG resume and validation behavior before switching. Preserve
the 8k main schedule, sample exposure, and original 4k/8k cooldowns. Each
cooldown must also use four GPUs; do not overlap the old independent GPU1 fork
with a four-rank main process. Keep the original pending production held until
the replacement passes its qualification gate.

After reviewing shared-ERA5, add a **batch-eight versus batch-32 comparison**
before or alongside the new processor experiments: four GPUs with microbatch
two versus eight, no accumulation, same architecture/data mix/optimizer/seed.
Start with unchanged LR to isolate batch size rather than automatically scaling
it. Report curves against examples seen, optimizer updates and wall time. The
primary comparison should match total example exposure and cooldown exposure,
with the LR/EMA schedules expressed in equivalent exposure units; at batch32,
a matched-example run has one quarter as many updates. An equal-update endpoint
is a separate comparison using four times as much data exposure, not a matched
batch-size ablation. Fix these budgets explicitly before launching this future
comparison. No batch32 or new-architecture scientific run launches until the
shared-ERA5 results are reviewed.
