<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra-style latent processor: proposed next experiments

## Authorized sequence (updated 2026-10-27)

The user accepted experiment 0's surface/interior tradeoff and explicitly
retained OBS loss **0.5 monthly T/S + 0.5 SST/SSH forecast + 0.1 initializer**.
The original five-component climatology-normalized RMSE blend remains unchanged.
The new baseline's selected EMA blend is **1.0648565300778452**, at update 5,000.
The earlier requirement that experiment 0 beat the original weighting is revoked.

1. **Samudra-style processor:** replace the small processor with the repository
   ConvNeXt U-Net, preserving the shared ERA5 8→32→3 adapter and coarse latent
   representation. Submitted as production `218495` and short qualification
   `218496`; [run protocol](shared-era5-samudra-2026-10-27.md).
2. **Automatically proceed if step 1 improves the blend:** remove the three-field
   bottleneck, supplying all eight normalized ERA5 channels to initializer and
   processor, retaining the new U-Net.
3. **Automatically proceed if step 2 improves against step 1:** increase latent
   spatial resolution on top of step 2. Retain 128 channels for a resolution-only
   contrast if memory qualification permits. If it does not fit, report the
   constraint before changing the capacity or batch size.

The user explicitly authorized these conditional follow-ups on 2026-10-27.
At each stage verify full results and compare selected EMA checkpoints on the
same 500-update grid, using the unchanged validation cohort and blend. Report
all components. Stop and report if a stage does not improve; a one-seed win is
screening evidence rather than a replicated claim.

Each arm trains from scratch for **4k main updates (50% OM4 / 50% OBS) + 1k
OBS-only cooldown**, with effective batch eight on four GPUs, seed 1729,
existing LR/EMA reset/objectives, compiled training denoising and eager evaluation.
Retain parallel bounded OM4 prefetch. Save full checkpoints every 500 updates;
evaluate raw and EMA weights at full precision after training, across four GPUs.
No batch-size, replay, additional seed, or latent-bound changes are included.

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
