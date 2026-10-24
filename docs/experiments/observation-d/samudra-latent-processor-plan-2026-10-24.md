<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra-style latent processor: proposed next experiments

**Plan only. Do not launch until the shared-ERA5 results have been reviewed.**
The current experiment retains its original 4k and 8k OBS cooldowns. The five
additional cooldowns were canceled before execution; main checkpoints at every
1k from 2k through 8k remain available.

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

## Performance defaults and microbatch qualification

For future runs, use activation checkpointing **off**, compiled denoising, and
fused AdamW as the agreed baseline. The current compilation is partial because
of deterministic padding graph breaks; retain that known numerical caveat.
A short real-data microbatch screen will compare 1/2/4/8 examples per device
while keeping effective batch eight through accumulation. Choose the fastest
stable size with memory headroom, then requalify it when the larger processor,
resolution, hardware, or numerical implementation changes. Checkpointing is a
memory fallback, not another default ablation. Existing shared-ERA5 runs remain
unchanged, and no new scientific campaign launches before reviewing them.
