<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Do future forcings explain the A/E gap?

Train two fresh, seed1729 models against the original A/E controls:

| Model | Main phase | Cooldown | Processor forcing |
|---|---|---|---|
| A-no-processor-forcing | 2k updates,50% OM4/50% OBS | 1k OBS-only | Zero |
| E-no-processor-forcing | 2k updates,100% OBS | 1k OBS-only | Zero |

Keep historical forcing in the initializer. Zero forcing inside the processor,
after any adapter and before its convolutions, for every training and inference
call. Calendar/geography remain; no other weather input enters the processor.
This removes prescribed future atmospheric information, not historical
atmospheric information encoded in the initial latent state.

Architecture, seed, data windows, losses, normalization, batch8, main LR1e-4,
clip1, EMA half-life100 and1k cooldown to1e-6 match original A/E. Reset EMA at
cooldown, retain Adam moments. Original128-channel45×90 two-slot latents; no
replay/bound or shared-ERA5 treatment. Same initialization parameter layout and
seed as original A/E; qualification updates are disposable.

## Curves and interpretation

Save EMA and raw-weight validation every250 updates, with SST/velocity/EKE,
upper/deep OHC components and unchanged RMSE-only selection. Save EMA mean
predictions and references at each validation for per-origin and seasonal-
anomaly analysis. Show cooldown boundaries explicitly. Plot against both total
updates and OBS updates: during main A has1k OBS updates, E has2k; including
cooldown totals are2k and3k. The exposure axis is informative but does not itself
remove curriculum or total-compute differences.

Primary contrast: how much removal worsens A versus E, and how their gap changes
before and after cooldown. A larger loss of E's surface advantage supports
stronger reliance on prescribed atmospheric information, not proof of a bug or
an illegitimate shortcut. Original monthly A has better OHC errors than E;
OHC integrals do not establish better detailed interior structure. Compare
surface and interior separately and inspect per-origin variability.

The most direct later test of marginal value would fork one common checkpoint
into equal-update OBS-only and mixed/model-task continuations with the same
cooldown. That is a proposed follow-on, not an additional launch in this wave.

All real data/model work runs on Beta. Tiny local synthetic tests verify that
changing future forcing cannot change the disabled processor output or receive
gradients, while historical forcing still affects initialization. Qualification
checks real data, finite losses and exact serialized optimizer/EMA/RNG replay.
The annual loader restores the stored architecture flag, preventing accidental
re-enabling of forcing when evaluating these checkpoints.
