<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native-resolution latent processor comparison

Compare the small convolutional processor and Samudra ConvNeXt U-Net with
**two 64-channel latent slots on the full 180×360 grid**. Both runs start from
scratch. This directly compares processors at the same latent resolution and
width; comparison to the earlier 45×90/128-channel runs also changes width.

| Run | Processor | Latent grid | Channels per slot |
|---|---|---|---:|
| A-shared-era5-native64-small | Four residual convolutional blocks | 180×360 | 64 |
| A-shared-era5-native64-samudra | ConvNeXt U-Net, widths 64/128/256 | 180×360 | 64 |

Both use seed 1729, shared ERA5 through the existing 8→32→3 adapter, the same
OM4/OBS dates and sample order, 4,000 mixed updates (50% each), and 1,000
OBS-only cooldown updates. Effective batch eight uses four GPUs × two examples.
OBS and OM4 loss weights remain 0.5 interior/monthly + 0.5 forecast + 0.1
initializer. Main LR is 1e-4; cooldown ends at 1e-6 with an EMA reset. No replay
or latent magnitude penalty is added. Initializer, decoder and forcing-adapter
initial weights are identical between the two new runs; processor construction
preserves the RNG stream. Processor normalization remains GroupNorm for the
small model and InstanceNorm for the U-Net, as in the prior comparison.

Checkpoint every 500 updates and evaluate raw and EMA weights after training
using the unchanged nine-origin, eight-member, 32-step validation. Select on the
original five-component climatology-normalized RMSE blend and show all component
curves. Report both matched new runs alongside the earlier coarse-latent pair.
No further scientific arms are automatically authorized by this comparison.

Each run uses its own full four-GPU Beta allocation, online W&B, parallel data
prefetch and compiled training denoising. A short beta_test qualification races
an inline production qualification under a shared lock. Qualification must pass
exact serialized optimizer/EMA/RNG replay, exact prefetch inputs, finite losses
and full-grid memory checks before training. Qualification weights are discarded.
If native-resolution memory does not fit, preserve failures and assess a matched
activation-checkpointing recovery without changing effective batch or science.

The canceled October 23 native-latent run is not reused: it used native OM4
forcing and different OBS weights. These two fresh runs use the current shared
ERA5 baseline. Submission IDs and runtime measurements follow below.
