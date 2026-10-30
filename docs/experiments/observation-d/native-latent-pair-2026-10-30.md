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
| A-shared-era5-native64-samudra-v2 | ConvNeXt U-Net, widths 280/380/480/520 | 180×360 | 64 |

Both use seed 1729, shared ERA5 through the existing 8→32→3 adapter, the same
OM4/OBS dates and sample order, 4,000 alternating source-specific updates (50% each), and 1,000
OBS-only cooldown updates. Effective batch eight uses four GPUs × two examples.
OBS and OM4 loss weights remain 0.5 interior/monthly + 0.5 forecast + 0.1
initializer. Main LR is 1e-4; cooldown linearly decays to 1e-6, resetting EMA at its start. No replay
or latent magnitude penalty is added. Initializer, decoder and forcing-adapter
initial weights are identical between the two new runs; processor construction
preserves the RNG stream. Processor normalization remains GroupNorm for the
small model and InstanceNorm for the U-Net, as in the prior comparison.

Checkpoint every 500 updates and evaluate raw and EMA weights after training
at all ten saved checkpoints using the unchanged nine-origin, eight-member,
32-step validation. Only EMA checkpoints compete for selection. Select on the
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

## Reference protocol

Data, five-day forecast steps, losses and validation follow the
[shared-ERA5 reweighted baseline](shared-era5-obs-reweight-2026-10-26.md) and
[coarse-latent processor comparison](shared-era5-samudra-2026-10-27.md).
Validation origins span November 2013 through July 2014; these same origins
supply the selection curves. The blend equally averages climatology-normalized
SST, geostrophic velocity, EKE, OHC 0–700 m and OHC 700–2000 m errors.
Forcing remains prescribed throughout rollout. This submission includes the
matched validation comparison; it does not yet include new annual rollouts.

CPU checks: 24 tests passed, including native-grid constructor/RNG agreement,
shared forcing, prefetch and deferred evaluation. Parameter counts from the
configured models are 47,372,608 total / 411,776 processor for small, and
130,253,440 total / 83,292,608 processor for Samudra v2. GPU qualification must still
establish runtime memory and throughput.

## Submission

Producer `ffdfd8f6fefb8609176edf4cda29093e6cc73442`.
Beta root: `/projects/ny/lz1955/multiscale/jrusak/runs/native-latent-pair-20261030`.
Each arm has frozen `paths.json`, verified source archive and separate outputs.

| Model | Production (standard QoS) | Qualification (test QoS) |
|---|---:|---:|
| Small | 220340 | 220341 |
| Superseded small U-Net (canceled before starting) | 220342 | 220343 |

All four jobs were submitted and released on October 30; they were pending at
the submission check. Production uses one four-GPU node per model with a 24h
limit. Test jobs have a two-hour limit and cannot duplicate production training.
The queue had 201 pending production jobs and 25 pending test jobs when checked;
start times and native-resolution runtime were unavailable, so no results ETA
is assigned yet. Remote completion callbacks and dedicated quiet monitors are
active, with six-hour fallback checks. Runtime and scientific results are pending.

## V2-sized replacement

The user replaced the undersized U-Net arm with widths 280/380/480/520 and
dilations 1/2/4/8, one block per scale and expansion factor two, following the
v2 backbone configuration. InstanceNorm follows the v2 high-resolution preset;
the residual latent update and 64-channel projections remain unchanged. The
small-processor control (220340/220341) is retained without modification.

The v2 arm retains the existing per-transition processor checkpointing, with
initializer/decoder checkpointing disabled, matching the small control. The
monthly objective backpropagates through its sequence of five-day latent
transitions. Qualification must establish actual memory usage and throughput.

The old pending Samudra jobs were canceled with zero allocated runtime and their
callback retired. Replacement production **220346** and qualification **220347** are submitted
and released, with standard/test QoS respectively. Both were pending at the
release check. Root suffix: `samudra-v2`; producer
`97bd0feb9110ebdee5051066bbd17b3959908e7e`. All 27 targeted CPU tests and commit
checks passed. Source archive integrity and the remote completion hook were
verified. A dedicated quiet monitor covers the replacement; the small arm keeps
its existing monitor.
