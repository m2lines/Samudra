<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Shared ERA5: Samudra-style latent processor

**Submitted, not yet scientifically evaluated.** Compare against the retained
[OBS-reweighted baseline](shared-era5-obs-reweight-2026-10-26.md), selected EMA
blend **1.0648565300778452** at update 5,000. The user accepted that baseline's
surface/interior tradeoff and authorized conditional follow-ups.

## Model and matched protocol

`A-shared-era5-obs50-samudra` replaces the four-block single-scale processor with
`SamudraLatentProcessor`: the repository `UNetBackboneConfig` ConvNeXt U-Net,
widths 64/128/256, dilations 1/2/4, one block per scale, expansion factor two,
InstanceNorm without running statistics, projection to 128 channels and a
small-initialized residual OceanConv output head. Longitude is periodic and
latitude uses the existing backbone's boundary handling. Odd-grid 45×90 output
shape and forcing/latent gradients are tested explicitly.

The processor consumes both 128-channel 45×90 latent slots, the pooled three
forcing channels, and five geographic/calendar channels. It predicts an
increment to the latest slot, then shifts the two-slot memory. Diffusion draws
are not fed back into the processor. This changes both architecture and
GroupNorm→InstanceNorm; it is not an isolated parameter-count ablation.

The initializer, diffusion decoder, 8→32→3 shared ERA5 adapter and their initial
weights are unchanged: replacement construction preserves the global RNG stream.
The processor itself is fresh, not resumed from the small-processor checkpoint.

- Seed 1729; 4k updates, 50% OM4 / 50% OBS; then 1k OBS-only cooldown.
- OBS loss 0.5 monthly interior T/S + 0.5 SST/SSH forecast + 0.1 initializer.
- OM4 loss 0.5 monthly + 0.5 native forecast + 0.1 initializer.
- Main LR 1e-4, cooldown to 1e-6 with EMA reset; effective batch eight,
  four GPUs × two examples, existing clipping and EMA half-life.
- Four-worker bounded CPU prefetch, unchanged normalization, compiled training
  denoiser, eager evaluation. No replay or latent magnitude penalty.
- Save checkpoints every 500 updates; evaluate all ten raw/EMA checkpoint pairs
  after training in parallel, nine origins, eight draws, 32 diffusion steps.
- Select on the original five-component RMSE blend, without spectra. Report
  component curves and surface/interior tradeoffs, not just the aggregate.

## Submission and qualification

Producer `1d012c406cf00a99bf1eb12b83f973897fc1ea2e`; 41 targeted CPU tests and
commit checks passed. Beta production **218495**, standard QoS, one four-GPU
node; short test qualification **218496**, test QoS, four GPUs. A shared lock
prevents duplicate qualification. Production qualifies inline if it starts first.
Qualification verifies exact optimizer/EMA/RNG replay, bitwise prefetch input
agreement, gradients, rank agreement and finite eager evaluation. It records
processor/total parameter counts, memory and warm throughput; those runtime
measurements are pending.

Root: `/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-samudra-20261027`.
Frozen source and data configuration: `code.tar` and `paths.json` at that root.
W&B is online under `ocean_emulators/observational-transfer`; URL pending startup.
Remote callback `a2a6873a-322f-4d1f-8e9c-9560a0d82b94` and a dedicated quiet
monitor cover qualification, training and posthoc evaluation. No runtime ETA is
claimed before queue/progress evidence is available.

## Conditional continuation

If the selected EMA blend improves below **1.0648565300778452**, automatically
implement, qualify and launch the matched eight-channel ERA5 forcing interface
on this processor. If that then improves against this arm, automatically proceed
to the higher-resolution latent arm. These follow-ups are authorized; do not
ask again merely because a successful stage completes. Preserve the same
training budget, loss, data, validation and batch across stages. Stop and report
if a gate fails or qualification exposes a constraint requiring a scientific
change. No subsequent stage is submitted yet.
