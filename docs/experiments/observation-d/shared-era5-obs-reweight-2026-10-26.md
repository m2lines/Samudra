<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Shared ERA5: balance OBS interior and surface training losses

**Submitted, no scientific results yet.** This is experiment 0 of the
[sequential processor campaign](samudra-latent-processor-plan-2026-10-24.md).
It tests whether emphasizing surface forecast supervision improves the existing
[shared-ERA5 4k + cooldown baseline](shared-era5-2026-10-23.md).

## Matched comparison

| Setting | Existing shared-ERA5 baseline | A-shared-era5-obs50 |
|---|---|---|
| OBS monthly interior T/S denoising | 0.8 | **0.5** |
| OBS five-day SST/SSH forecast denoising | 0.2 | **0.5** |
| OBS initializer SST/SSH reconstruction | 0.1 | 0.1 |
| OM4 monthly / five-day / initializer losses | 0.5 / 0.5 / 0.1 | unchanged |
| Main training | 4,000 updates, 50% OM4 / 50% OBS | unchanged |
| Cooldown | 1,000 OBS-only updates, EMA reset | unchanged; new OBS weights retained |
| Seed / effective batch | 1729 / 8 | unchanged |

Fresh initialization, existing four-block latent processor, two 128-channel
45×90 slots, diffusion decoder and shared 8→32→3 ERA5 adapter. No replay or
latent penalty. Main LR 1e-4, linear cooldown to 1e-6, EMA half-life 100 updates,
gradient clipping at 1. Four GPUs each process two examples; activation
checkpointing off, fused AdamW, compiled training denoiser, eager evaluation.

**Selection is unchanged:** equal average of five RMSE/climatology-RMSE ratios
(SST, SSH-derived velocity, EKE, upper OHC, deep OHC), original nine validation
origins and eight-member/32-step decoding. There is no RMSE blend v2. Salinity
is supervised in training but not directly in selection. Save raw and EMA
validation every 250 updates, plus immutable 2k/4k/5k checkpoints. Compare both
selected and final models and all individual components; one seed is a screen,
not a replicated finding. Later processor/forcing/resolution stages remain
gated on improvement; they have not been submitted.

## Execution and qualification

- Producer: `a8af22e9ac315be64075f60bd2414bcc1c0346cb`.
- Beta production: `218258`, `standard`, one node / four GPUs, 24-hour limit.
- Short parallel scheduling candidate: `218259`, `beta_test` / `test`, four
  GPUs, two-hour limit. An exclusive filesystem lock prevents duplicate
  qualification; production qualifies inline if it starts first. A failed
  qualification stops production rather than silently retrying.
- Root: `/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-reweight-20261026`.
- Frozen data paths, normalization, forcing-cache proof and run configuration:
  `paths.json` at that root. The source checkpoint path is a provenance check;
  this fresh mode does not initialize weights from it.
- Qualification exercises both sources, exact optimizer/EMA/RNG serialization,
  rank agreement, finite evaluation and measured OBS loss equal to
  `0.5 * interior + 0.5 * surface + 0.1 * reconstruction`.
- Local validation: 35 targeted CPU tests passed; commit hooks passed.
- W&B configured online: `emulators/observational-transfer`; the URL will be
  recorded after initialization, not inferred while queued.
- Remote completion callback: `2a8b0dea-484f-44ed-baed-bd796be32acc`, with a
  dedicated quiet monitor and fallback reminder. Login-node authentication
  passed; each job also checks callback access inside its compute container.

At submission both jobs were pending without a start estimate. Expected runtime
is approximately eight hours after allocation, extrapolated from the completed
shared-ERA5 run; qualification and additional raw validation add uncertainty.
No completion ETA is asserted until scheduling evidence is available.
