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
is supervised in training but not directly in selection. Save full raw/EMA/optimizer checkpoints every 500 updates. Defer all training
validation until the end, distributing the ten checkpoints across four GPUs.
Each retains all nine origins, eight members and 32 diffusion steps for both
raw and EMA weights. Qualification still includes its small safety evaluation. Compare both
selected and final models and all individual components; one seed is a screen,
not a replicated finding. Later processor/forcing/resolution stages remain
gated on improvement; they have not been submitted.

## Execution and qualification

- Producer: `b4135aeee490ad0d1b993e3b14ede2f2b727afbf`.
- Beta production: `218264`, `standard`, one node / four GPUs, 24-hour limit.
- Short parallel scheduling candidate: `218265`, `beta_test` / `test`, four
  GPUs, two-hour limit. An exclusive filesystem lock prevents duplicate
  qualification; production qualifies inline if it starts first. A failed
  qualification stops production rather than silently retrying.
- Root: `/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-reweight-20261026/deferred-validation`.
- Frozen data paths, normalization, forcing-cache proof and run configuration:
  `paths.json` at that root. The source checkpoint path is a provenance check;
  this fresh mode does not initialize weights from it.
- Qualification exercises both sources, exact optimizer/EMA/RNG serialization,
  rank agreement, finite evaluation and measured OBS loss equal to
  `0.5 * interior + 0.5 * surface + 0.1 * reconstruction`.
- Local validation: 36 targeted CPU tests passed; commit hooks passed.
- W&B configured online: `emulators/observational-transfer`; the URL will be
  recorded after initialization, not inferred while queued.
- Remote completion callback: `4a1196cb-9469-4c94-a98c-4ee2403d7bb0`, with a
  dedicated quiet monitor and fallback reminder. Login-node authentication
  passed; each job also checks callback access inside its compute container.

At submission both revised jobs were pending without a start estimate. The
completed reference's matched update timing totals 5.67 hours; ten checkpoint
raw/EMA evaluations, distributed across four GPUs, should add roughly 24 minutes
instead of the previous 2.6 hours. Allow approximately 6–7 hours after allocation
including qualification/startup, subject to filesystem variation.

The initial attempt `218258` was canceled during qualification, before scientific
training began, when the user changed the validation schedule. Its 8m54s on four
GPUs (0.5933 allocated GPU-hours) and logs are retained in the parent directory;
its pending test `218259` never started. The replacement starts fresh and does
not promote qualification weights.

Selection now considers 500-update checkpoints. Compare the baseline on the
same 500-update candidate grid rather than crediting a difference in selection
frequency. Full component curves live in `validation-*.json`, checkpoint SHA and
per-pass timing receipts in `validation-receipt-*.json`, and final selection in
`EVALUATION_COMPLETE.json`. W&B uses `posthoc/checkpoint_step` for these curves
so resumed logging does not discard earlier optimizer-step measurements.

## Remaining data-preparation opportunity

The reader memory-maps a verified uncompressed float32 `frames.npy` cache at
`monthly-latent-pilots-20261016/packed-om4`: shape 2702×77×180×360, 53,927,596,928
bytes. It is not reading 77 separate Zarr arrays per sample. Rank zero nevertheless
serially gathers eight windows, checks finite values, masks and normalizes on
CPU, transfers to GPU, and only then broadcasts prepared examples. There is no
prefetch overlap. Eight 25–26-frame examples traverse about 4 GB of raw fields
per update, plus repeated memory passes/intermediates. Native OM4 forcing is
still read from Zarr before the shared ERA5 path replaces it.

Bounded prefetch, GPU normalization and avoiding the unused native-forcing read
are candidate improvements, not changes included in this producer. Preserve
normalization numerics, sample order and integrity checks when qualifying them.
