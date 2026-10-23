<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Original A: initializer, processor and diffusion anatomy

This diagnostic holds the January 2015 example fixed and exposes three cases:

| Case | Checkpoint | Initial ocean inputs | Prescribed forcing |
|---|---|---|---|
| A / OM4 before cooldown | Original A, 2k main-training EMA | 19 five-day OM4 SST/SSH frames | OM4 tauuo, tauvo, hfds |
| A / OBS before cooldown | The identical 2k EMA | 19 five-day observed SST/SSH frames | Eight ERA5 fields through the learned adapter |
| A / OBS after cooldown | Original A, 3k EMA, including 1k OBS-only cooldown | Identical observation history | Identical ERA5 inputs, adapted by this checkpoint |

These are **original A**, not A-replay. The first comparison holds weights
fixed while changing data source; the second holds data fixed while changing
cooldown exposure. One held-out example and one decoder draw cannot establish
general skill or a causal explanation for an aggregate metric.

## What the app shows

The grid retains the same variable/depth ordering across cases: 19 layers each
of U, V, temperature and salinity, plus SSH. Every available field has prediction,
reference and error maps. Missing OBS labels remain explicitly blank.

- **Inputs:** all 19 historical frames, physical availability masks, forcing and
  geographic/seasonal context. Only SST/SSH enter the initializer as ocean fields.
- **Initialized decode:** the last historical five-day interval, December 27–January 1.
- **One processor step:** January 1–6. Only the latent state evolves; sampled
  physical outputs and future gold never feed the processor.
- **Monthly decode:** January's calendar-weighted latent average decoded directly
  against the monthly target. This requires additional latent steps and is shown
  separately from the single-step view. There is no top-temperature replacement
  in these raw monthly diagnostics.
- **Diffusion:** one fixed draw; final sample, noisy input or estimated clean
  output at any of 32 Heun intervals. All 63 predictor/corrector denoiser calls
  are retained. Each instrumented final output must equal a rerun of the untouched
  sampler bit for bit. These are sampling diagnostics; training uses individual
  noisy-target denoising calls, not differentiation through this full sampler.
- **Latents:** both memory slots, selectable channel, initialization and first
  processor output, differences and RMS. Latents have no physical-unit conversion.

Physical versus normalized display uses the same OM4 mean/std for all ocean
fields and all three cases. Differences are divided/multiplied by std without
adding a mean. Map color limits pool the final predictions and references across
cases and remain fixed during source/step switching. Early high-noise states can
exceed those ranges; clipped percentages are explicit. Maps retain native grid
pixels; Gaussian latitude spacing is not uniform in degrees. Map RMSE is an
unweighted spatial diagnostic, not the official pooled validation score.

## Gold and supervision

OM4 initialization and forecast have full 77-channel gold, including interiors
hidden from the initializer. Native OBS decoding requests only SST/SSH; monthly
OBS decoding requests IAP T/S at 14 depths, excluding top temperature (27 active
channels). Full-77 OBS decoding is also available as an explicitly separate
**diagnostic request** with contemporaneous OM4 simulation context. That context
is not observational truth and was not an OBS training target.

All model-data tasks use the same losses: 0.5 full-field monthly denoising +
0.5 full-field native forecast denoising + 0.1 full-field initial reconstruction.
OBS loss is 0.8 monthly interior denoising + 0.2 native surface denoising +
0.1 initial surface reconstruction. Original A has no variable/spatial masking
multitask treatment beyond surface-only initialization, no replay, and no latent
norm penalty. See [the pilot methods](monthly-latent-pilots-2026-10-16.md) and
[the pretraining methods](monthly-model-pretraining-2026-10-20.md#model-objective-and-sources).

## What this example shows

These are **single-draw, unweighted spatial RMSEs**, not eight-member means or
pooled validation metrics. Surface comparisons use their respective source's
truth and decoder task; they are not a controlled ranking across data sources.

| Case | Initial SST (°C) | Day-5 SST (°C) | Initial SSH (m) | Day-5 SSH (m) |
|---|---:|---:|---:|---:|
| A 2k, OM4 full77 | 7.562 | 7.503 | 0.427 | 0.424 |
| A 2k, OBS surface-only | 0.805 | 0.920 | 0.082 | 0.090 |
| A 3k cooled, OBS surface-only | 0.574 | 0.722 | 0.069 | 0.076 |

The noisy full-field OM4 output is already present in the initial decode;
it is not created solely by the first processor step. Interior-temperature
RMSE (all wet cells in the 18 subsurface layers, equally weighted per cell)
is 3.586°C initially and 3.532°C after one step. This does not separate poor
latent initialization from poor conditional decoding; inspect both the latent
changes and the denoising trajectory before assigning a cause.

For the same OBS example, cooldown improves native surface reconstruction and
forecasting. Monthly subsurface-temperature RMSE against IAP declines from
2.034°C to 0.897°C; monthly salinity RMSE from 0.487 to 0.318 in source salinity
units. Those monthly metrics cover the available IAP layers, unlike the deeper
OM4 comparison. This example is consistent with better OBS-task readout after
cooldown, but does not establish its generalization or full-state accuracy.

## Export and verification

Beta **216421** completed in 13m27s, **0.224167 GPU-hours**, with no retries.
Producer `6735615ac`; no training or model selection was performed. The exporter
is [`export_monthly_step_anatomy.py`](../../../scripts/export_monthly_step_anatomy.py).
All 851 data files (11.54 GB) were downloaded and SHA256-verified, then verified
again after installing the viewer bundle. All 13 traces have 32 intervals,
63 denoiser calls and bitwise agreement with the uninstrumented sampler.

The UI is published on `codex/observation-viewer`, commit `ca734ca`.
Use the **A: initialization & diffusion anatomy** link in the existing viewer,
or add `?anatomy=1` to its app URL. Thirty-two lightweight tests pass; a real-data
browser test exercises all three cases, normalized units, forecast and monthly
views, diffusion clean estimates and full77 context. The installed service is
verified separately after deployment. Arrays stay outside Git; source checkpoint
hashes, exact normalization and latent fields are retained in the bundle manifest.

## Follow-up: denoising failure localization

The [matched corrupted-gold diagnostic](om4-denoiser-diagnostic-2026-10-23.md)
compares the supervised OM4 sampler trajectory with training-style corrupted
truth across all 32 predictor noise levels, holding the saved latent fixed.
It includes per-variable curves and distinguishes mid-noise denoising weakness
from low-noise persistence of sampling errors.
