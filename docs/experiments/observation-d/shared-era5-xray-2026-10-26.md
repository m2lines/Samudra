<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Shared ERA5 A: fixed-example X-ray export

The viewer adds shared-forcing A alongside original A, preserving the original
bundle. Four cases hold the January 2015 example fixed:

| Case | EMA checkpoint | Ocean history | Atmospheric input |
|---|---|---|---|
| OM4 · 4k | Main endpoint 4,000 | 19 OM4 SST/SSH frames | Matched ERA5 |
| OBS · 4k | Same endpoint | 19 observed SST/SSH frames | ERA5 |
| OM4 · 5k cooled | Selected 4k + 1k OBS cooldown, step 5,000 | Same OM4 history | Same matched ERA5 |
| OBS · 5k cooled | Same selected checkpoint | Same observation history | Same ERA5 |

For OM4, reconstruct all 26 five-day forcing intervals from daily ERA5 using
exact native OM4 interval bounds, the training strict-mean operation, and frozen
OBS atmospheric statistics. Record hashes of all 130 daily inputs. Both datasets
use the checkpoint's learned eight-to-three adapter; adapted channels are learned
coordinates, not physical wind stress or heat flux. No native OM4 forcing enters
these shared-forcing trajectories.

The bundle includes the original X-ray views: ocean history and availability,
raw/normalized ERA5 and adapted forcing, context, two latent slots, initialization,
one five-day processor step, and calendar-weighted monthly decoding. OM4 has all
77 physical reference channels; OBS surface and monthly supervision retain their
actual masks. OBS full-77 requests are explicitly diagnostic and use contemporaneous
OM4 context, not observational interior truth. Each trace retains all 63 calls of
the 32-step Heun sampler, plus final outputs and per-channel denoiser RMSE curves.
This remains one example and one draw, not aggregate skill evaluation.

## Verification and provenance

- Export producer: `2b1612ee5`; lightweight forcing/timestamp test passed, all
  repository commit checks passed.
- Beta job **218033**, four parallel case workers on four GPUs, completed exit 0
  in **6m23s**, **0.42556 GPU-hours**. No retraining.
- All **16 traces** passed bitwise equality between instrumented and untouched
  sampling; each curve contains 63 calls × 77 channels.
- Export root: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-xray-20261026/output`.
- `COMPLETE.json` records 1,065 SHA-256 file entries, **16,054,788,327 bytes**, case
  checkpoint hashes, source daily hashes, interval bounds, and configuration hash.
  The launcher read back and verified every child file before assembling the bundle.
- Source main checkpoint: `A-shared-era5-8k/endpoint-04000.pt`; selected cooled
  checkpoint: `A-shared-era5-cool4k/best.pt`, both beneath the shared-ERA5 DDP
  `eager-evaluation/production` directory. Raw model state is not used for this view.
- Decoder evaluation is eager, matching the qualified evaluation path. Compilation
  is a training optimization and is not activated during this export.

The viewer is deployed at `55bdf307d` on `codex/observation-viewer`, with the
**A · shared ERA5** model selector entry. All 1,065 transferred files passed local
SHA-256 read-back. Viewer CI passed; the local suite passed 37 tests (13 optional
integration tests skipped). A live browser check separately exercised all four
cases, diffusion RMSE sparklines, monthly fields, shared ERA5 forcing panels,
latents, and switching back to Original A. The service is active. The direct
query is `?xray=1&xray_model=shared-era5`.

## Eight-member initialized velocity comparison

A follow-up fixed-example export averages eight independent 32-step eager decoder
samples for three initialized states: shared-ERA5 cooled A with OM4 inputs,
shared-ERA5 cooled A with OBS inputs, and original cooled A with OBS inputs.
These use identical member seeds 905001–905008 across cases. Member zero exactly
reproduced each original saved single draw across all 77 output channels. The
saved bfloat16 initializer latent was reconstructed exactly from its float32
export; no initializer or processor was retrained or rerun.

Beta **218165** completed exit 0 in **1m46s / 0.11778 GPU-hours**, distributing
24 decodes across four GPUs. Producer `349e40975` passed its provenance test and
all repository commit checks. The exported means and population standard
deviations contain U/V at 2.5 m and 250 m, converted to m/s using the original
OM4 mean/std. Gold is bitwise identical across cases after physical conversion.
OBS full-77 output remains diagnostic; its velocity reference is OM4 context,
not observed velocity truth. This comparison isolates sampling variability for
these fixed latents and does not measure aggregate forecast skill.

The small output `velocity-means.npz` and `COMPLETE.json` are retained under
`beta:/projects/ny/lz1955/multiscale/jrusak/runs/xray-velocity-ensemble-20261026`.
The manifest records all 24 member checksums and the final output checksum.

The initializer comparison figure uses eight-member means in row order `uo_0`,
`uo_7`, `vo_0`, `vo_7`, with per-row color limits set by the OM4 reference 2nd–98th percentiles,
shared across all columns, and the diverging color midpoint fixed at zero.
Columns are OM4 reference, shared-ERA5 OM4 initialization, shared-ERA5 OBS
initialization, and original A OBS initialization. Training exposure differs:
4k + 1k cooldown versus 2k + 1k; this is not a matched forcing ablation.

[Initializer mean figure (PNG)](figures/shared-era5-xray/velocity-initial-mean8.png) ·
[PDF](figures/shared-era5-xray/velocity-initial-mean8.pdf).
