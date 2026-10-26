<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Physical-state diffusion validation

Evaluation of final step 1200 (600 OM4 and 600 observation updates), using EMA,
OM4 normalization, shared eight-channel ERA5 forcing, eight sampling levels and
eight coherent ensemble trajectories. Each member feeds its own physical states
forward; no future ocean observations are supplied.

Beta job 218015 started on 2026-10-26 on four GB200 GPUs with standard QoS.
Outputs: `/projects/ny/lz1955/multiscale/jrusak/runs/physical-state-validation-20261026`.
Completed successfully in 13m38s. All 52 artifact checksums verified after download (6.64 GB). Training loss is not a validation score.

The four workers cover nine monthly validation cases and the annual validation
origin 2013-11-01, plus reporting-only annual test origins 2015-01-01, 2018-01-01,
and 2021-01-01. Existing frozen point metrics, calibration and annual rollout
metrics are retained. The legacy selection score is not the later
four-component climatology-normalized blended RMSE.

Full-state exports include all 77 physical channels at initialization and days
30 and 365, with ensemble mean, member 1 and spread. Annual temporal diagnostics
retain all eight members for SSH, SST and selected interior channels. Interior
monthly observations must not be presented as instantaneous reference states.

A separate `app?xray=physical` page is prepared in the viewer feature branch
`codex/viewer-physical-diffusion`. Deployment awaits verified outputs and browser
checks. This page inspects saved outputs, not internal denoiser activations.

## Results

The nine-origin monthly validation selection score is **1.43158** (legacy
control-normalized RMSE/spectral blend; lower is better). Day-30 SST RMSE across
these nine cases is **0.8634°C**. These are distinct from the single annual
validation origin below. Scores use the eight-member ensemble mean; the viewer
also exposes member 1 and spread.

| Origin | Split | SST RMSE day 30 / 365 (°C) | SSH RMSE day 30 / 365 (cm) |
|---|---|---:|---:|
| 2013-11-01 | Validation | 0.758 / 0.781 | 8.98 / 9.71 |
| 2015-01-01 | Test | 0.876 / 0.835 | 9.21 / 10.60 |
| 2018-01-01 | Test | 0.883 / 0.786 | 9.76 / 10.28 |
| 2021-01-01 | Test | 0.845 / 0.822 | 10.62 / 11.32 |

All sampled trajectories passed finite-value checks. The annual surface RMSEs
show no explosive growth in these four cases. This does not establish realistic
long-term dynamics: spectral quality is weak, and low ensemble-mean error can
hide overly smooth fields or poor individual members. The checkpoint is an early
600-observation/600-OM4 pilot, not a matched comparison with the earlier 8k/8k
models. Atmospheric forcing is prescribed ERA5, not an operational forecast.

Artifacts are retained locally under `outputs/physical-state-validation-20261026`
and at the Beta path above. `VERIFIED_SUMMARY.json` contains reduced metrics;
per-worker `input.json` and `COMPLETE.json` record provenance and content hashes.

## Individual-member temporal behavior

The annual validation members still fluctuate excessively. Mean member five-day
SST increment RMS is approximately **2.69°C**, versus **0.414°C** for observations;
SSH is **0.172 m**, versus **0.0260 m**. Within-year cell-anomaly lag-one
correlation is approximately **0.48** for member SST versus **0.977** for the
reference (SSH approximately **0.15** versus **0.930**). These diagnostics are
cosine-area pooled and computed from the saved five-day trajectories. They show
that bounded ensemble-mean RMSE does not imply realistic individual evolution,
even though the implementation propagates each member's own sampled state.
