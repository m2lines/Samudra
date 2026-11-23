<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longer six-hour training and observation-only cooldown

The mixed LLC + observations model improved after longer training, but the
observations-only control still has lower surface forecast error at 24–72 hours
at the same optimizer budget. After 10,000 main updates and 1,000 observation-only cooldown
updates, aggregate normalized RMSE is 0.023007 for mixed training and 0.019358
for the control. Mixed is 18.9% worse than the control and 9.3% better than
persistence (0.025361). Lower RMSE is better. The original mixed EMA at update
2,000 scored 0.038645; its OBS-only control scored 0.026248 on this same cohort.

LLC denotes coarsened LLC4320 ocean-simulation snapshots. Observations (OBS)
are OISST sea-surface temperature (SST) and DUACS sea-surface height (SSH).
Both arms continue their original 2,000-update checkpoints, preserving raw
model weights, optimizer, exponential-moving-average (EMA) weights, source
exposure, sampler and random states. These are longer versions of the same
seed, not independent replicates. The [continuation protocol](../long-run.md)
specifies checkpoint migration and qualification.

Both arms use the same deterministic surface-initialized U-Net model on a global
1-degree Gaussian grid (180 × 360). Source-specific input adapters feed shared
initializer and evolution networks. The output is U/V velocity, temperature and
salinity at 51 native LLC depths plus SSH (205 channels), including in the OBS-only
control; that control has no direct interior targets. Ocean input history contains
only surface temperature and SSH, alongside ERA5 atmosphere inputs: 29 frames
six hours apart span one week through the forecast origin t. Daily observations
occupy their midnight analysis slots; other six-hour surface slots remain masked.
The initializer infers states at t−6 h and t, then dynamics advances every six
hours. Daily scores compare individual forecast states with daily analyses,
not daily-mean predictions.

| Fixed EMA checkpoint | Mixed normalized RMSE | OBS-only normalized RMSE | Mixed relative to control |
| --- | ---: | ---: | ---: |
| 10,000, before cooldown | 0.024636 | 0.019670 | +25.2% |
| 11,000, after cooldown | 0.023007 | 0.019358 | +18.9% |

At update 11,000, mixed training has 40,000 LLC and 48,000 OBS window draws;
the control has 88,000 OBS draws. The secondary exposure comparison uses the
control's update-6,000 EMA checkpoint, with 48,000 OBS draws and normalized
RMSE 0.025393. Mixed's final score is 9.4% lower. That control has fewer total
optimizer updates and no cooldown; its optimizer and EMA histories differ.
This result does not isolate the benefit of LLC supervision.

The [training-curve stability analysis](training-curve-stability.md) shows all
110 validation checkpoints per arm, transient error spikes, late-training
consistency, and paired validation-date sensitivity.

![Validation normalized RMSE over all 11,000 updates and a late-training zoom](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-overview.png)

Both arms have transient error spikes earlier in training. The full history uses
a logarithmic error axis; the late-training zoom uses a linear axis. Shading
marks the observation-only cooldown. These are validation RMSE curves with no
additional smoothing; lower is better. The [per-field curves and analysis](training-curve-stability.md)
separate SST and SSH at each lead and quantify sensitivity to validation dates.

## Daily observation forecasts

| Lead | SST mixed / OBS-only / persistence (°C) | SSH mixed / OBS-only / persistence (m) |
| --- | --- | --- |
| 24 h | 0.2302 / 0.2077 / 0.2197 | 0.006867 / 0.004122 / 0.005885 |
| 48 h | 0.3119 / 0.2966 / 0.3458 | 0.010238 / 0.006657 / 0.011623 |
| 72 h | 0.3560 / 0.3381 / 0.4265 | 0.014433 / 0.010238 / 0.017129 |

These are fixed update-11,000 EMA scores. Mixed improves over persistence for
both variables at 48 and 72 hours, but not at 24 hours. The OBS-only control
improves over persistence at all six variable/lead combinations. Mixed remains
worse than the control at every combination.

The unchanged cohort has eight non-overlapping forecast origins, all at midnight:
2013-11-09, 2013-12-16, 2014-01-22, 2014-02-28, 2014-04-07, 2014-05-14,
2014-06-20 and 2014-07-28. These are the same validation origins used throughout
checkpoint selection. OBS training covers 1993-05 through 2013-07, excluding the
LLC validation interval in late 2012; complete histories and forecasts must stay
within their split. Per-channel squared errors are wet-area averaged on
the intersection of origin and target validity masks, then equally averaged
across origins before taking square roots. Both models and persistence use the
same support. Persistence repeats the analysis at the forecast origin, also
supplied as the model's latest history frame. The aggregate averages normalized
RMSE equally over SST/SSH and 24/48/72-hour leads. Training-support standard
deviations are approximately 10.0999°C and 0.641986 m, respectively.

The best validation-selected mixed EMA occurs at update 11,000 (0.023007).
The control's best occurs at 10,900 (0.019343); the fixed 11,000 comparison above
does not substitute that selected checkpoint. Selection spans updates 2,100
through 11,000. One seed and eight origins give a descriptive comparison;
the linked date-resampling analysis does not estimate uncertainty across training
seeds or provide independent test-period results.

## Cooldown and LLC diagnostics

The main phase uses constant LR 1e-4. The final 1,000 updates use observations
only, linear LR decay to 1e-6, and an EMA reset at update 10,000. Mixed observation
normalized RMSE decreases by 6.6% across this interval. Its LLC aggregate
normalized RMSE increases by 11.1%, from 0.368204 to 0.408950. Before/after
changes combine additional training, source selection, LR decay and EMA reset;
these runs do not separate their effects.

LLC full-state persistence scores 0.238923. It repeats all 205 true initial
channels, while the model infers its state from surface temperature and SSH
history. The baseline therefore has more initial information for interior fields.
LLC scores use eight fixed complete validation windows in the 2012-09-14 through
2012-11-15 holdout, depth-specific wet area, and training-channel normalization.
Surface temperature and SSH use the shared observation-training normalization;
other channels use the LLC store's training statistics. The aggregate equally
averages normalized RMSE across all channels and daily leads, so its 204
depth-resolved channels dominate SSH. The training loss instead gives equal
weight to the U, V, temperature, salinity and SSH groups.

The [paired training-date audit](../qualification/surface-audit.json) found mean
LLC-minus-OBS SSH of −0.243 m on shared valid support. Raw LLC Eta (model sea-surface
height) and observational ADT (absolute dynamic topography) retain their original
references in these runs. The measured difference does not establish how much comes from vertical datum
conventions versus simulation-observation differences; its cause remains unresolved. Future ERA5 weather
is prescribed, so the forecast experiments are hindcasts.

## Saved 30-day diagnostics and viewer

The 30-day forecasts show a different ordering from the three-day training task.
At day 30, mixed has lower error than OBS-only for both observed variables, but
neither final model beats persistence. The mixed temperature score is close to
persistence; SSH errors are several times larger in both models.

| Day-30 EMA checkpoint | SST mixed / OBS-only / persistence (°C) | SSH mixed / OBS-only / persistence (m) |
| --- | --- | --- |
| 10,000, before cooldown | 1.2517 / 2.5577 / 1.2598 | 0.34859 / 0.46468 / 0.07738 |
| 11,000, after cooldown | 1.2920 / 3.4592 / 1.2598 | 0.35573 / 0.48041 / 0.07738 |

These forecasts run continuously for 120 six-hour steps from the same eight OBS
origins, without reinitialization or assimilation. Errors use the same per-origin
wet-area weighting and origin/target validity intersection as the daily scores
above, against actual daily observations. The
[30-day curves](long-30day-observation-validation.csv) contain all four checkpoints,
30 daily leads and both variables. Recomputing their first three days from the
exported physical arrays reproduced all 24 corresponding validation RMSEs saved during training within
rtol 2e-4 and atol 2e-7. Later leads are new diagnostics beyond the trained
three-day horizon. The last origin extends into August 2014, using a separate
evaluation-only forcing/observation preparation; training data and normalization
were unchanged.

All exported six-hour forecast states remained finite on wet support. Finiteness
does not establish realistic temporal behavior: for example, mixed EMA11,000 SST
at 30.415°N, 329.5°E in the 2013-11-09 OBS case has mean within-day range 1.27°C
over days 1–3 and 3.75°C over days 28–30. The viewer exposes this oscillation in
the six-hour series. Daily observations do not validate the intervening subdaily
variation. The day-30 skill ordering and these behaviors need broader cohorts
and independent seeds before drawing a transfer conclusion.

The Samudra Results Viewer now serves the separate `/llc` page (deployment link
shared in the accompanying experiment conversation) with four model
choices (both arms at EMA10,000 and EMA11,000), 16 source/origin cases each, and
all 51 depths. Each case includes two inferred initial states, the 29-frame
surface input history, six-hourly surface predictions through day 30, and full
205-channel snapshots at hours 6, 12, 24, 48, 72, 120, 240, 360, 480, 600 and 720.
Actual observed references are available at daily slots. LLC fields are matched
in time where available; other years use explicitly dated seasonal context.
Seasonal context is excluded from error maps and scores. OBS-only's LLC task
adapter received no LLC training, which the page states.

Export jobs `234702` (mixed, 6 min 56 s) and `234703` (control, 6 min 28 s)
completed with exit 0. Each checkpoint reproduced 2,484 saved three-day validation
metrics across both sources and all channels. Remote CPU assembly `210616`
verified all checksums and cohorts and deduplicated references by hard links.
Direct SSH transfer copied 61.50 GB at 62.4 MB/s; a full local reread reproduced
the remote catalog hash before atomic publication. The existing viewer data
remains intact.

Live browser checks passed model/case/depth selection, initial states, input
history, three-day and 30-day controls, actual context dates, blank context error
maps, observed references, shared selections and WebSocket interaction, with no
JavaScript errors. Viewer code is `1411c56097948cedfcc4ca25c3c8c9946f97ba2f`;
export code is `ca764b2049cfdfc7f8ecbc6f2b321cb6e232795f`.
Catalog SHA-256 is
`26846f038bb83d0bca61035ee076acf4e27f92d1517b28f163c25a4f74d6e335`.
The 30-day export bundle is retained on EAI at
`/projects/ny/lz1955/multiscale/jrusak/runs/llc-viewer-evaluation-20261120/bundle`
and locally at
`/home/fomo-bot/samudra_viewer/apps/observation_viewer/.data-llc-20261120`.
Live screenshots are in the working checkout's `agent-logs/llc-viewer-live/`
and are shared with the result notification. These data are excluded from git.

## Execution and reproducibility

Mixed continuation job `234626` completed with exit 0 in 11 h 45 min 50 s;
control job `234629` completed with exit 0 in 6 h 15 min 51 s. Both used four
GB200 GPUs under standard EAI QoS, global batch eight, and online W&B:
[mixed l2d2u8sw](https://wandb.ai/ocean_emulators/observational-transfer/runs/l2d2u8sw)
and [control 31laq271](https://wandb.ai/ocean_emulators/observational-transfer/runs/31laq271).
Each continuation has 9,000 new sequential updates and all 90 planned validations
from update 2,100 through 11,000. Final receipts show 5,000 LLC / 6,000 OBS updates
for mixed and 11,000 OBS / zero LLC updates for the control, with no replay.
Window draws can repeat dates.

Runtime is `3416705c51f521c773318b9798db14aa5a549f5f`, archive SHA-256
`2b73b7884d94932bc1cdd17ffae4d335fe357ab006674e43a5f159ee3f3fad21`.
Prepared training-data proof SHA-256 remains
`3fb27c8fe8d926e82b2aa2758aac9bd1100a20d86ac4baa44733de037895cb7e`.
Mixed full-metrics SHA-256 is
`1137f2bdfe9b72b8f80170387358d65606f9acc166d3a51ff6a8197d95f1c445`;
control full-metrics SHA-256 is
`7c24c46e43843375d2458c1a7057296923f5263843c28cbae74be06b59bdbe2b`.

Machine-readable results: [mixed surface curves](mixed-long-observation-validation.csv),
[control surface curves](obs-only-long-validation.csv),
[mixed aggregate curves](mixed-long-validation-aggregates.csv),
[fixed-checkpoint comparisons](mixed-long-versus-obs-only.csv), and
[final LLC field/depth/lead metrics](mixed-long-llc-final-validation.csv).
The control-at-6,000 columns in the comparison CSV match OBS exposure only for
mixed at 11,000; the 10,000 row is included for reference.

Run roots on EAI are
`/projects/ny/lz1955/multiscale/jrusak/runs/llc-obs-six-hour-long-20261120`
and `obs-only-six-hour-long-20261120` under the same parent directory.
Their `outputs/llc-obs-six-hour-long/saved_nets/` and
`outputs/obs-only-six-hour-long/saved_nets/` directories retain
`main-10000.pt` and `endpoint.pt` (11,000); use the EMA state for these results.
Lineage receipts retain parent and child manifests and the permitted changes.
Checkpoints and exported arrays are excluded from git.
