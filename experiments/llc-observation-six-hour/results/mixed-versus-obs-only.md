<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Initial LLC + observations result

The mixed model completed 2,000 optimizer updates, alternating coarsened LLC4320
simulation snapshots and observations (OBS): OISST sea-surface temperature (SST)
and DUACS sea-surface height (SSH). On the fixed
observation validation cohort, its final exponential-moving-average (EMA) weights
have higher error than the OBS-only baseline at every reported variable and lead.
The aggregate normalized RMSE is 0.038645 for mixed training, 0.026248 for OBS-only
training, and 0.025361 for persistence. Lower is better. Mixed training is 47.2%
worse than OBS-only training at the same optimizer budget in this initial run.

Both models use the same 205-channel architecture, initialization seed 1729,
29 six-hour history frames, six-hour dynamics, and daily targets over a three-day
forecast. They share ERA5 inputs, normalization, holdouts and validation origins.
The state has velocity, temperature and salinity at 51 LLC depths
(204 channels), plus SSH. Source-specific input adapters feed shared initializer
and evolution networks. The mixed run saw 8,000 LLC and 8,000 OBS window draws; the OBS-only run saw 16,000
OBS window draws. Draws can repeat dates. Thus equal optimizer budgets give the
OBS-only model twice the observation exposure. The
[experiment design](../README.md) and [OBS-only record](obs-only.md) specify the
remaining settings.

| Lead | SST mixed / OBS-only / persistence (°C) | SSH mixed / OBS-only / persistence (m) |
| --- | --- | --- |
| 24 h | 0.3392 / 0.2603 / 0.2197 | 0.018874 / 0.009781 / 0.005885 |
| 48 h | 0.4039 / 0.3325 / 0.3458 | 0.025005 / 0.012933 / 0.011623 |
| 72 h | 0.4460 / 0.3707 / 0.4265 | 0.029398 / 0.017144 / 0.017129 |

The secondary comparison matches observation exposure: mixed at update 2,000
against OBS-only at update 1,000, each with 8,000 OBS draws. Mixed aggregate
normalized RMSE is 10.2% higher (0.038645 versus 0.035059), with higher error at
all six variable/lead combinations. This comparison has different total optimizer
budgets and EMA weight histories. It does not isolate a mechanism for the gap.
The [comparison CSV](mixed-versus-obs-only.csv) includes both controls.

These are fixed-checkpoint comparisons using EMA weights, not a choice between
best checkpoints. The validation-selected best EMA also occurs at update 2,000
in each run. The cohort contains eight fixed, non-overlapping observation forecast
origins: 2013-11-09, 2013-12-16, 2014-01-22, 2014-02-28, 2014-04-07,
2014-05-14, 2014-06-20 and 2014-07-28. For each channel and lead, squared errors
are wet-area averaged on the intersection of origin and target validity masks
for that lead, averaged equally across origins, then square-rooted. Both models
and persistence use this same support. Persistence repeats the daily surface
analysis at the forecast origin, also supplied as the model's latest history frame. Channel standard deviations fitted on observation-training
support (10.0999°C for SST and 0.641986 m for SSH) normalize the RMSEs; the
aggregate averages them equally across two channels and three leads.

One seed and eight origins support a descriptive initial comparison. No uncertainty
intervals, test-period scores or long-rollout results are available. Future ERA5
weather is prescribed, so this is a hindcast experiment. The
[paired training-date audit](../qualification/surface-audit.json) found a mean LLC
minus OBS SSH difference of −0.243 m on shared valid support. The first run retains
raw LLC Eta (model sea-surface height) and observational ADT (absolute dynamic
topography) without a datum correction. Their vertical reference levels need
not coincide, and this audit does not isolate a reference offset from other
simulation-observation differences. This mismatch,
source-dependent sampling and optimization remain possible explanations; these
runs do not identify the cause of the observation error gap.

## LLC validation

The mixed model's LLC aggregate normalized RMSE is 0.338275 versus 0.238923 for
full-state persistence. Full-state persistence repeats all 205 true initial
channels, whereas the model initializes from surface temperature and SSH history.
That baseline has more initial information for interior fields. The aggregate
averages all channel/lead normalized RMSEs equally, so the 204 depth-resolved channels
dominate its single SSH channel; this diagnostic differs from the equal-variable-
group training loss. That loss gives equal weight to eastward velocity, northward
velocity, temperature, salinity and SSH, averaging depths within each group.

LLC validation uses eight evenly spaced complete windows in the held-out
2012-09-14 through 2012-11-15 interval. The one-week histories and three-day
forecasts stay within that interval. Per-channel errors are averaged over depth-
specific wet area, then equally across origins before taking square roots.
LLC normalization uses embedded statistics from 100 training-period snapshots,
except surface temperature and SSH, which use the shared observation-training
statistics stated above.

| LLC field | Model normalized RMSE | Full-state persistence normalized RMSE |
| --- | ---: | ---: |
| Eastward velocity | 0.551174 | 0.405746 |
| Northward velocity | 0.521401 | 0.494864 |
| Temperature | 0.126862 | 0.025042 |
| Salinity | 0.155334 | 0.023198 |
| SSH | 0.253122 | 0.587928 |

Each row averages normalized RMSE equally across its depth channels and
24/48/72-hour leads. The model improves LLC SSH over persistence, while the
depth-resolved group averages are worse. The
[final LLC metrics](mixed-llc-final-validation.csv) retain physical RMSE by depth,
field and lead. OBS-only's LLC adapter received no training, so its LLC diagnostic
is excluded from this comparison.

## Execution and artifacts

EAI Beta job `234501` completed with exit 0 in 2 h 46 min 13 s, including inline
qualification, using all four GB200 GPUs under standard QoS. Its qualification
ran four updates, resumed for four more, and checked daily validation before
production started from scratch in separate outputs. The production receipt
records 1,000 LLC updates and 1,000 OBS updates, 8,000 sampled windows per source, without
replay. All 20 planned validations are present. Redundant pending qualification
job `234502` was cancelled before allocating GPUs.

[Mixed W&B run 2a1lxyp9](https://wandb.ai/ocean_emulators/observational-transfer/runs/2a1lxyp9)
and [OBS-only W&B run 0oci13x4](https://wandb.ai/ocean_emulators/observational-transfer/runs/0oci13x4)
contain training histories. Machine-readable artifacts include
[mixed observation curves](mixed-observation-validation.csv),
[mixed aggregate curves](mixed-validation-aggregates.csv), and
[OBS-only observation curves](obs-only-validation.csv).

Both arms ran code `83df2678f91c29aba4187e79fdc9e8e46bfe0d54`, archive SHA-256
`a9402e3edf174b876358db7736a984e46672088e76417f0f7b9114d246aaed71`, with prepared-data
proof SHA-256 `3fb27c8fe8d926e82b2aa2758aac9bd1100a20d86ac4baa44733de037895cb7e`.
Mixed full-metrics SHA-256 is
`49df1c86bfa3a0208255b1328525ba5859268819727702317436c6f37b315368`.

Mixed receipts, logs and checkpoints remain on EAI under
`/projects/ny/lz1955/multiscale/jrusak/runs/llc-obs-six-hour-20261120`.
The final checkpoint is `outputs/llc-obs-six-hour-v1/saved_nets/main-2000.pt`;
use its EMA state to reproduce these scores. Checkpoints are excluded from git.
