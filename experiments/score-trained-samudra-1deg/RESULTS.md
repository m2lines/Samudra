<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Score-trained Samudra: first-wave annual forecast results

**Status, 2026-10-12:** all training and all four primary annual evaluations completed successfully.
This report covers repeated forecasts. Continuous-rollout drift/structure diagnostics passed qualification
and are queued as Beta job 212420;
persistence, training-period climatology and observational comparisons are still outstanding. This is an
experiment/results record, not intended to merge. The first-wave report is not yet complete.

## What the completed comparison establishes

The stochastic U-Nets produce useful forecast distributions: at 360 nominal days, C/D reduce empirical
CRPS by about **31% versus plain MSE B**, and ensemble-mean RMSE by **5.8%**. Against the stronger reused
dynamic-MSE control A, the improvements are about **28% in CRPS and 2.4% in RMSE**.

There are three qualifications:

- **A still has the best aggregate RMSE at short and intermediate leads.** C/D overtake A at 180 and 360 days.
  This is not a uniform improvement at every lead or in every variable.
- **C/D remain underdispersed.** At 360 days, the target lies inside their 16-member range about 82.5% of the
  time, versus 88.24% for 16 exchangeable forecast members and a verifying target. Their finite-member-adjusted
  RMS spread/error ratios are 0.84 and 0.86, with 1 as the calibrated reference under the usual exchangeability
  assumptions. Positive spread alone did not establish this.
- **D has no clear advantage over C in aggregate RMSE/CRPS.** Their 360-day differences are smaller than the
  paired year-block intervals below. Whether D improves structure is a separate question for the pending
  diagnostics; a null aggregate comparison does not settle it.

These are OM4-to-OM4 forecasts initialized with complete ocean states, not surface-only initialization,
observational interior verification, or a demonstration that the model has learned the full joint ocean
distribution. “Score-trained” here means an almost-fair **CRPS scoring rule**, not a denoising-score diffusion
model. Sampling takes one stochastic U-Net call per two output states; it has no iterative diffusion decoder.

## Matched evaluation and controls

| Arm | Objective | Training | Primary evaluation |
| --- | --- | --- | --- |
| A | Dynamic MSE | Reused existing control; no retraining | Final epoch-70 EMA, deterministic |
| B | Plain MSE | New seed-15 run | Final epoch-70 EMA, deterministic |
| C | Marginal almost-fair CRPS | New seed-15 run; four training members | Final epoch-70 EMA, 16 independently evolving members |
| D | C plus block and depth/time difference scores | New seed-15 run; frozen training-only auxiliary coefficients | Final epoch-70 EMA, 16 independently evolving members |

The common recipe is 1° OM4, two input/two output states, four autoregressive calls per training context,
InstanceNorm, bfloat16, Rust loading, existing normalization and `tauuo/tauvo/hfds` forcing. No `hfds_anoms`
are used. Both stochastic models have the same backbone as B, plus 13,280 noise-projection parameters.

All arms are evaluated on **30 common origins**, from 2014-10-15 through 2021-12-09, at nominal leads
5/10/20/40/90/180/360 days. Each arm supplies 16,170 origin/lead/channel rows. There are 77 channels:
19 depths each of U, V, temperature and salinity, plus SSH. Models, masks, normalization, target timestamps
and saved target maps were checked for consistency. A's checkpoint loaded strictly into the common evaluator.

“Days” in the evaluator mean source steps multiplied by five. Two leap-year boundaries have six-day spacing;
72 forecast states consequently span 360 or 361 calendar days. Actual timestamps are retained in every raw
row. The common evaluation contract is in [annual-contract.json](results/annual-contract.json).

The headline score averages per-channel, per-origin wet-area scores, after normalizing each channel by its
existing standard deviation. It is **not a single globally pooled RMSE**, a volume-weighted score, or a
physical-unit average across unrelated quantities. U/V together account for 38 of the 77 equally weighted
channels, and SSH for one. Physical-unit breakdowns below help expose the resulting trade-offs.

A is a historical recipe-matched control, not a bitwise identical training execution. Its training source
and annual training-time validation differ from the experiment-local B/C/D trainer. B/C/D used the disclosed
two 90-day validation origins per epoch. These results use final EMA checkpoints and never select checkpoints
using held-out scores. Raw and validation-selected checkpoint sensitivities have not yet been evaluated.

## Forecast scores

![Common forecast scores and finite-member-adjusted spread/error](results/annual-scores.jpg)

Mean normalized channel RMSE, lower is better:

| Nominal lead | A: dynamic MSE | B: plain MSE | C: CRPS | D: structured CRPS |
| --- | ---: | ---: | ---: | ---: |
| 5 days | 0.13162 | 0.13889 | 0.13403 | 0.13370 |
| 40 days | 0.25003 | 0.26143 | 0.25846 | 0.25991 |
| 90 days | 0.29005 | 0.30273 | 0.29734 | 0.29756 |
| 180 days | 0.32510 | 0.33709 | 0.32423 | 0.32437 |
| 360 days | 0.34919 | 0.36159 | 0.34076 | 0.34080 |

At 360 nominal days:

| Metric | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Empirical CRPS, normalized | 0.21208 | 0.21942 | 0.15196 | 0.15186 |
| Fair CRPS, normalized | 0.21208 | 0.21942 | 0.14447 | 0.14410 |
| Mean member RMSE, normalized | 0.34919 | 0.36159 | 0.43266 | 0.43726 |
| Mean normalized spread | 0 | 0 | 0.27514 | 0.28256 |
| Target inside member range | — | — | 82.47% | 82.64% |
| Finite-member-adjusted RMS spread/error | — | — | 0.836 | 0.863 |

For deterministic predictions, empirical CRPS equals MAE. Fair CRPS removes the finite-ensemble bias from
the pairwise term; it is reported separately, not silently substituted for empirical CRPS. Individual stochastic
members have larger errors than the ensemble mean. That is compatible with a useful distribution and does
not mean every sampled trajectory should outperform a point forecast.

The spread/error ratio above is `sqrt(17/16) * sqrt(mean(spread²)) / sqrt(mean(mean_forecast_RMSE²))`, over
the same origin/channel rows. The correction accounts for sampling error in a 16-member ensemble mean.
The range reference is `15/17`. Interpolated 5th–95th percentile coverage is also retained in the raw tables,
but its finite-ensemble reference is not simply 90%, so that number is not the primary calibration conclusion.

![Weighted target rank histograms](results/annual-ranks.jpg)

Rank histograms give every channel and origin equal total weight, then use cosine-latitude wet-area weights
within each field. Exact ties are randomized with a separate RNG. The horizontal line is the uniform
17-bin reference. The excess at the ends is consistent with the range/spread evidence of underdispersion;
these histograms do not diagnose all spatial or temporal dependence errors.

## Physical-unit trade-offs

Mean physical RMSE across origins and the 19 depths within each 3D family, at 360 nominal days. Depths have
equal weight here; this is not a volume-weighted mean:

| Variable | Units | A | B | C | D |
| --- | --- | ---: | ---: | ---: | ---: |
| Temperature | °C | 0.32471 | 0.37922 | 0.32743 | 0.32643 |
| Salinity | OM4 salinity units | 0.07242 | 0.08767 | 0.08405 | 0.08297 |
| U velocity | m/s | 0.03274 | 0.03353 | 0.03185 | 0.03178 |
| V velocity | m/s | 0.03144 | 0.03198 | 0.03035 | 0.03046 |
| SSH | m | 0.04383 | 0.04990 | 0.04282 | 0.04296 |

For example, C improves salinity RMSE over B but remains about 16% worse than A on this physical-unit
depth average. A's temperature RMSE also remains slightly better than C/D. An aggregate win should not be
presented as an across-the-board interior-state improvement. The per-channel tables retain every depth:
[A](results/annual-channel-A.csv), [B](results/annual-channel-B.csv), [C](results/annual-channel-C.csv),
[D](results/annual-channel-D.csv), plus the [family summary](results/annual-family.csv).

## Paired uncertainty

Descriptive 95% intervals resample whole origin years (2014–2021), preserving paired arms and taking the
ratio of resampled score sums. There are 4,000 fixed-seed resamples. All channels from an origin stay together.
Partial years retain their actual numbers of origins. These intervals quantify variation across this set of
forecast years, **not training-seed uncertainty**. There are only eight blocks, and overlapping annual forecast
windows can remain dependent across adjacent blocks.

| 360-day comparison | Metric | Relative change | Year-block interval |
| --- | --- | ---: | ---: |
| C vs B | RMSE | −5.76% | [−6.59%, −4.96%] |
| C vs A | RMSE | −2.41% | [−3.17%, −1.71%] |
| C vs B | Empirical CRPS | −30.75% | [−31.09%, −30.42%] |
| C vs A | Empirical CRPS | −28.35% | [−28.64%, −28.09%] |
| D vs C | RMSE | +0.014% | [−0.189%, +0.179%] |
| D vs C | Empirical CRPS | −0.066% | [−0.221%, +0.062%] |

All leads and comparisons are in [annual-paired-year-intervals.csv](results/annual-paired-year-intervals.csv).

## Fixed-date fields

These maps all use the first prespecified initialization, 2014-10-15 12:00. The 360-day target is
2015-10-10 12:00. No dates or members were selected for favorable results. Member 0 is shown for C/D;
the original cluster NPZ files also retain member 1.

Each map panel is exactly 360 × 180 image pixels for the 360 × 180 data grid, with nearest-neighbor rendering.
The horizontal and vertical axes are grid indices; Gaussian latitude spacing is not reprojected. Full-field
panels share the full combined range across truth and predictions, with a separate spread scale. This can
hide small open-ocean variations when a small basin has an extreme value. The additional error maps use a
common symmetric 99th-percentile absolute-error scale; each panel states its fraction beyond that scale.
Both views are provided so range choices are explicit. Units are those in the table above.

![550 m salinity fields and spread](results/annual-so_9-360d.jpg)

![550 m salinity errors on a shared scale](results/annual-error-so_9-360d.jpg)

Other fixed-date assets:

- SSH: [fields](results/annual-zos-360d.jpg), [errors](results/annual-error-zos-360d.jpg).
- Surface temperature: [fields](results/annual-thetao_0-360d.jpg), [errors](results/annual-error-thetao_0-360d.jpg).
- 550 m U velocity: [fields](results/annual-uo_9-360d.jpg), [errors](results/annual-error-uo_9-360d.jpg).
- 550 m salinity at 40 nominal days: [fields](results/annual-so_9-040d.jpg).

## Remaining diagnostics and proposed decisions

The continuous-rollout runner initializes once and evolves through the held-out period without resets,
using four members for C/D. It measures member and mean errors, regional RMSE, physical spatial variance,
wet-pair zonal structure functions, temporal increments, adjacent-depth differences, T/S and U/V covariance,
and depth-weighted mean-state/kinetic-energy proxies. Nonlinear quantities are calculated separately on each
member and on the ensemble mean. The bulk quantities use cosine-latitude weights, binary masks and nominal
layer thicknesses; they are not exact OM4 heat/salt inventories with partial-cell geometry.

Still outstanding are that continuous analysis, persistence and training-only seasonal climatology references,
secondary observation scoring where applicable, and raw/validation-selected checkpoint sensitivities.
No conclusion about climate stability, observational skill or improved joint structure is claimed yet.

Provisional next-wave suggestions, for review after those diagnostics:

1. **Reproducibility:** repeat B/C at seeds 16/17. Include D if the joint diagnostics show a useful structural
   benefit; its aggregate scores alone do not currently justify preferring its extra loss terms.
2. **Underdispersion:** compare C's four-member training with an eight-member version at the same exposure,
   while reporting the additional compute. Add a validation-only spread-calibration diagnostic as a cheap
   reference, clearly separated from dynamically generated, physically coherent members.
3. **Variable trade-offs:** compare C against a version with fixed, training-derived channel weights motivated
   by A's temperature/salinity advantage. Check whether thermodynamic gains cost velocity/SSH skill instead
   of optimizing only the 77-channel headline.
4. **Structured multitask learning:** use C as the simpler provisional generative baseline. Add masking/tasks
   with a matched forecast-only control once calibration and structural behavior are understood. These
   results do not require changing to diffusion, nor do they establish that diffusion would be inferior.

No next-wave training has been submitted.

## Provenance and reproduction

The [execution record](RUNS.md) contains job IDs, immutable training/evaluator sources, paths, validation
deviations, qualification and recovery. Annual evaluation jobs are A 212403, B 212404, C 212406 and D 212407,
all using evaluator `0ed45a791` and epoch-70 EMA checkpoints. W&B links and exact checkpoint paths are in the
[evaluation contract](results/annual-contract.json). Full original manifests on Beta include checkpoint
SHA-256 and container fingerprints.

New B/C/D training consumed about **121.89 allocated B200 GPU-hours**. Including qualification, cancelled
notification preflights and all annual evaluations, approximately **124.5 GPU-hours** have been used before
the continuous diagnostics, within the approximately 200-GPU-hour first-wave envelope. These are allocation
costs, not kernel-active GPU time. A's historical training cost is excluded; it was reused. Equal epochs and
training contexts are not equal compute, and no equal-compute superiority claim is made.

[annual-summary.csv](results/annual-summary.csv) contains the plotted aggregate scores.
[annual-raw/](results/annual-raw/) retains all per-origin/channel/lead metrics, split by arm/lead into small
gzip CSV parts. Read every part once; the split is purely for repository file-size limits. Units and all
normalization values are retained, and no origin was dropped from aggregation. `analyze_score.py` validates
completeness, unique keys, finite scores, unit conversions, checkpoint epochs, common manifests, saved target
equality, rank counts and member diversity before generating any tables or figures.

Reproduce with `uv run python experiments/score-trained-samudra-1deg/analyze_score.py`, passing `--A`, `--B`,
`--C`, `--D` paths to the downloaded `outputs/eval-full-<arm>-<job>/` directories, `--data-manifest` pointing
to B's `outputs/train-B/data-manifest.json`, and `--output` for the generated assets. All remote paths are
relative to the campaign root documented in `RUNS.md`.
