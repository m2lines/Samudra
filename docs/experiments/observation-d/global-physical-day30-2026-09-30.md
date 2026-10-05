<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global physical-only: initialization, day-30 structure and annual heat content

**OM4 mixing helps at equal observation exposure, but twice as many observation updates closes the monthly score gap.** At day 30, mixed training reaches SST/SSH RMSE of 0.640°C/0.0792 m; obs-only reaches 0.694°C/0.0886 m at 8k and 0.572°C/0.0826 m at 16k. Mixed retains better day-30 SSH spectra and visibly less pronounced annual SST ripples. Learned initializer T/S fields are very similar at the endpoints. These results suggest a benefit to forecast structure, rather than a large unique advantage in recovering the initial interior state.

This report compares the **final global physical-only mixed model** with a **matched observation-only run at 8k and 16k**. The day-30 spectra and surface-error diagnostics remain separate from the added **day-365-only spectra** below; no spectral curve blends rollout leads. The established combined benchmark is separately labeled because it also includes other leads and monthly OHC. Annual global means, OHC totals in ZJ and maps remain diagnostic. None of the three endpoints beats its own inferred-initial-state persistence on that combined benchmark.

## Which checkpoints are being compared?

| Literal model label | Total optimizer updates | OM4 updates | Observation updates | Meaning |
|---|---:|---:|---:|---|
| Final: 8,000 obs | 16,000 | 8,000 | 8,000 | Mixed trajectory, fixed endpoint |
| Obs-only: 8,000 obs | 8,000 | 0 | 8,000 | Fresh observation-only trajectory; equal observation exposure |
| Obs-only: 16,000 obs | 16,000 | 0 | 16,000 | Same observation-only trajectory; equal total optimizer updates |

This is the **62,966,546-parameter Small conditioned mixed global** model on the 180×360 OM4 Gaussian grid: 77 physical state slots, affine InstanceNorm, an initializer and processor shared across tasks with source-specific, identity-initialized 1×1 input adapters, and no additional memory channels. Training starts from random weights and mixes OM4 with observations throughout; observation frequency increases with training. The mixed endpoint includes both sources throughout training, rather than a pure OM4 pretraining phase followed by fine tuning. There is no observation-only finish. The final checkpoint is the 8k/8k endpoint, not the validation-selected checkpoint.

## Completed matched observation-only comparison

**Small conditioned scratch global** uses exactly the mixed model's architecture, seed, observation-derived normalization, input adapters, missingness handling, global observation losses and optimizer settings. It starts from fresh random weights and performs zero OM4 / 16,000 observation updates. It never loads OM4 weights or qualification weights; its OM4 input adapter stays inactive. There is no separate reconstruction-only phase: initializer reconstruction/completion remains part of the joint observation objective, as in the mixed control. The observation sampling stream is the same at equal observation exposure. Both models use all available latitudes for inputs, losses, validation and scoring.

The 8k checkpoint matches observation exposure; the 16k checkpoint matches total optimizer updates. Mixed spends half of its 16k updates on OM4. These are rough update-budget comparisons, **not exact FLOP or GPU-time matches**. All three report checkpoints are immutable fixed milestones, not the validation minimum. [Run design and completion](global-scratch-2026-09-30.md).

### Existing combined benchmark and its components

The table reproduces the existing global benchmark on 96 monthly test origins, 2015–2022. SST, geostrophic velocity and EKE errors combine leads 5/15/30; OHC errors compare calendar-month means. Spectral dex averages 27 validation-qualified SST/SSH/scalar-EKE comparisons at those leads. **This table is not the wider day-30-only spatial-spectrum calculation below.** Lower values are better.

| Model | Composite | SST °C | Geo. velocity m/s | EKE m²/s² | OHC 0–700 GJ/m² | OHC 700–2000 GJ/m² | Spectral dex |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final: 8,000 obs | 0.5927 | 0.577 | 0.1048 | 0.02338 | 0.740 | 0.422 | 0.344 |
| Obs-only: 8,000 obs | 0.6255 | 0.612 | 0.1073 | 0.02384 | 0.809 | 0.443 | 0.367 |
| Obs-only: 16,000 obs | 0.5862 | 0.509 | 0.1071 | 0.02357 | 0.723 | 0.418 | 0.348 |

Reporting composite = 0.5 × mean of five integrated-error ratios to the **same held-out training-climatology errors** + 0.5 × mean spectral dex. Training selection retains its original **frozen validation** climatology denominators. The qualification, mixed and scratch reference hashes are identical. Reporting uses the same normalization as the preceding global comparison and reproduces its mixed endpoint exactly; test values never select checkpoints or tune the protocol. Dex is an absolute log10 power discrepancy: 0.3 is approximately a factor of two and 1.0 a factor of ten.

At equal observation exposure, the mixed composite is **5.2% lower**. At equal total updates, obs-only 16k is **1.1% lower** than mixed: SST and OHC compensate for worse velocity/EKE and slightly worse aggregate spectra. Extending obs-only from 8k to 16k improves this fixed-checkpoint benchmark by **6.3%**. One seed and a narrow difference do not establish a general winner.

| Model | Forecast composite | Persistence of that model's inferred initial state |
|---|---:|---:|
| Final: 8,000 obs | 0.5927 | 0.5682 |
| Obs-only: 8,000 obs | 0.6255 | 0.5984 |
| Obs-only: 16,000 obs | 0.5862 | 0.5821 |

Each persistence baseline holds that checkpoint's full initialized state fixed. Surface observations are copied identically, but learned interior states differ, so its composite is model-dependent. All three forecasts remain worse than their own persistence on the combined metric despite improving SST. A between-model gain therefore does not establish aggregate skill above this baseline.

Mixed finishes at validation composite 0.565245, compared with 0.626532/0.593291 for obs-only 8k/16k. Mixed therefore remains ahead on validation; scratch's narrow held-out endpoint advantage is not a stable across-cohort ranking. All report values use the fixed checkpoints listed above.

### Validation score during training

![Global integrated-plus-spectral validation score: raw traces and EMA, versus total and observation updates](artifacts/2026-10-01-global-validation-ema/global-validation-training-curves.png)

Thin traces show all 173 mixed and 164 obs-only validation scores; bold traces show an exponential moving average (EMA) with a **500-total-update half-life**. The EMA starts at the first raw score and updates in linear score space using `decay = 2^(-Δtotal_updates / 500)`. The same EMA values appear on both axes; the right panel changes only the horizontal coordinate to observation updates. This accounts for irregular validation intervals, including extra milestone checks. Filled circles mark the raw scores at the three report checkpoints. The two obs-only checkpoints belong to the same training trajectory.

Both curves use the identical frozen validation-climatology reference and the established integrated-plus-spectral score. Smoothing is for display; it does not change checkpoint selection or the reported endpoint numbers. [Raw points, EMA values and definition](artifacts/2026-10-01-global-validation-ema/validation-curve-results.json.gz) · [Source hashes and checks](artifacts/2026-10-01-global-validation-ema/provenance.json.gz).

## Learned initializer fields and observation context

SST and SSH reference panels show the **same final five-day history interval** used by the initializer: the five days immediately before the forecast origin. Available surface values are copied into the initialized state by construction; agreement there verifies the supplied state, rather than learned reconstruction skill. **These panels do not test whether the network learns an identity function: an explicit overwrite supplies observed SST/SSH after the network runs.** Their learned surface behavior is confined to missing ocean cells. The [pinned initializer code](https://github.com/m2lines/Samudra/blob/79025a163817a577ab81af95b401f5cb0563cd12/src/samudra/experiments/initializer_models.py#L240) implements this overwrite. **Gray marks fixed model land (or the field’s depth mask); white marks missing observations within model ocean cells.** In missing ocean cells the initializer learns a completion from history, geography, season and forcing.

[Initialized SST: copied input plus completion](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel0.png) · [Initialized SSH: copied input plus completion](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel6.png)

Interior reference panels use the **preceding December IAP monthly analysis**, explicitly labeled as context. A monthly average is not instantaneous initializer truth. U/V reference panels use the contemporaneous DUACS geostrophic surface-velocity proxy; these are not direct observations of the model’s full velocity state and are not its observation-task training targets. Those distinctions apply to every map link below.

**Why the mixed model’s observation-initialized velocities differ from OM4:** the [supplemental task-path analysis](velocity-task-supplement-2026-10-01.md) compares the same frozen checkpoint on OM4 and observation inputs and swaps only its task adapter. Its normal OM4 path recovers recognizable currents (mean U spatial correlation 0.791), while the observation path differs substantially (0.312). The supplement includes paired maps, separate U/V errors and the actual evaluated loss contributions.

The initializer U/V panels also include **physical OM4 surface velocities** (`uo_0`/`vo_0`) as model-data context. For the January 2015 and 2018 examples, OM4 covers December 27–31 of the preceding year, matching the last five-day input interval. For January 2021, the available OM4 bin is **December 26–30, 2020**, overlapping four of the five December 27–31 initialization days. The stored midpoint ±2.5-day windows leave December 31 uncovered; the panel explicitly says **4/5 days overlap**, and no gap filling was applied. OM4 physical currents and DUACS geostrophic currents represent different quantities; neither panel turns the observation-trained internal U/V slots into supervised current predictions. [Dates, coverage and source audit](artifacts/2026-10-01-global-physical-initial-om4/provenance.json.gz).


The featured diagnostics use **learned temperature below the copied surface level and learned salinity**, rather than surface copying or coastal infilling. The depth profiles compare the three report checkpoints with the preceding December IAP analysis and **Training December climatology**: per-location, per-depth averages of the twenty December analyses in the training set, 1993–2012. This baseline is a diagnostic reference; it does not provide the model's missing inputs or initialized fields.

![Learned temperature and salinity depth profiles, with IAP and training climatology](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-profiles.png)

Profiles cover global observed support and the displayed North Atlantic, North Pacific and Southern Ocean boxes. Each depth uses the same finite analysis/climatology/model wet support and spherical cell areas for every curve. Temperature at 2.5 m is omitted because it is copied; the displayed temperature depths are 10–1850 m, while all salinity depths from 2.5–1850 m require prediction.

### Whole-December initializer reconstruction

The featured latitude–depth comparison now uses **whole-December model means from independent initializations**, matching the monthly quantity in the observation reconstruction objective. It calls the original `reconstruct_month` operator: initialize separately from each rolling 95-day surface/forcing history, take its latest T/S state, and average the seven December bins with calendar weights **5/31 six times and 1/31 once**. The processor never runs. Model means and IAP December means both subtract the **same per-location/depth training December climatology, 1993–2012**, with unchanged common finite support and color scales.

![Whole-December independently initialized temperature and salinity anomalies](artifacts/2026-10-01-initializer-monthly/2014-12-initializer-zonal-anomalies.png)

[December 2017](artifacts/2026-10-01-initializer-monthly/2017-12-initializer-zonal-anomalies.png) · [December 2020](artifacts/2026-10-01-initializer-monthly/2020-12-initializer-zonal-anomalies.png) · [Original late-December-state panel](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-zonal-anomalies.png).

**This is retrospective monthly reconstruction, not a forecast.** Each initialization sees surface observations through its current five-day bin. As in training, the final bin spans December 31–January 4, with only December 31's 1/31 weight in the monthly mean. These few following-month observations are part of the existing boundary-bin convention; no IAP interior values are initializer inputs. December 2014 was assembled as an additional held-out diagnostic from the existing prepared daily products; training splits and benchmark cohorts are unchanged.

Much of the formerly shared cold anomaly was caused by comparing a late-December state with a whole-month reference. In December 2014, the mixed model's area-weighted temperature anomaly at 30–60°N and modeled depths 10–105 m changes from **−0.540°C** in the original panel to **+0.006°C** for the monthly mean, versus **+0.136°C** in IAP. Obs-only 16k changes from **−0.440°C to +0.131°C**. All three models shift warmer by approximately 0.55°C in this band; the corresponding shifts are similar in December 2017 and 2020. This removes much of the apparent common error without changing the checkpoints. Remaining anomaly differences, particularly salinity, are still visible.

The separate 550 m map below still shows the **last pre-January initialized state**, with December IAP as descriptive context; it is not the monthly mean above.

![Learned 550 m temperature departures from training December climatology, including IAP context](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-temperature550-anomaly.png)

For the monthly reconstruction comparison below, each depth's global spatial RMSE is averaged equally across the learned levels and then across **December 2014, 2017 and 2020**. Temperature at the copied 2.5 m level remains excluded. These are errors of independently initialized monthly means against IAP, with no forecast or checkpoint selection implied.

| Model/reference | Monthly reconstruction T RMSE, 10–1850 m °C | Monthly reconstruction S RMSE, 2.5–1850 m psu |
|---|---:|---:|
| Final: 8,000 obs | 0.310 | 0.0787 |
| Obs-only: 8,000 obs | 0.330 | 0.0794 |
| Obs-only: 16,000 obs | 0.306 | 0.0785 |
| Training December climatology | 0.332 | 0.0668 |

**Mixed and obs-only 16k beat temperature climatology in each of the three monthly examples; none beats salinity climatology.** Mixed improves temperature reconstruction versus scratch at equal observation exposure, while scratch16k is slightly better at equal total updates. Salinity reconstruction remains very similar across the three models. This corrects the interpretation of the earlier last-five-day/context comparison, where all three were worse than climatology for both quantities. The older profiles, context-error curves and single-state maps are retained and labeled as descriptive context; they do not represent this monthly reconstruction statistic. Plausible absolute maps alone remain insufficient evidence of recovered variability, and these diagnostics do not establish whether forecasts use the learned interior state causally. [Monthly values, sources and checks](artifacts/2026-10-01-initializer-monthly/provenance.json.gz).

| Diagnostic | 2015 | 2018 | 2021 |
|---|---|---|---|
| Learned T/S depth profiles | [profiles](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-profiles.png) | [profiles](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-profiles.png) | [profiles](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-profiles.png) |
| Context agreement by depth | [errors](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-context-errors.png) | [errors](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-context-errors.png) | [errors](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-context-errors.png) |
| Whole-December latitude–depth anomalies | [December 2014](artifacts/2026-10-01-initializer-monthly/2014-12-initializer-zonal-anomalies.png) | [December 2017](artifacts/2026-10-01-initializer-monthly/2017-12-initializer-zonal-anomalies.png) | [December 2020](artifacts/2026-10-01-initializer-monthly/2020-12-initializer-zonal-anomalies.png) |
| Learned temperature at 550 m | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-temperature550-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-temperature550-anomaly.png) | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-temperature550-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-temperature550-anomaly.png) | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-temperature550-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-temperature550-anomaly.png) |
| Learned surface salinity | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-salinity0-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2015-01-01-initializer-salinity0-anomaly.png) | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-salinity0-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2018-01-01-initializer-salinity0-anomaly.png) | [absolute](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-salinity0-absolute.png), [anomaly](artifacts/2026-10-01-global-physical-endpoints/initializer/2021-01-01-initializer-salinity0-anomaly.png) |
| Supplied/copied SST | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-initial-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-initial-channel0.png) |
| Supplied/copied SSH | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-initial-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-initial-channel6.png) |
| Surface zonal velocity proxy comparison | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-initial-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-initial-channel4.png) |
| Surface meridional velocity proxy comparison | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-initial-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-initial-channel5.png) |

[Learned initializer numbers](artifacts/2026-10-01-global-physical-endpoints/initializer-results.json.gz) · [Saved-state export and source provenance](artifacts/2026-10-01-global-physical-comparison/provenance.json.gz). The learned-field extraction reuses saved initialized states: 12 CPU seconds for mixed and nine for scratch, with no additional GPU inference beyond the endpoint evaluations.

**Why observation coastlines look different:** every panel uses the same fixed OM4 wet-cell mask. The observation panel additionally lacks values in coarse cells rejected during remapping: valid source area must cover at least 90% of the *full* target cell, which excludes many coastal/island cells. In the 2021 day-30 SST example, 3,598 of 44,892 model ocean cells lack an observation (7.0% of cosine-weighted wet area). Those cells were previously the same gray as land; they are now white. The model predicts in those ocean cells, but they have no observation error score. This is coverage, not a learned coastline.

## Day-30 errors and spatial structure

The error table averages each origin’s area-weighted RMSE over 96 independently initialized monthly test forecasts in 2015–2022, taking only the sixth five-day output (the interval from days 25 to 30). It does not average leads 5, 15 and 30. Persistence holds each checkpoint’s inferred initial state fixed; climatology is the same training-only monthly surface climatology.

| Model | SST RMSE °C | SST persistence °C | SSH RMSE m | SSH persistence m |
|---|---:|---:|---:|---:|
| Final: 8,000 obs | 0.640 | 1.168 | 0.0792 | 0.0800 |
| Obs-only: 8,000 obs | 0.694 | 1.168 | 0.0886 | 0.0800 |
| Obs-only: 16,000 obs | 0.572 | 1.168 | 0.0826 | 0.0800 |

### Day-30 official-kernel metrics, matched map cohort

Surface scores use **January 26–30** of 2015/2018/2021, exactly the day-30 map intervals. OHC uses each **full January forecast mean**, including the final day of the month; it is not an instantaneous day-30 comparison to monthly IAP. Lower is better.

| Model | SST RMSE °C | Geostrophic velocity RMSE m/s | OHC 0–700 RMSE GJ/m² | OHC 700–2000 RMSE GJ/m² |
|---|---:|---:|---:|---:|
| Final: 8,000 obs | 0.695 | 0.168 | 0.764 | 0.470 |
| Obs-only: 8,000 obs | 0.706 | 0.170 | 0.794 | 0.489 |
| Obs-only: 16,000 obs | 0.629 | 0.170 | 0.727 | 0.482 |
| Date-matched OM4 1° (context) | 0.799 | 0.194 | 2.089 | 1.578 |

On this cohort, obs-only 16k has lower SST and upper-OHC error; mixed has lower geostrophic velocity and deep-OHC error. See the applicability decisions below before comparing with the original suite's published long-rollout numbers.

### Why is OM4 OHC error larger?

A follow-up audit reproduces the scores and finds that **a global offset does not explain most of the gap**. Below, bias is prediction minus IAP, averaged over the three January cases. Centered RMSE removes each case's area-weighted mean error before pooling its spatial squared errors. All entries are GJ/m², on the identical comparison support; centered scores are diagnostics and do not replace the official-kernel table.

| Model | Layer | RMSE | Mean bias | Centered RMSE |
|---|---|---:|---:|---:|
| Date-matched OM4 1° (context) | 0–700 m | 2.089 | -0.380 | 2.054 |
| Date-matched OM4 1° (context) | 700–2000 m | 1.578 | -0.804 | 1.358 |
| Final: 8,000 obs | 0–700 m | 0.764 | -0.449 | 0.617 |
| Final: 8,000 obs | 700–2000 m | 0.470 | -0.208 | 0.422 |
| Obs-only: 16,000 obs | 0–700 m | 0.727 | -0.382 | 0.619 |
| Obs-only: 16,000 obs | 700–2000 m | 0.482 | -0.184 | 0.445 |

OM4's upper-layer RMSE is equivalent to about **0.75°C** error in the 700-m layer-average temperature; its deep-layer error corresponds to about **0.30°C** over 1,300 m. The extraction uses °C, exact model layer thicknesses, full monthly overlap weights, the same heat-capacity/density constants and matching complete-column masks. No unit or pairing error was found in these checks. A double-precision bias/variance decomposition reproduces every previously reported per-case OHC MSE within 2×10⁻⁶ relative error.

As a magnitude check, the [original metrics commit](https://github.com/m2lines/Samudra/commit/14455906efb26e227171b4c467fe4589d95aa01e) reported OM4 OHC RMSEs of about **3.01/2.20 GJ/m²** for the two layers. That uses a different resolution, cohort and support, so it is not a numerical acceptance target for our **2.09/1.58** result; it shows that multi-GJ/m² OM4 errors are not unprecedented.

The defensible interpretation is improved agreement with **IAP**, which directly supervises the learned models. OM4 is a separate simulated realization with its own mean spatial structure; this audit does not separate persistent regional climatology differences from time-varying mismatch. Lower IAP RMSE alone does not establish better ocean dynamics, and the previously documented vertical-representation differences remain. All six months and four models' decompositions are archived in the metric provenance.

### Official metric definitions: applicability to these rollouts

These tables use the numerical kernels from [the official metrics commit `14455906`](https://github.com/m2lines/Samudra/commit/14455906efb26e227171b4c467fe4589d95aa01e), with explicit fixed-lead adaptations. They are **not the unmodified complete-calendar-year official report**, and do not replace the training selection score. “Final: 8,000 obs” is the mixed 8k OM4 + 8k observation endpoint; the two obs-only rows are the fixed 8k and 16k endpoints defined above. OM4 is the date-matched simulation reference, not an observation-initialized forecast.

Both lead tables use the **same three January 1 initializations: 2015, 2018 and 2021**. For each case we compute spatial area-weighted MSE on the native prepared observation grid, average the three case MSEs equally, then take the square root. This differs from averaging case RMSEs. It is a three-example diagnostic: the earlier day-30 table uses 96 origins, and the spatial spectra use twelve monthly 2015 origins. Comparisons across those tables also change the cohort and spatial operator.

| Official metric or feature | Treatment here |
|---|---|
| `obs/sst/total_rmse` | SST versus native 0.25° OISST; model field linearly interpolated to the observation grid. Exact matching five-day intervals. |
| `obs/velocity/total_rmse` | Derive U/V from model SSH on its own grid, interpolate to native 0.125° DUACS, compare with DUACS `ugos`/`vgos`. Vector squared error is Δu² + Δv²; exclude ±5° as specified by the kernel. The model's unconstrained U/V state channels are not used. |
| `obs/ohc_0_700/per_area_total_rmse`, `obs/ohc_700_2000/per_area_total_rmse` | Calendar-month mean OHC maps, model layers versus native IAP layers, with ρ=1035 kg/m³ and Cp=3850 J/(kg K), referenced to 0°C. Model OHC is interpolated to the native 0.5° IAP grid. We retain this experiment's complete-column support on both sides; the original suite can include partial columns, so this is an additional explicit support restriction. These are per-area spatial errors, not the global ZJ totals plotted below. |
| `obs/eke/total_rmse` | **Not reported at an isolated lead.** Official instantaneous EKE uses anomalies about each field's time mean, not ½(u²+v²). Estimating that mean from only three December endpoints (or three January endpoints) would define a different, poorly sampled variability diagnostic. Full-trajectory EKE would blend leads and is deferred rather than mislabeled as day-365 EKE. |
| SST residual-variance map RMSE / pattern correlation | **Not reported:** fixed-lead samples do not provide the continuous seasonal record required for the official detrending/deseasonalization procedure. |
| Upper-700-m OHC residual-variance map RMSE / pattern correlation | **Not reported:** same temporal limitation. |
| Annual standard deviation and calendar-year bootstrap CI | **Not reported:** three endpoint/month samples are not complete-year blocks. The original official driver deliberately requires complete years; its guards were not bypassed. |
| Temporal spectra | Omitted as requested. Spatial spectra remain separate diagnostics, not substitutes for missing temporal metrics. |

SST and velocity retain approximately **98.5% and 93.2%** of their finite native observation area after pairing. OHC retains approximately **99.0% and 99.1% of complete-column observation area** in the two layers; those percentages are not fractions of the whole ocean. The native-grid operator erodes coastal support where interpolation touches missing model cells. All three learned models have the same pairing support. We preserve the official global domain, with no ±60° cutoff.

The original suite scores a completed time series on calendar-year blocks; these tables instead answer the requested lead-specific question. OHC necessarily answers a monthly question, and the native-layer/model-layer representation difference documented below still contributes. No learned correction, detrending, fitted SSH offset or retuning is applied. OM4 OHC uses overlap-weighted full January/December means of its 19 native temperature levels, integrated with the same model-layer thicknesses and constants. Its complete-column masks and native-IAP pairing support exactly match those of the learned models. The OM4 monthly samples are structural simulation context, not observation-initialized forecasts.

![SST at day 30 with matching observations](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel0.png)

![SSH at day 30 with matching observations](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel6.png)

At day 30, mixed SST RMSE is **45.2% lower than persistence** and its SSH advantage over persistence is **1.1%**. Obs-only 16k improves day-30 SST by **10.7%** versus mixed, while SSH is **4.3% worse** and remains worse than surface persistence. These per-origin-mean RMSEs differ slightly from the benchmark's pooled RMSEs; estimators are kept separate throughout.

The mixed endpoint retains a mean day-30 SST bias of **−0.273°C** and SSH bias of **−0.0489 m** over the 96 origins. In the January 2015 day-30 example, the coldest predicted wet SST is −3.98°C, still colder than the reference minimum of −1.80°C. Obs-only 8k has SST/SSH biases of −0.375°C/−0.0623 m; at 16k they improve to −0.202°C/−0.0522 m. SSH ripples and conspicuous internal-velocity bands remain. The internal U/V maps are useful diagnostics of what is carried forward, but their comparison with geostrophic proxies is not a full-current skill assessment.

**Date-matched OM4 context:** all day-30 and day-365 field maps, including SST, SSH, temperature at 550 m and surface salinity, include an **OM4 model-data sample** for the same five-day forecast interval. Physical U/V maps include that OM4 comparison alongside the existing DUACS geostrophic proxy. Panel labels print the inclusive dates: January 26–30 at day 30, December 27–31 at day 365, for each of 2015/2018/2021. OM4 uses the same Gaussian grid, field/depth masks, physical units and color scales; source averages are overlap-weighted when needed. These samples provide structural model-data context, **not observational ground truth or an observation-initialized forecast baseline**. Unconstrained physical slots can carry learned information, so their departure from OM4 does not by itself establish an observation forecast error. [Reference dates and field provenance](artifacts/2026-10-01-global-physical-om4-maps/provenance.json.gz).

| Field | Initialization, 2015 | Day 30, 2015 | Day 30, 2018 | Day 30, 2021 |
|---|---|---|---|---|
| SST | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel0.png) |
| SSH | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel6.png) |
| Temperature at 550 m | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel1.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel1.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel1.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel1.png) |
| Surface salinity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel2.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel2.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel2.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel2.png) |
| Surface zonal velocity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel4.png) |
| Surface meridional velocity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-initial-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day30-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day30-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day30-channel5.png) |

All model tiles retain one image pixel per model grid location, common scales and full polar coverage. The [numerical artifact](artifacts/2026-10-01-global-physical-endpoints/map-results.json.gz) records ranges and color-limit clipping, so saturated cold/warm values are not hidden by interpretation of the maps.

## Wider spatial spectra, all at day 30

The six regions follow [Samudra 2, Figure 9 and Appendix A.5](https://arxiv.org/abs/2606.02610v2): Gulf Stream, Kuroshio, Agulhas, Malvinas, Niño 3.4 and Tropical Pacific. The last box spans 130–290°E and 30°S–30°N. Curves average **power spectra of twelve day-30 forecasts**, one from each month of 2015. No curve blends forecast leads or uses annual forecast states.

Each final comparison panel includes the mixed 8k endpoint and obs-only 8k/16k endpoints on the nominal 1° OM4 grid, together with OM4 on that grid, observations on that grid and native observations. Native OISST is 0.25°; native prepared DUACS ADT is 0.125°. “Native” means the existing prepared product grid before the 1° remap. All use the same calendar five-day intervals. OM4 averages are combined by exact time-interval overlap when their native five-day phase differs from the observation bins; those weights are saved. OM4 is a model-data reference, not the verifying ocean state for an observation-initialized forecast.

![Final SST spectra across six regions](artifacts/2026-10-01-global-physical-comparison/final-sst-spectra.png)

![Final SSH spectra across six regions](artifacts/2026-10-01-global-physical-comparison/final-ssh-spectra.png)

![Final geostrophic KE spectra across six regions](artifacts/2026-10-01-global-physical-comparison/final-geostrophic-ke-spectra.png)

The mixed endpoint is closer to coarse observations than OM4 in **SSH and its derived geostrophic velocity** at resolved wavelengths. It still lacks power at shorter resolved wavelengths relative to **observations already coarsened to 1°**. The native curves additionally reveal variability beyond the model grid's representable range.

At 16k, obs-only matches coarse SST spectra more closely than mixed, but additional observation training does not close the mixed model's SSH/geostrophic-KE spectral advantage. This is consistent with the day-30 field errors and suggests a field-dependent source benefit.

To quantify the endpoint differences, the table gives mean absolute differences in log10 spectral power. It averages bins at wavelengths of at least four grid spacings in each of the four boundary-current boxes, then weights those boxes equally. Lower is closer; a difference of 0.3 corresponds to a factor of about two in power, and 1.0 to a factor of ten. This is a descriptive diagnostic, not the training selection metric.

| Model | SST vs obs 1° | SST vs OM4 1° | SSH vs obs 1° | SSH vs OM4 1° | Geo. KE vs obs 1° | Geo. KE vs OM4 1° |
|---|---:|---:|---:|---:|---:|---:|
| Final: 8,000 obs | 0.189 | 0.208 | 0.075 | 0.583 | 0.065 | 0.863 |
| Obs-only: 8,000 obs | 0.199 | 0.225 | 0.120 | 0.507 | 0.111 | 0.795 |
| Obs-only: 16,000 obs | 0.165 | 0.228 | 0.113 | 0.510 | 0.114 | 0.791 |

The plotted KE curve is **½(Su + Sv)** from geostrophic velocities derived from each SSH field, with its own spatial mean/plane removed. This follows the component-spectrum construction in the paper, but is a spatial velocity-fluctuation diagnostic, not temporally defined EKE or the spectrum of the scalar EKE field used in prior selection. Deriving velocities from every SSH grid makes their construction explicit; it also changes the reference from the supplied DUACS U/V used by the training metrics. Geostrophy excludes ±5°, so the Niño 3.4 velocity panel is unavailable rather than filled or extrapolated.

**Land and scale treatment:** these are masked regional pseudo-spectra. Coarse curves share finite reference/OM4 wet support, fixed across all twelve dates. Native curves use that geographical support plus their native missing-value mask. A plane is fitted on finite cells; unsupported residuals contribute zero to the Fourier transform. A periodic 2D Hann window and its wet-support variance correction are applied. The mask’s spectral coupling is not deconvolved, so coastal/island features can affect especially short wavelengths. The 2D PSD is normalized per angular-wavenumber area, with k × PSD on the vertical axis. Angular wavenumbers are shown only below each grid’s lower Nyquist. Gaussian-grid latitude spacing is approximated by regional median spacing, and longitude by the cosine of regional mean latitude; the very broad Tropical Pacific panel has greater geometric approximation than the smaller boxes.

The figures retain absolute snapshot fluctuations after spatial plane removal; they are not a seasonal-anomaly or continuous eight-year spectrum reproduction of the paper. Powers are averaged after transforming individual snapshots. These wider diagnostics do not replace or alter the validated checkpoint-selection metrics. Native-to-coarse daily remapping and five-day aggregation are numerically checked against the actual coarse targets.

## Annual SST/SSH means, OHC totals and maps

One initialization, 73 five-day autoregressive steps, prescribed future ERA5, no future surface-state corrections. The plots show **undetrended SST/SSH means and total OHC in ZJ**, for all three endpoints, with observations, each model's initialized-state persistence and training-only monthly climatology. SST and SSH use five-day intervals; OHC uses calendar-month averages, separately for 0–700 and 700–2000 m. OHC is the sum of column heat content times spherical cell area, divided by 10²¹ J/ZJ. It is integrated over fixed, common observed support, with no extrapolation into unobserved ocean. Its temperature reference remains 0°C.

### Day-365 official-kernel metrics, matched map cohort

Surface scores use **December 27–31** after the same three January initializations; they are endpoint errors, not averages over all 73 leads. OHC uses each **full December forecast mean**. Definitions, native-grid pairing, unavailable temporal diagnostics and uncertainty limitations are given in [the applicability table](#official-metric-definitions-applicability-to-these-rollouts). Lower is better.

| Model | SST RMSE °C | Geostrophic velocity RMSE m/s | OHC 0–700 RMSE GJ/m² | OHC 700–2000 RMSE GJ/m² |
|---|---:|---:|---:|---:|
| Final: 8,000 obs | 0.751 | 0.191 | 1.193 | 0.581 |
| Obs-only: 8,000 obs | 0.871 | 0.225 | 2.459 | 1.140 |
| Obs-only: 16,000 obs | 0.798 | 0.250 | 2.770 | 1.078 |
| Date-matched OM4 1° (context) | 0.774 | 0.196 | 2.102 | 1.563 |

Mixed has lower error than both obs-only endpoints in every computed column on these three examples. In particular, a smaller global OHC mean offset for obs-only does not imply lower spatial OHC error: opposing regional errors can cancel in a global total. These native-grid endpoint results are descriptive, not a new checkpoint-selection criterion.

![2015 annual SST/SSH means and OHC totals](artifacts/2026-10-01-global-physical-comparison/2015-01-01-annual-global-means.png)

[2018 global means](artifacts/2026-10-01-global-physical-comparison/2018-01-01-annual-global-means.png) · [2021 global means](artifacts/2026-10-01-global-physical-comparison/2021-01-01-annual-global-means.png)

The mixed endpoint broadly follows the seasonal SST cycle but remains cold; SSH settles near the older training climatology. Obs-only 8k is colder and has lower SSH. At 16k, obs-only approaches mixed in SST and upper-OHC global means and has **smaller deep-OHC mean offsets** in all three years. Its SSH mean remains lower. These global averages do not make the obs-only annual spatial behavior preferable: **both scratch checkpoints have much stronger equatorial SST ripples in all three displayed annual maps**. Smaller scalar bias can coexist with worse spatial structure.

For the mixed endpoint, both OHC bands have persistent negative offsets exceeding the native-IAP climatology baseline's absolute offset in all three cases. Its initialized-state persistence is already low, and evolution lowers OHC further. The obs-only 16k deep-OHC mean moves closer to observations during evolution, but this alone cannot validate unconstrained interior physics; the vertical-representation contribution below also affects the absolute offsets.

The mean forecast-minus-observation offsets are shown below. SST/SSH average the 73 five-day points; OHC averages the twelve monthly points equally:

| Model | Origin | SST °C | SSH m | OHC 0–700 ZJ | OHC 700–2000 ZJ |
|---|---|---:|---:|---:|---:|
| Final: 8,000 obs | 2015-01-01 | -0.254 | -0.0413 | -155.7 | -56.7 |
| Final: 8,000 obs | 2018-01-01 | -0.202 | -0.0478 | -158.8 | -56.2 |
| Final: 8,000 obs | 2021-01-01 | -0.210 | -0.0441 | -141.6 | -55.7 |
| Obs-only: 8,000 obs | 2015-01-01 | -0.433 | -0.0619 | -213.2 | -43.8 |
| Obs-only: 8,000 obs | 2018-01-01 | -0.371 | -0.0657 | -207.4 | -47.1 |
| Obs-only: 8,000 obs | 2021-01-01 | -0.388 | -0.0685 | -200.4 | -56.0 |
| Obs-only: 16,000 obs | 2015-01-01 | -0.261 | -0.0541 | -163.9 | -18.0 |
| Obs-only: 16,000 obs | 2018-01-01 | -0.190 | -0.0562 | -154.5 | -20.5 |
| Obs-only: 16,000 obs | 2021-01-01 | -0.214 | -0.0589 | -144.9 | -26.6 |

These are temporal mean offsets of the plotted series: area means for SST/SSH, supported area totals for OHC. They are not spatial RMSEs. The annual maps still show nonphysical structure, so matching seasonality or a global mean cannot establish spatial realism.

**OHC representation check:** the observed OHC integrates native IAP layers, whereas forecast OHC integrates the model's layers. Taking the *same* preceding-December analysis, sampling it at the model depths and applying the forecast integration already gives the following offsets relative to its native-layer integral:

| IAP context month | Model-layer minus native OHC 0–700 ZJ | Model-layer minus native OHC 700–2000 ZJ |
|---|---:|---:|
| 2014-12 | -64.7 | -17.2 |
| 2017-12 | -63.8 | -16.9 |
| 2020-12 | -63.8 | -17.0 |

This demonstrates a material representation contribution before model error. It is a three-month context check, not a correction to the annual curves or a precise decomposition of their biases. The context check has slightly smaller upper-band support (3.073×10¹⁴ m²) because the preceding-December analysis has additional missing columns. The additional drop from initialized-state persistence to the forecast is independent evidence of evolution changing the mean heat content.

SST/SSH means use cosine-latitude weights on the model grid. OHC totals use spherical areas from the same midpoint latitude/longitude cell bounds as observation remapping (R=6371 km, polar edges at ±90°). Each binary wet cell contributes its full grid-cell area; no fractional wet-area correction is available. The two retained OHC areas are 3.086×10¹⁴ m² (0–700 m) and 2.875×10¹⁴ m² (700–2000 m). Every curve in a given annual panel uses the same fixed support: model wet cells with finite observations and baseline values throughout that year. This removes changes in missing-data coverage as a source of apparent time-series drift. “Global” therefore means globally distributed observed support, not an estimate of unobserved ocean cells. The retained fraction of the model’s surface wet area is:

| Origin | SST | SSH | OHC 0–700 | OHC 700–2000 |
|---|---:|---:|---:|---:|
| 2015-01-01 | 93.0% | 93.0% | 81.8% | 76.2% |
| 2018-01-01 | 93.0% | 93.0% | 81.8% | 76.2% |
| 2021-01-01 | 93.0% | 93.0% | 81.8% | 76.2% |

![SST after a year: mixed final and obs-only 8k/16k](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel0.png)

![SSH after a year: mixed final and obs-only 8k/16k](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel6.png)

| Field | Annual, 2015 | Annual, 2018 | Annual, 2021 |
|---|---|---|---|
| SST | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel0.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel0.png) |
| SSH | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel6.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel6.png) |
| Temperature at 550 m | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel1.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel1.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel1.png) |
| Surface salinity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel2.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel2.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel2.png) |
| Surface zonal velocity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel4.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel4.png) |
| Surface meridional velocity | [maps](artifacts/2026-10-01-global-physical-endpoints/2015-01-01-day365-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2018-01-01-day365-channel5.png) | [maps](artifacts/2026-10-01-global-physical-endpoints/2021-01-01-day365-channel5.png) |

### Day-365 spatial spectra

These use the same six regions, transform, model labels and reference products as the day-30 plots: mixed 8k, obs-only 8k/16k, OM4 1°, observations 1° and observations on their native prepared grids. Each model curve averages **three individual day-365 power spectra**, from the January 2015/2018/2021 annual rollouts. Only the 73rd output is used: **days 360–365**, December 27–31 of the corresponding year (half-open interval ending January 1). OM4 and both observation grids match that exact interval. This is a spatial endpoint spectrum, not a spectrum averaged over the year or over forecast leads.

**The cohorts differ:** the day-30 panels average twelve monthly starts in 2015; these day-365 panels use three January starts in different years. Reference-power changes between the two sections can reflect season/year sampling as well as forecast behavior. The annual panels verify their references against the actual saved forecast targets; native-to-coarse masks/values and final map-versus-surface arrays agree.

![Day-365 SST spectra across six regions](artifacts/2026-10-01-global-physical-day365/day365-final-sst-spectra.png)

![Day-365 SSH spectra across six regions](artifacts/2026-10-01-global-physical-day365/day365-final-ssh-spectra.png)

![Day-365 geostrophic KE spectra across six regions](artifacts/2026-10-01-global-physical-day365/day365-final-geostrophic-ke-spectra.png)

The Tropical Pacific panels show a pronounced excess-power bump in obs-only 16k SSH and its derived geostrophic velocity. Obs-only 8k also has excess power there. Mixed avoids that bump but has too little SST power at the shortest represented wavelengths in the boundary-current regions. These are regional, field-dependent differences: plausible integrated power in other regions does not establish spatial realism, and a power spectrum cannot show whether fronts or eddies are in the right locations.

For context, the same descriptive log10-power distance used above, restricted to wavelengths of at least four grid spacings and equally weighting the four boundary-current boxes, gives:

| Model | SST vs obs 1° | SSH vs obs 1° | Geo. KE vs obs 1° |
|---|---:|---:|---:|
| Final: 8,000 obs | 0.110 | 0.144 | 0.168 |
| Obs-only: 8,000 obs | 0.161 | 0.143 | 0.141 |
| Obs-only: 16,000 obs | 0.168 | 0.126 | 0.152 |

Mixed is closer in this SST summary; SSH/geostrophic-KE distances do **not** uniformly favor mixed at day 365. This four-region summary excludes the Tropical Pacific bump, which is why the full regional panels remain essential. As at day 30, geostrophy excludes ±5°, making the Niño 3.4 KE panel unavailable. All distances are diagnostic, with checkpoint selection unchanged.

[Day-365 spectral numbers and masks](artifacts/2026-10-01-global-physical-day365/results.json.gz) · [Exact reference intervals, source hashes and CPU accounting](artifacts/2026-10-01-global-physical-day365/provenance.json.gz). The addition reused all saved forecasts, required two successful CPU-only extractions (7/11 seconds elapsed), and added **zero GPU-hours**. Sharing the spectrum code reproduces every previous day-30 curve exactly; the day-30 figures and forecast scores are unchanged.

## Methods and limits

Both model families use the Small D ConvNeXt U-Net: widths 128/192/256/384, one block per level, dilations 1/2/4/8, periodic longitude padding and zero latitude padding. The initializer takes 19 five-day history frames (95 days), finite-input indicators, forcings, spherical geographic coordinates and annual sine/cosine. It emits two states containing T/S/U/V at 19 depths plus SSH; the processor uses the previous two states. No physical freezing or velocity constraint is imposed.

Seed 1729; AdamW at 1e-4, weight decay 0.01, clipping 1, effective batch 8, no warmup. OM4 uses original forcings and full-state forecast/reconstruction losses. Observation forcings use the learned ERA5 adapter; observations supervise monthly T/S, SST/SSH forecast, interior reconstruction and artificial-gap surface completion. Forecast training covers six OM4 steps or six/seven observation steps spanning a month. The observation objective is 0.8 monthly T/S + 0.1 SST + 0.1 SSH + 0.1 interior reconstruction + 0.1 artificial-gap surface completion. These coefficients are not fractions of gradient contribution. Reconstruction trains the initializer from histories throughout the month; those later surface observations do not enter the forecast trajectory. The [preceding methods report](global-domain-results-2026-09-30.md#model-and-training-details) describes the shared experiment setup.

OHC integrates temperature relative to 0°C, using ρ=1035 kg/m³ and cp=3850 J/(kg K), with each product’s layer overlap against the depth band and complete column support. Forecasts use OM4 model layers; observations use native IAP layers before horizontal remapping. Observation OHC climatology is the per-location, per-calendar-month mean of training-only native IAP OHC integrals; surface climatology is the existing training-only monthly spatial climatology. These baselines do not supply the initializer’s missing values.

All checkpoint lineage hashes and counts, exact cohorts, global evaluation flags, prepared/native reference hashes and archive members are retained in the [evidence and provenance](artifacts/2026-10-01-global-physical-comparison/provenance.json.gz). The ZJ conversion passed full-sphere area and constant-field integration checks, plus missing-support/no-extrapolation checks. The new regional transform passed amplitude/plane-invariance, resolved-wavelength, Nyquist, empty-support and geostrophic-sign/equator checks. Map pixels and native/coarse equivalence were checked. Failed launch attempts are included in allocation accounting.

One seed per trajectory, twelve day-30 spectral snapshots and three day-365 snapshots support descriptive conclusions. The scratch run provides controlled equal-observation-exposure and equal-total-update endpoint comparisons against mixed training. No state intervention or source-specific supervision ablation is included, so these results do not causally explain the remaining structural advantage. Internal fields lacking observation supervision remain diagnostic states, and finite analyzed products do not establish direct measurement coverage under ice. Annual means can look good while spatial patterns are poor, which is why the maps remain alongside them.

The matched scratch run used **14.6258 training + 0.1683 evaluation = 14.7942 actual allocated GPU-hours**, with no GPU retries; training 18890321 and all four monthly/annual evaluations completed 0:0. Final collection 18892522 took 237 CPU seconds; learned scratch-state extraction 18895376 took nine CPU seconds. Previously recorded diagnostics and accounting remain in the provenance archive.

[Machine-readable map results](artifacts/2026-10-01-global-physical-endpoints/map-results.json.gz) · [execution plan](global-physical-focus-plan-2026-09-30.md) · [previous global model comparison](global-domain-results-2026-09-30.md)

The ZJ/coverage update reused the existing forecasts and added one 36-second CPU extraction of initialization OHC and training climatology; it used no GPU time. Its output checksum and sources are retained in the [heat-context evidence](artifacts/2026-09-30-global-physical-focus/heat-context-provenance.json.gz).

The matched comparison is complete as of October 1. [Completion, allocation and checkpoint receipts](artifacts/2026-10-01-global-physical-comparison/completion.json.gz) · [Benchmark components and baseline comparisons](artifacts/2026-10-01-global-physical-comparison/primary-comparison.json.gz) · [Validation curves and immutable milestones](artifacts/2026-10-01-global-physical-comparison/validation-curve-results.json.gz). The reported endpoint numbers, spectra and annual totals reproduce the original completed comparison.

The date-matched OM4 references came from one successful CPU extraction (18956440, seven seconds elapsed) and **zero new GPU-hours**. The same reference intervals and samples are retained in the endpoint maps. [Reference extraction audit](artifacts/2026-10-01-global-physical-om4-maps/map-audit.json.gz). Forecast scores, spectra, OHC series and checkpoint selection are unchanged.

The displayed model comparisons now contain only the mixed 8k/8k endpoint and obs-only 8k/16k. Maps, initializer profiles and sections were rendered from existing arrays; retained model/reference tiles, numbers, color scales and 1:1 pixels are unchanged. No new training, inference or GPU time was required. [Endpoint figure audit and sources](artifacts/2026-10-01-global-physical-endpoints/provenance.json.gz).

The initializer-velocity OM4 addition uses a separate successful CPU extraction, with zero new GPU-hours. Two short failed attempts are retained in the audit: the strict coverage check exposed the uncovered December 31, 2020 before the explicitly labeled partial reference was produced. All six updated figures retain their original model/DUACS tiles and color scales; all 48 other field maps are byte-identical. [Initializer reference and figure audit](artifacts/2026-10-01-global-physical-initial-om4/map-audit.json.gz).

The whole-December initializer diagnostic adds **0.0158 actual GPU-hours**, including the short duplicate request stopped by the output-preservation guard. All nine model/month cases passed seven-initializer/zero-processor checks and exact independent weighted-sum verification. Archive/member readback, prior climatology/reference equality and pinned runtime/checkpoint lineage checks passed. The [diagnostic provenance](artifacts/2026-10-01-initializer-monthly/provenance.json.gz) preserves both attempt logs, Alpha preparation inputs and executed helper source. No new training or rollout was performed.

The SST/SSH rollout maps also retain the observation reference and now include date-matched OM4 context for all three origins and both leads. OM4 SSH is the native `zos` field in meters; no offset is fitted to align it with DUACS ADT. These are model-data context panels, not an assumption of a shared absolute sea-level datum. All six OM4 intervals have complete five-day coverage. The 12 updated figures preserve all 48 original model/observation tiles pixel-for-pixel; the other endpoint maps are unchanged. Day-365 OM4 values reproduce those already used in the spectral comparison. CPU extraction job **18977362** completed successfully in seven seconds, with **zero GPU-hours**. [Surface-reference and figure audit](artifacts/2026-10-01-global-physical-surface-om4/provenance.json.gz).

The official-kernel endpoint tables were computed on an Alpha CPU node, with no model inference or GPU allocation. The archived results contain every case MSE, pairing fraction, original kernel/source hashes, input-array hashes and the exact aggregation. [Metrics provenance and numerical results](artifacts/2026-10-01-official-fixed-lead-metrics/provenance.json.gz).

The OM4 OHC baseline was added from six full-month temperature extractions. Torch CPU job **18978279** completed in nine seconds after one six-second schema-lookup failure; Alpha CPU scoring job **112642** completed in six seconds. All attempts are retained in the existing metrics provenance. Complete calendar coverage, model/OM4 support equality, observation fingerprint and pairing fractions passed; every preexisting metric value is unchanged. No GPU time was used.

## Deep T/S imprinting check

The [Samudra 2 comparison](imprinting-comparison-2026-10-05.md) examines T/S changes at 550/1850/3100 m and velocity-pattern associations. Scratch shows the strongest repeated deep ripples and salinity/velocity association; mixed is quieter but still develops broad deep-temperature bands. Much of the departure begins in the first processor step. The supplement distinguishes the paper’s fully supervised, eight-year anomalies/variance from our sparse one-year states and unconstrained observation channels below 2000 m.
