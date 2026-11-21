<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Results: Yuan's normalized-delta weighting vs PR #969

Written by Claude Code (Claude Opus 5.5) on 2026-11-14. This is an experiment record for PR #984 and is not
intended to merge.

## Summary

PR #984 trained the 1° Samudra U-Net with Yuan's `normalized_delta` loss weighting and compared it with PR #969. Both
runs predict the change from the latest state, scaled by each channel's typical five-day change (τ), with 2 inputs and
1 output. Only the weighting differs (see [the README](../README.md)). Both checkpoints were scored on the same
official `v2026-09` data.

Yuan's weighting is better than #969's capped weighting on most measures:

- **Long-rollout drift is much smaller.** Over eight years, global-mean SST drifts by −0.20 °C instead of −0.31 °C.
  Global-mean surface salinity drifts by −0.01 psu instead of −0.08 psu.
- **SST error against satellite observations is 13% lower** (0.98 °C vs 1.13 °C; OM4 itself scores 0.77 °C).
- **One-step skill at 1050–5000 m is clearly better.** At 1850 m, the share of ocean area where the salinity forecast
  is worse than persistence falls from 72% to 43%.
- **Deep variability is closer to OM4's** once slow drift is removed. At 6000 m it is 0.7× OM4's, against 0.2× for #969.

It is worse in three places:

- **The bottom level (6000 m)** now loses to persistence for temperature and salinity, where #969 won narrowly.
- **Deep velocities** are 3–10% worse at one step.
- **Local SST runaways remain:** the hottest cell reaches 40.0 °C, where OM4 never exceeds 35.5 °C.

The validation score is unchanged (0.268 vs 0.265). Of the four predictions made before training, three were right and
one was partly right.

## Contents

1. [What was compared](#what-was-compared)
2. [How to read the measures](#how-to-read-the-measures)
3. [Training](#training)
4. [One-step forecasts](#one-step-forecasts)
5. [Eight-year rollout: drift and hot spots](#eight-year-rollout-drift-and-hot-spots)
6. [Variability by depth](#variability-by-depth)
7. [Skill against observations](#skill-against-observations)
8. [The pre-registered predictions](#the-pre-registered-predictions)
9. [What this means for #987, #988 and #989](#what-this-means-for-987-988-and-989)
10. [Provenance and reproduction](#provenance-and-reproduction)

## What was compared

| | #969 | #984 |
|---|---|---|
| τ, the typical five-day change per channel | root-mean-square over the whole grid, land counted as zero change | root-mean-square over ocean ("wet") cells only |
| Loss weights | 1/q², where q = τ/σ, capped at 1000:1 and rescaled to mean 1 | 1/q², uncapped |
| Training data | `om4_onedeg_v3` | official `v2026-09` |
| Evaluation data | `v2026-09` (re-evaluated for this comparison) | `v2026-09` |

σ is each channel's normalization standard deviation. The two runs share everything else: architecture, 70 epochs,
optimizer, an 8-step (40-day) training rollout, and seed. Both are evaluated from their final-epoch EMA checkpoint.

Under Yuan's weighting, each variable gets about the same share of the loss. Within a variable, each depth level's
share is proportional to its ocean area. Under #969's cap, deep temperature and salinity got far less: the cap binds
on 31 of 77 channels, all deep T/S.

## How to read the measures

- **Persistence** is the forecast "nothing changes over the next five days". It is the baseline any short forecast
  must beat. In the deep ocean, five-day changes are tiny, so even a model that has learned nothing has a small raw
  error there. Ratios to persistence are therefore more informative than raw errors.
- **One-step forecasts** start from the true OM4 state and predict one five-day step. They measure what the network
  learned, without accumulated error. There are 598 test dates, from 2014-10 to 2022-12. RMSE and bias are averaged
  over ocean cells, weighted by cos(latitude).
- **"Area worse than persistence"** is the share of ocean area where a cell's RMSE over all 598 dates exceeds
  persistence's RMSE there.
- **The rollout** starts from the true state in October 2014 and feeds each prediction back as the next input for
  598 steps, about eight years, with the true atmospheric forcing. It measures stability and drift.
- **Observation metrics** score the rollout against satellite and in-situ products, using `samudra.metrics`:
  - SST against OISST;
  - surface geostrophic velocity and eddy kinetic energy against DUACS altimetry;
  - ocean heat content (OHC) against IAP.

  OM4 is scored the same way, so "1× OM4" means the emulator matches its parent simulation.

## Training

![Best validation score by epoch](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/validation.png)

The validation score is the training code's rollout RMSE on the validation years. The log records it only when it
improves, so each curve is the best score so far.

- **#984 learns faster** (0.30 by epoch 13, which #969 reaches at about epoch 20), but it levels off near 0.268 from
  about epoch 44.
- **#969 ends at 0.265.** Its log starts at epoch 7 because that run resumed from a checkpoint.
- The validation score weights all channels equally in normalized units, so it is dominated by velocity. It does not
  register most of the differences below.

The stacked runs #988 and #989 are still training and are shown for reference. They will get their own reports.

## One-step forecasts

![One-step RMSE relative to persistence, by depth](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/onestep_rmse_by_depth.png)

Each point is a model's RMSE divided by persistence's at that depth. Below 1 beats persistence.

- **Above about 500 m the two runs are indistinguishable** for every variable. Surface temperature is 0.685 for #984
  and 0.680 for #969; surface salinity is 0.814 and 0.815.
- **Below about 500 m, and especially between 1050 and 5000 m, #984 is much better for temperature and salinity:**

  | Depth | T: #984 | T: #969 | S: #984 | S: #969 |
  |---|---:|---:|---:|---:|
  | 1850 m | **0.716** | 0.777 | **0.666** | 0.754 |
  | 3100 m | **0.658** | 0.729 | **0.763** | 0.855 |
  | 4000 m | **0.757** | 0.835 | **0.764** | 0.866 |
  | 5000 m | **0.920** | 0.965 | **0.930** | 0.977 |
  | 6000 m | 1.025 | **0.971** | 1.046 | **0.977** |

- **At 6000 m it reverses.** #984 becomes worse than persistence, by 2.5% for temperature and 4.6% for salinity, while
  #969 beats it narrowly. #984 beats persistence at 18 of 19 levels for T and S; #969 does at all 19.
- **Velocities are equal in the upper ocean and worse at depth for #984.** Zonal velocity at 6000 m is 0.769 against
  0.699, and meridional 0.722 against 0.656. The velocity share of the loss fell from 66% to 50%, so this was expected.
  Sea-surface height is equal (0.793 vs 0.790).
- **The equatorial surface temperature bias halves,** from −0.019 to −0.010 °C per step. This is the cold bias that
  #969's comment linked to its SST drift.

![Share of ocean area worse than persistence, by depth](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/onestep_area_worse.png)

The area view shows the same pattern. For salinity between 1050 and 3100 m, #984 loses to persistence over 43–52% of
the ocean, against 66–81% for #969. At 6000 m it loses over 91%, against 71%.

![Where each model beats persistence, salinity at 1850 m and temperature at 6000 m](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/onestep_maps.png)

Blue cells beat persistence and red cells lose to it.

- **At 1850 m,** #984 turns the tropical Pacific and much of the Indian Ocean from red to blue for salinity. Both runs
  still lose in the North Atlantic and the Arctic.
- **At 6000 m,** only scattered trenches and basins reach that depth. #984 is red in most of them, especially the
  northwest Pacific. #969 is close to neutral gray: it barely changes the state there at all.

**A hypothesis for the 6000 m reversal.** #984 gives the 6000 m channels *more* loss weight than #969, yet does worse
there, so loss weight is unlikely to be the cause. A better candidate is the output scale q = τ/σ. #969 averaged τ over
the whole grid with land counted as zero. At 6000 m, where almost every cell is land, that makes τ, and hence the
model's output step, very small, so #969 can only nudge the state, which keeps it near persistence. #984's τ averages
only the few deep wet cells, so its output step is larger, and with so little training signal at that depth the extra
freedom adds noise. PR #989 (a per-cell τ) tests this directly.

## Eight-year rollout: drift and hot spots

![Global mean SST and salinity bias, and the hottest SST anywhere, over the rollout](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/rollout_drift.png)

| | #984 | #969 | OM4 |
|---|---|---|---|
| Global mean SST bias, year 1 → last year | −0.05 → −0.20 °C | −0.11 → −0.31 °C | |
| Global mean surface salinity bias, year 1 → last year | −0.001 → −0.010 psu | −0.014 → −0.081 psu | |
| Hottest SST anywhere, year-1 mean → last-year mean | 33.2 → 37.6 °C | 33.7 → 39.8 °C | 33.0 °C in the last year |
| Single hottest value | 40.0 °C (Aug 2022) | 43.0 °C (Aug 2015) | 35.5 °C |
| Ocean cells above 34 °C at some point in the last year | 0.20% | 0.45% | 0.04% |

- **Both runs cool globally,** mostly in the first three years, and then level off. #984's cooling is about two-thirds
  of #969's.
- **Surface salinity drift almost disappears.** #969's falls steadily. #984's stays within 0.02 psu of OM4 for all
  eight years. Global-mean salinity should not drift at all, because total salt is conserved, so this is a real
  physical improvement.
- **Hot spots still grow,** though more slowly. They are local rather than global: the global mean cools while a few
  cells overheat.

![SST bias in the last year, with the hottest cell circled](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/sst_bias_last_year.png)

The hottest cells are on coasts where rivers enter shallow seas:

- **#984:** the head of the Bay of Bengal, at the Ganges–Brahmaputra delta (90.5°E, 23.4°N). It reaches 40.0 °C there,
  where OM4 reaches 33.7 °C.
- **#969:** the Kara Sea near the Ob estuary (83.5°E, 72.3°N), at 43.0 °C, a transient in its first year. In its last
  year the hottest cell is in the Gulf of Aden: 41.8 °C, where OM4 reaches 30.9 °C.

## Variability by depth

![Temporal standard deviation relative to OM4, by depth](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/variance_by_depth.png)

This figure is the standard deviation over time of each cell in the rollout, divided by OM4's, pooled over the ocean.
1 means the emulator varies as much as OM4.

- **Solid lines (raw).** Both runs look *too variable* in the deep ocean: 1.0–1.5× OM4 for #984 and 1.6–2.0× for #969
  below 1400 m.
- **Dashed lines (each cell's linear trend removed first).** Most of that excess disappears. Both runs are then
  *damped* at depth, which means the raw excess was slow drift, not variability.
  - With the trend removed, #984 sits at 0.82–0.95× OM4 between 1000 and 5000 m, and at 0.7× at 6000 m.
  - #969 falls to 0.53–0.59× at 5000 m and to 0.16–0.20× at 6000 m.
- **Velocity variability** is damped similarly in both runs, falling from 0.9× near the surface to about 0.6× at
  6000 m.

## Skill against observations

![RMSE against observations relative to OM4's, 2015–2022](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/observation_metrics.png)

Each point is the rollout's RMSE against an observational product, divided by OM4's own RMSE against it. Error bars
are 95% intervals from a year-block bootstrap.

| Measure, 2015–2022 | #984 | #969 | OM4 |
|---|---|---|---|
| SST vs OISST (°C) | **0.98** [0.91, 1.04] | 1.13 [1.01, 1.24] | 0.77 |
| Surface geostrophic velocity vs DUACS (m/s) | **0.197** | 0.207 | 0.194 |
| Surface EKE vs DUACS | 0.0455 | 0.0457 | 0.0451 |
| OHC 0–700 m vs IAP (GJ/m²) | **4.12** | 4.19 | 4.12 |
| OHC 700–2000 m vs IAP (GJ/m²) | **2.69** | 2.72 | 2.67 |

#984 is within 2% of OM4 on everything except SST, where its error is 1.27× OM4's, against 1.47× for #969. The two
runs' intervals overlap slightly for SST and fully for EKE; the velocity and OHC differences lie outside them.

## The pre-registered predictions

These were written in the PR description before training.

| # | Prediction | Confidence | Verdict |
|---|---|---|---|
| 1 | Balance across levels improves, mostly from better deep T/S | ~70% | **Mostly right.** T/S at 1050–5000 m improved by 5–12% in RMSE, and by up to 29 points in area worse than persistence. The bottom level got worse and now loses to persistence. |
| 2 | Velocity is about the same or slightly worse | ~60% | **Right.** Equal above 1000 m, and 3–10% worse below. |
| 3 | Deep variance stays damped | ~75% | **Right,** once slow drift is separated out: 0.7–0.95× OM4 at depth. The raw numbers looked too variable because of drift. #984 is much less damped than #969. |
| 4 | The surface SST runaway and drift persist | ~60% | **Right, but both are smaller.** Mean SST drift is about two-thirds of #969's, surface salinity drift is 8× smaller, and the peak SST is 40.0 instead of 43.0 °C. |

## What this means for #987, #988 and #989

- **Keep Yuan's weighting as the base.** It improves on #969 in drift, in skill against observations, and in deep
  one-step skill, at the cost of the bottom level and some deep velocity skill. All three stacked runs build on it.
- **#989 (per-cell τ)** is the direct test of the 6000 m hypothesis above. If the bottom level recovers, the output
  scale was the cause.
- **#987 (rollout curriculum)** targets what is left: drift and hot spots accumulate over the rollout, while one-step
  skill near the surface is already good.
- **#988 (large-scale penalty)** targets the global-mean cold drift, now −0.2 °C. Its validation score trails #984's
  (0.304 vs 0.268 at the same epoch), so its rollout will need to make up for that.
- **A new lead: the hot spots sit at river mouths in shallow seas.** The model is not given river runoff or freshwater
  flux (`wfo`) as forcing. These cells' heat budget depends on a very thin, fresh surface layer, so they may be poorly
  constrained. Worth checking before adding a bound or a curriculum aimed at them.

## Provenance and reproduction

- **#984 run:** `2026-11-13-normalized-delta-2in1out` on NYU Torch.
  - Training job 23073054: code `8285b6e7`, Rust loader, 4 RTX6000 GPUs with gradient accumulation 2, 70 epochs.
  - Eval 23153908, metrics 23153909, one-step forecasts 23153910.
  - Outputs are in `/scratch/am16581/runs/2026-11-13-normalized-delta-2in1out{,-eval}`.
- **#969 comparison:**
  - Rollout: re-evaluated on `v2026-09` (eval 23044852, metrics 23044853), with output in
    `runs/2026-11-09-onedeg-tendency-scaled-2in1out-rust-rtx8-eval-v2026-09`.
  - One-step forecasts: `compression_baselines/pr969_onestep/out_official/test`.
- **One-step scoring:**
  - Forecasts: #980's `one_step.py`, given `--preset compression_baselines/nd984_onestep/cfg/eval.yaml --backend cpu`.
    That preset is an obs-free copy of `eval.yaml`.
  - Scores: [`onestep_pr969/score_onestep.py`](../onestep_pr969/score_onestep.py), written to
    `compression_baselines/nd984_onestep/scores.json`.
- **Figures:** made by [`report_figures.py`](../report_figures.py) from the config `fig984.json`. They are stored,
  along with the config and per-depth variability numbers (`rollout_stats.json`), at
  `https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/normalized-delta-2in1out/reports/984/`.
  To reproduce:

  ```bash
  python report_figures.py fig984.json OUT_DIR
  ```

- **Data:** official `v2026-09` 1° (`docs/data.md`).
