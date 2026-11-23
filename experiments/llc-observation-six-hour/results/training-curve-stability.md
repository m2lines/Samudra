<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Validation curves and checkpoint stability

The observation-only control's lower aggregate error across 24–72 hours persists
through late training in these runs. Both runs have earlier transient validation spikes,
so the full history does not support a claim that either trajectory is uniformly
stable. This comparison uses one training seed and the same eight validation
origins used for checkpoint selection; it does not measure seed variability or
independent test-period performance.

The mixed arm combines coarsened LLC4320 ocean-simulation snapshots with observed
sea-surface temperature/height targets; OBS-only uses observed targets throughout training. The arm names
remain fixed during the final observation-only cooldown. An origin is a forecast
initialization date. The eight origins span November 9, 2013 through July 28, 2014,
spaced approximately 37 days apart.

The [plotting script](../plot_validation_curves.py) reads all 110 saved
exponential-moving-average (EMA) validations per arm, from update 100 through
11,000 at intervals of 100. It plots the complete aggregate history, a late-training
zoom, and six physical-RMSE histories for sea-surface temperature (SST) and height
(SSH) at 24, 48 and 72 hours. No additional smoothing is applied. The full-history
and field panels use logarithmic error axes; the late zoom uses a linear axis.
The shaded final 1,000 updates use observation-only supervision, a learning-rate
cooldown, and an EMA reset at the 10,000-update boundary. The initial 2,000 updates
and their continuation are joined by total optimizer update, not restarted at zero.

The aggregate averages normalized RMSE equally across the two variables and three
leads. Within each variable/lead, wet-area mean squared errors are averaged across
the eight origins before taking the square root. Model and persistence have the
same joint origin/target observation support. The
[long-run report](long-training-comparison.md) gives normalization, split and
exposure definitions; lower error is better throughout.

![Validation normalized RMSE over all 11,000 updates and a late-training zoom](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-overview.png)

The left panel retains the transient spikes on a logarithmic axis; the right
panel expands late training on a linear axis. Lower RMSE is better. The shaded
interval marks the 1,000-update observation-only cooldown and EMA reset.

![SST and SSH validation RMSE at 24, 48, and 72 hours throughout training](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-fields.png)

Each panel compares the same field and forecast lead for both arms with
persistence. These are physical-unit validation RMSE curves, not the training
objective. The logarithmic axes retain the full spikes without clipping.

Mixed training spikes near updates 3,400 and 6,700–6,800. OBS-only spikes near
4,900–5,000, temporarily reversing the ranking. Both SST and SSH show these
transients. Validation reuses fixed dates, so a changing draw of validation dates
does not explain the spikes. The curves alone do not identify their cause.

| Window | Mixed aggregate error relative to OBS-only | Variable/lead comparisons with higher mixed error |
| --- | --- | --- |
| Last 20 main-phase checkpoints, 8,100–10,000 | +24.8% to +27.3%; median +25.7% | 120 / 120 |
| Ten cooldown checkpoints, 10,100–11,000 | +18.9% to +26.2%; median +20.5% | 60 / 60 |

At the final checkpoint, the mixed aggregate error is 18.85% higher. It is higher
on all eight individual validation origins at both 10,000 and 11,000 updates.
Recomputing the score after removing each origin in turn, without retraining,
leaves the final gap between 18.50% and 19.28%. No single omitted date reverses
the final ranking. Separately, the checkpoint table shows that the ranking
persists throughout the observed late-training window.
The neighboring checkpoints share training and EMA history; these 180 comparisons
are correlated checks, not 180 independent statistical replicates.

A descriptive paired-origin bootstrap resamples eight origin indices with
replacement, applying the same draw to both arms and all six variable/lead
components. Each of 20,000 draws recomputes the official MSE-then-RMSE aggregate.
Its central 95% range for mixed's relative excess error is +23.97% to +26.48%
at 10,000 and +17.73% to +19.97% at 11,000 (random seed 20261121). These ranges
measure sensitivity to resampling these eight dates. They are not uncertainty
intervals across training seeds, broader seasonal coverage, or a new test period.
The date-level scores are in [validation-origin-mse.csv](validation-origin-mse.csv);
[validation-origin-sensitivity.csv](validation-origin-sensitivity.csv) gives each
origin's aggregate comparison.

The 30-day ordering is a separate result. Mixed has lower day-30 SST and SSH error
than OBS-only at both saved evaluation checkpoints (10,000 and 11,000), but neither
final model beats day-30 persistence. There is no dense 30-day evaluation history
during training, so the curves here cannot establish checkpoint stability of that
long-horizon advantage. Independent seeds and additional held-out dates are needed
before interpreting either ordering as a general transfer effect. These are
matched optimizer budgets, not matched observation exposure.

To reproduce the figures and summary tables without training or inference:

```bash
uv run python experiments/llc-observation-six-hour/plot_validation_curves.py \
  --output agent-logs/validation-curves
```

The output contains PNG and SVG versions of `validation-training-overview` and
`validation-training-fields`, a combined aggregate CSV, paired variable/lead CSV,
origin sensitivity CSV, and a JSON summary with input hashes. Rendered images stay
outside Git. Published artifacts include [overview SVG](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-overview.svg),
[field SVG](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-fields.svg),
[aggregate CSV](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-aggregate.csv),
[paired field/lead CSV](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-paired.csv), and
[summary with input hashes](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/llc-observation-six-hour/validation-curves-20261123-e1c1cc8d4ed1/validation-training-summary.json).
The source histories are
[mixed initial](mixed-observation-validation.csv),
[mixed continuation](mixed-long-observation-validation.csv),
[OBS-only initial](obs-only-validation.csv), and
[OBS-only continuation](obs-only-long-validation.csv).
