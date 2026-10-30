<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Loss ablations: matched 2k comparison

Per-member MSE restores stochastic model B's initializer velocity RMSE to the
deterministic MSE control on the fixed OM4 diagnostic case. Its ensemble spread
also nearly disappears. Conversely, deterministic MAE produces velocity errors
close to stochastic fair CRPS. Changing the loss largely closes this case's velocity gap; stochastic
architecture alone does not prevent learning velocity. These are one-seed results, with interior scoring on one held-out
state, so they do not establish performance across dates.

All entries below use EMA weights after 2,000 mixed optimizer updates: 1,000 OM4
and 1,000 observation updates, effective batch eight, without cooldown. A has a
deterministic initializer and processor. B has a stochastic initializer and a
deterministic processor. The original A/B runs provide saved 2k controls; they
were not retrained. The initializer-only arm retains the 0.1 reconstruction loss
weight, freezes the untrained processor, and receives no forecast loss. Its
processor is not evaluated. “Joint” means reconstruction plus forecast training.
All arms use training seed 1729, Adam, learning rate 1e-4, EMA half-life 100
updates, the same data splits and normalization. The stochastic arms train with
four members. This is a shared-seed comparison, not a claim of identical random
number consumption across different objectives or microbatch schedules.

The reconstruction coefficient is 0.1 in joint and initializer-only training;
monthly and native forecast terms each have coefficient 0.5. The CRPS plus mean
MSE arm adds 1.0 times ensemble-mean MSE. Multiscale CRPS uses Gaussian widths
1, 2, 4 and 8 grid cells to construct five residual bands with equal weights,
including the coarsest field and finest residual. It retains fair CRPS rather
than changing the ensemble correction. See the [experiment plan](loss-ablation-plan.md)
for the filtering and mask conventions.

## Velocity initialization on one OM4 case

The target is the five-day OM4 state centered on 2013-12-29. Inputs are 19
surface-history frames ending on that date and shared ERA5. Interior truth is
used only for scoring. The last training target interval ends on 2013-08-04.
The published OM4 scoring store differs in version from the training store;
all arms use the same scoring store, grid, masks, normalization and random seed.
The input receipts match exactly except for the evaluation-script fingerprint.

Each score averages cosine-latitude-weighted errors over wet cells and gives
each of 19 depths equal weight. “Mean RMSE” is the square root of depth-averaged MSE of the eight-member mean.
“Member RMSE” is the square root of MSE averaged over members and depths, not
the arithmetic mean of member RMSEs. Fair CRPS is pointwise absolute error minus
the finite-ensemble-corrected spread term, then averaged over space and depth.
All velocity entries are in m/s; lower is better.

| Model / training loss | u mean RMSE | u member RMSE | u fair CRPS | v mean RMSE | v member RMSE | v fair CRPS |
|---|---:|---:|---:|---:|---:|---:|
| A control: MSE, joint | 0.04539 | 0.04539 | 0.02302 | 0.03532 | 0.03532 | 0.01794 |
| A: MAE, joint | 0.06888 | 0.06888 | 0.03247 | 0.05474 | 0.05474 | 0.02572 |
| B control: fair CRPS, joint | 0.06848 | 0.07026 | 0.02702 | 0.05437 | 0.05599 | 0.02111 |
| B: per-member MSE, joint | 0.04516 | 0.04517 | 0.02271 | 0.03507 | 0.03507 | 0.01768 |
| B: fair CRPS + mean MSE, joint | 0.05901 | 0.05949 | 0.02516 | 0.04696 | 0.04742 | 0.02024 |
| B: multiscale fair CRPS, joint | 0.06705 | 0.06872 | 0.02723 | 0.05369 | 0.05497 | 0.02194 |
| B: fair CRPS, initializer only | 0.06359 | 0.07407 | 0.02112 | 0.05374 | 0.06095 | 0.01858 |

Compared with B's joint control, initializer-only training lowers mean u/v RMSE
by 7.1%/1.2% and fair CRPS by 21.8%/12.0%, while raising member RMSE by
5.4%/8.9%. Thus the improvement in distributional score does not mean that
individual velocity maps are closer to this target. Spatial coherence and
spectra are not measured by this table.

MAE also changes the deterministic tradeoff across fields: temperature RMSE is
0.562 versus 0.697 °C for A's control, while salinity RMSE is 0.485 versus
0.259. Velocity degradation is therefore not a uniform worsening of every
initializer output. The loss choice is sufficient to produce a velocity gap
without stochastic sampling, although this single case does not explain the
full cause of B's velocity errors.

## Loss and spread

B with per-member MSE reduces mean u/v RMSE by about 34%/36% relative to B's
fair-CRPS control. Member and mean RMSE are almost identical. The u ensemble spread RMS falls
from 0.01574 to 0.000446 m/s. This is the square root of member MSE minus
ensemble-mean MSE on the same weighted support; [all spread values](results/loss-ablation-partial/om4-initializer-spread-2000.csv)
are retained. Per-member MSE
penalizes dispersion around a common target, so this result does not establish
a useful stochastic ensemble. Fair CRPS plus ensemble-mean MSE produces an
intermediate result: mean u/v RMSE falls about 14% while some spread remains.

Multiscale fair CRPS changes mean u/v RMSE by only about 2%/1%, and worsens
pointwise velocity CRPS. It improves salinity mean RMSE (0.478 versus 0.544)
in the dataset’s salinity units,
but worsens temperature (0.647 versus 0.634 °C) and SSH (0.02965 versus
0.02687 m). Member spectra and maps have not yet been compared; these scalar
scores cannot establish whether multiscale training improves spatial structure.

## Observation forecast validation

The score below is the unchanged nine-origin blend of climatology-normalized
SST, SSH-derived velocity, EKE, shallow OHC and deep OHC errors. It scores
ensemble-mean forecasts through 30 days and monthly OHC, not initialization.
The blend is the arithmetic mean of five error-to-climatology ratios, with OHC
layers 0–700 and 700–2000 m. Origins are November 2013 through July 2014.
Lower is better. All rows use the same 2k budget and EMA weights.

| Model / training loss | Forecast blend |
|---|---:|
| A control: MSE | 0.94011 |
| A: MAE | 0.90524 |
| B control: fair CRPS | 0.93462 |
| B: per-member MSE | 0.93841 |
| B: fair CRPS + mean MSE | 0.96047 |
| B: multiscale fair CRPS | 0.95039 |

Initializer-only has no forecast score because its processor is frozen and
untrained. Across the same nine observation origins, its initializer SST RMSE improves from 0.4259 to
0.2496 °C and SSH from 0.02441 to 0.02247 m compared with joint B. Those
are reconstructions of supplied surface history, not future predictions.

A's MAE arm has the best forecast blend here despite poor velocity
initialization on the OM4 case. Restoring B's velocity reconstruction with MSE
leaves its forecast blend close to the original fair-CRPS control. These metrics
measure different tasks; this pilot does not identify one loss that improves
both interior reconstruction and forecast quality without reducing spread.

All five new arms completed 2k updates. The [forecast curves](results/loss-ablation-partial/forecast-curves.csv)
retain raw and EMA scores at 500, 1,000, 1,500 and 2,000 updates.

[Exact OM4 scores](results/loss-ablation-partial/om4-initializer-2000.csv),
[matched-input proof and source receipts](results/loss-ablation-partial/provenance.json),
[original A/B observation initializer summaries](results/loss-ablation-partial/control-obs-initializer-2000.json),
and [initializer-only observation summaries](results/loss-ablation-partial/initializer-only-obs-2000.json)
are retained with checkpoint fingerprints. The observation summaries use nine
origins and a different aggregation from the single-case OM4 table; compare
like cohorts and aggregation rules.
