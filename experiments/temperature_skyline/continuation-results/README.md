<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Continued point-temperature specialists

The classifier and regression continuations completed on November 5, 2026.
Both selected checkpoints use exponential moving average (EMA) weights.
The regression's validation normalized MSE fell 13.6% from its selected initial
checkpoint. Classifier validation bin NLL fell from 4.042 to 2.836 nats.
This is an experiment/results record and is not intended to merge.

| Validation metric (lower is better) | Selected initial raw | Final continued raw | Final continued EMA |
|---|---:|---:|---:|
| Regression mean-depth normalized MSE | 0.000389406 | 0.000343546 | 0.000336509 |
| Classifier mean-depth bin NLL, nats | 4.041757 | 2.901090 | 2.836074 |
| Classifier argmax mean-depth normalized MSE | 0.009771361 | 0.000373790 | 0.000354092 |

All entries use the same 9,728 sampled date–cell–depth validation cases
(9,726 distinct cases), from eight origins in October 2013–October 2014.
We verified identical coordinates and OM4 targets between the initial and
continued models and between raw and EMA validation. Each of the 19 depths
has 512 samples and equal weight in the reported mean. Normalized MSE divides
temperature MSE by the depth's training temperature variance. Classifier
point error maps the most probable bin to its temperature center. There are
2,048 asinh-spaced absolute-temperature centers per depth, with midpoint bin
boundaries and training-audited finite support. Bin NLL scores the full
predictive distribution.
The table's raw/EMA comparison holds the final optimizer update fixed.
EMA reduces final regression normalized MSE by 2.05% relative to raw weights.

Selection considered the original classifier update 20,000 and regression
update 18,000, plus every continued raw and EMA validation checkpoint.
Minimum mean-depth bin NLL selects classifier EMA at continued update 7,624;
minimum mean-depth normalized MSE selects regression EMA at update 7,711.
Qualification checkpoints and held-out map errors were excluded.
The classifier's minimum-error checkpoint is a different EMA checkpoint at
update 6,904 (normalized MSE 0.000348589); the confidence assessment uses
its minimum-NLL checkpoint. [Selection and target hashes](selection.json),
[all validation checkpoints](validation-curve.csv), and
[per-depth initial/selected scores](per-depth-selected.csv) preserve these choices.

| Arm | Engaging job | New optimizer updates | New accepted examples | Four-L40S allocation | GPU-hours |
|---|---:|---:|---:|---|---:|
| Classifier | 27944931 | 7,624 | 1,951,744 | 9h46m24s | 39.0933 |
| Regression | 27944932 | 7,711 | 1,974,016 | 9h46m25s | 39.0944 |

Accepted examples count training draws used in updates, including repeated
date–cell–depth cases; they are not distinct coverage counts.

Production used 78.1878 L40S GPU-hours. Two failed and two successful
engineering qualifications added 0.3567 L40S GPU-hours. These figures exclude
the earlier initial training and separate map inference. Each production arm
stopped training after its cumulative 9h45m threshold, then validated and
saved within its ten-hour allocation. Neither production job requeued.

The source is `4132facf98cc3a9a5ee9ad262ed0305c7080266f`. Run outputs are under
`engaging:/orcd/scratch/orcd/014/jrusak/temperature-skyline-20261104/recovery2`.
The [training design and migration audit](../stage8/README.md) describe the
task and data checks. The inputs contain two successive observed ocean states:
temperature, salinity, both horizontal velocities at all 19 depths, plus sea
level, wind stress, and heat flux. Each query uses a moving 33×33 native-cell
crop to predict one point's temperature five days later, conditioned on depth.
Samudra receives the same fields and times over the full global grid.
Four workers each
accumulated two microbatches of 32, giving effective batch 256. Coverage-based
sampling groups candidates by observed ocean state and geography, then
oversamples sparsely represented groups within each candidate pool. Exact
importance weights retain the original area-weighted objective in expectation. Inputs stayed continuous; classifier cross-entropy had
no density weighting. Each optimizer update also updated EMA once.

The first qualifications exposed rank-local elapsed-time learning rates.
Broadcasting rank zero's learning rate fixed the replica differences. Both
replacement qualifications passed eight updates. All four production replicas
ended with exactly identical raw and EMA tensor bytes. Each arm completed
19 paired validation passes with finite predictions and logged training losses;
EMA update counts matched optimizer update counts. The verification records
are [classifier](absolute-verification.json) and
[regression](regression-verification.json). The 34 skyline CPU tests and
commit checks passed before the continued-weight map evaluation was submitted.

This is one continuation per objective. More exposure, coverage sampling,
effective batch, a fresh optimizer schedule, and EMA changed together relative
to the initial runs, so the improvement cannot be assigned to sampling alone.
The same-update raw/EMA comparison isolates the choice of saved weights within
these trajectories. Validation results do not establish held-out calibration
or how well classifier surprise detects Samudra errors; those require the
separate, fixed 2015-calibration/2016-assessment map protocol. Fresh inference
with both selected EMA checkpoints is queued as Torch job 22009444; held-out
results are not yet available.
