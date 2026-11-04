<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Eight-GPU continuation and held-out maps

This is an experiment/results record, not intended to merge. The active
November 4, 2026 EMA-enabled training submissions extend the [initial point-temperature experiments](../initial-20k/README.md).
Both arms retain two observed quarter-degree ocean states, a moving 33×33 crop,
a depth-conditioned point query, and a single +5-day temperature prediction.
Inputs remain continuous. Classification uses ordinary cross-entropy over
2,048 absolute-temperature bins; it has no class-density loss weighting.

| Work | Torch job | Requested allocation | Dependency |
|---|---:|---|---|
| Classifier EMA qualification | 21861894 | 8 RTX6000, 10 minutes maximum | none |
| Regression EMA qualification | 21861897 | 8 RTX6000, 10 minutes maximum | classifier qualification |
| Classifier EMA continuation | 21861899 | 8 RTX6000, 5 hours | regression qualification |
| Regression EMA continuation | 21861901 | 8 RTX6000, 5 hours | classifier continuation |
| Map qualification | 21803870 | 1 H200, 15 minutes maximum | none |
| Initial-checkpoint maps | 21803874 | 4 H200, 3 hours per allocation | map qualification |

Dependencies require successful completion. RTX6000 jobs are eligible for both
`rtx6000_lzanna` and `rtx6000`; H200 jobs use preemptible capacity. All can requeue.
The table lists the active EMA training jobs, recorded in `ema-submission.json`.
They replace the previous non-EMA requests, cancelled before allocation.
Earlier receipts preserve superseded requests, including the routing change
in `dual-routing.json`. Training uses source `11212fb88`; the initial-checkpoint
map workflow uses `641922718`. Training is queued, and map qualification has
passed. No full map or new training result is claimed here.

The first map qualification, 21802680, failed after 45 H200-seconds because the
container lacked the requested `torchrun` executable. It cached Samudra temperature forecasts for all eight evaluation dates before
failing; no classifier or regression map inference ran. The
replacement launches through `python -m torch.distributed.run`, verified with
`--help` inside the same container. The failed attempt cost 0.0125 H200-hours.
Its dependent map job was cancelled without allocation. The four original
training jobs were replaced while still queued and consumed no GPU time.
Replacement map qualification 21803870 completed in 106 seconds and evaluated
3,344 matched points across all 19 depths. Eight-GPU training qualification
remains pending.

The initial continuation segments target roughly 40 allocated GPU-hours per
arm. The user clarified that this is a planning scale, not a strict overall
ceiling. The current runner accounts for preempted allocations and stops ten
minutes before its five-hour segment ends to allow validation and saving.
Further continuation can depend on data exposure and validation results. Qualification and map
compute are separate. Utilization telemetry is retained, with no low-utilization
stop rule. Each qualification exercises eight updates, distributed validation,
checkpoint writing, and agreement between the eight model replicas.

Each worker draws a fresh pool of 8,192 area-sampled, depth-uniform candidates
per minibatch. Within each depth, farthest-point Voronoi grouping defines up to eight groups
from observed temperature, salinity, velocity, temperature change, wet masks,
and geographic position. The [sampler](../../../src/samudra/skyline/coverage.py)
contains the exact feature construction. Half the proposal is uniform over pool occurrences; half is
uniform over groups, then their members. Each selected example receives
`1 / (8192 * q)` loss weight, where `q` is its proposal probability within
the pool. This recovers the original objective and its gradient in expectation
before gradient clipping: uniform dates and depths, with cell-area-proportional
wet locations within each depth. Weights are neither
clipped nor normalized within a batch. Future temperatures do not enter group
construction. There is no cumulative visited-state index. Eight workers each
use 32 examples, for a global batch of 256; AdamW restarts at a configured peak
learning rate of 0.0002. Initial classifier weights are its 20,000-update
checkpoint selected by minimum validation bin NLL; regression uses its
18,000-update checkpoint selected by minimum validation normalized MSE.

Both training arms now track an exponential moving average (EMA) of model
parameters in float32. Each EMA starts from its arm's selected initial raw
checkpoint; no historical EMA is reconstructed. After optimizer update `n`,
its decay is `min(0.999, (1+n)/(10+n))`, following the project's existing warm-up
rule. Every rank updates its own EMA from the synchronized model parameters.
The checkpoint stores the update count and all EMA parameters for exact resume.

At every validation event, raw and EMA weights are evaluated on identical dates,
locations, depths, and targets. W&B logs `validation/raw/*`, `validation/ema/*`,
and `validation/ema_minus_raw/*`, including RMSE and normalized MSE for all 19
depths and classifier bin/density NLL. Negative EMA-minus-raw error or NLL means
EMA scored better. `validation-comparison-STEP.json` contains both metric sets
and their differences. EMA weights are temporarily copied into the model;
raw parameters are restored even if evaluation or saving raises an exception.
The optimizer always continues from raw weights.

Checkpoint selection is independent for the two weight sets, using identical
criteria: minimum mean of the 19 per-depth normalized MSEs for best error,
and minimum mean of the 19 per-depth bin NLLs for best likelihood. Raw checkpoints
retain `best-error.pt` and, for classification, `best-likelihood.pt`; EMA adds
`best-error-ema.pt` and `best-likelihood-ema.pt`. Their `model` field contains
the selected weight set, identified by `weights_kind`. Resumable `latest.pt`
contains raw model/optimizer state, full EMA state, and both selection histories.
Qualification checks agreement between ranks for raw and EMA parameters.
The paired final validation has a ten-minute reserve within each five-hour
allocation. The existing initial-checkpoint map job still uses its original
raw checkpoints.

The first map report uses those existing checkpoints while continuation waits.
It compares regression, classifier mode, frozen Samudra epoch-62 outputs, and
OM4 targets on identical inputs and points. The requested calendar origins are
January 15, April 15, July 15, and October 15: 2015 supplies calibration and
2016 supplies assessment. Each request resolves to the first valid test origin
on or after that date. The July 2016 origin supplies the dense maps: every
fourth native row and column globally, plus every wet native point in the
North Pacific, Gulf Stream, and Agulhas spectral boxes. Other dates use 512
area-sampled wet points per depth. All 19 depths receive data and metrics;
figures show 2.5, 250, and 1,850 m.

The classifier scores both OM4 and Samudra using one predictive distribution.
The report includes bin NLL, bin perplexity, density NLL per degree Celsius,
50/80/90/95% interval coverage, and support failures. Density NLL can be
negative. Unconditional NLL/perplexity is reported as null when a candidate
falls outside finite support, alongside its failure fraction and conditional
in-support score. Null therefore denotes an infinite score, not missing work.

Error detection scores Samudra temperatures using bin NLL and density NLL
without future OM4 values. Out-of-support candidates tie at positive infinity.
Controls use Samudra–regression and Samudra–persistence disagreement. At each
depth, the 2015 area-sampled points set the 90th percentile of absolute
Samudra-minus-OM4 error and the 95th percentile of classifier density NLL on OM4.
Assessment reports AUROC and average precision for exceeding the error
threshold, rank correlation with absolute Samudra error, and the fraction of
Samudra squared error in the highest-surprise assessment decile (including
ties). An OM4 false alarm means its density NLL exceeds the calibration
95th-percentile threshold. Global assessment dates
receive equal weight; dense global points use native cell areas. Regional
metrics cover the single dense date and use global per-depth thresholds.
OM4 is a model reference, not an observation. Correlated points are not treated
as independent replicates for confidence intervals.

Validation before EMA submission: 30 skyline CPU tests and three existing
checkpoint tests passed. Checks cover EMA arithmetic and resume, restoration
after exceptions, unchanged optimizer state, and matched validation targets
for both objectives. Formatting, linting, typing,
schema validation, secret scanning, and license checks passed. Eight-GPU training qualification
and the full scientific assessment remain pending; the map qualification passed. The same pinned assessment
protocol is to be repeated after training. For that report, select regression
weights by validation error and classifier weights by validation bin NLL,
considering both raw and EMA checkpoints and recording each selected step and
weight type. Those selections can come from different updates. Claims about
EMA at matched training exposure use the same-update validation comparisons.
