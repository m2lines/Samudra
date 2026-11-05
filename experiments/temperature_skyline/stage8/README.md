<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Four-GPU continuation and held-out maps

This is an experiment/results record, not intended to merge. The active
November 4, 2026 four-GPU EMA-enabled training submissions extend the [initial point-temperature experiments](../initial-20k/README.md).
Both arms retain two observed quarter-degree ocean states, a moving 33×33 crop,
a depth-conditioned point query, and a single +5-day temperature prediction.
Inputs remain continuous. Classification uses ordinary cross-entropy over
2,048 absolute-temperature bins; it has no class-density loss weighting.

| Work | Host | Job | Requested allocation | Dependency |
|---|---|---:|---|---|
| Classifier EMA qualification | Engaging | 27943836 | 4 GPUs, 20 minutes maximum | none |
| Regression EMA qualification | Engaging | 27943840 | 4 GPUs, 20 minutes maximum | classifier qualification |
| Classifier EMA continuation | Engaging | 27943843 | 4 GPUs, 10 hours | regression qualification |
| Regression EMA continuation | Engaging | 27943848 | 4 GPUs, 10 hours | classifier continuation |
| Map qualification | Torch | 21803870 | 1 H200, 15 minutes maximum | none |
| Initial-checkpoint maps | Torch | 21803874 | 4 H200, 3 hours per allocation | map qualification |

Dependencies require successful completion. Engaging training uses account and
QoS `mit_amf_advanced_gpu` on `mit_normal_gpu`, which permits four GPUs per user
and 48-hour jobs. The submitted GPU request was untyped; Slurm resolved it to four L40S GPUs.
The actual hardware is also recorded in run provenance. Torch H200 maps retain their preemptible routing. All GPU
jobs can requeue. The training source is `d5f88b950`; the map source remains
`641922718`. Submission commands and configs are in `engaging-submission.json`.
The four-GPU Torch requests in `accumulation-submission.json` were cancelled
before allocation. Earlier receipts preserve superseded requests.
Training qualification and the full map evaluation remain pending; map
qualification passed. No new training or full map result is claimed here.

Engaging reads the existing shared dataset
`/orcd/data/abodner/002/jrusak/om4_quarterdeg_v2`. All 80 required channels have
4,745 time chunks. Coordinates, timestamps, 19 masks, and required normalization
files match Torch byte-for-byte. Canonical metadata matches after removing the
compression setting: Engaging arrays are uncompressed; Torch uses Blosc/Zstd.
Decoded payloads match for 45 chunks spanning nine fields and five dates.
This is a sampled payload comparison. The original training-only bin-support
audit remains in `support-torch.json`; `support-engaging.json` binds those
bounds to the Engaging source receipt and preserves the original audit source.
No full training-support scan was repeated on Engaging. See
`engaging-data-verification.json` for the comparison scope and hashes.

The two selected initial checkpoints were copied through Torch's data-transfer
node and verified by SHA-256 on Engaging. The existing Engaging PhysicsNeMo
26.05 container uses Python 3.12.3, PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, and Zarr 2.18.7, matching
those runtime versions on Torch. Host Python user packages are disabled.
The actual launcher imports successfully in this container. GPU qualification
is still required for device compatibility, memory use, and throughput.
Run outputs, caches, and logs use
`/orcd/scratch/orcd/014/jrusak/temperature-skyline-20261104`.
The user authorized returning to Torch if Engaging encounters a blocker.
Only one active continuation chain is permitted across the two clusters.

The first map qualification, 21802680, failed after 45 H200-seconds because the
container lacked the requested `torchrun` executable. It cached Samudra temperature forecasts for all eight evaluation dates before
failing; no classifier or regression map inference ran. The
replacement launches through `python -m torch.distributed.run`, verified with
`--help` inside the same container. The failed attempt cost 0.0125 H200-hours.
Its dependent map job was cancelled without allocation. The four original
training jobs were replaced while still queued and consumed no GPU time.
Replacement map qualification 21803870 completed in 106 seconds and evaluated
3,344 matched points across all 19 depths. Four-GPU training qualification
remains pending.

The initial continuation segments target roughly 40 allocated GPU-hours per
arm. The user clarified that this is a planning scale, not a strict overall
ceiling. Each arm now requests four GPUs for ten hours, with 32 CPUs and
64 GiB host memory. Each arm has one cumulative ten-hour budget across
requeues, including preempted allocations. Training stops at nine hours and
forty-five minutes of cumulative allocation time, reserving the final fifteen
minutes for validation and saving.
Further continuation can depend on data exposure and validation results. Qualification and map
compute are separate. Utilization telemetry is retained, with no low-utilization
stop rule. Each qualification exercises eight updates, distributed validation,
checkpoint writing, and agreement between the four model replicas.

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
construction. There is no cumulative visited-state index. Four workers each
accumulate two microbatches of 32 examples, retaining an effective batch of 256.
Each microbatch loss is divided by two before backward; DDP synchronizes
gradients on the second microbatch. Gradient clipping, AdamW, and EMA each
run once per optimizer update. AdamW starts fresh. Its base learning rate is
0.0002, multiplied by a warm-up factor rising from 1/200 to 1 over the first
200 optimizer updates and by a cosine decay factor. Cosine decay follows
elapsed allocation time over the ten-hour segment, excluding the final reserve.
The training sample sequence changes with the worker count. Validation retains
the same eight dates and 9,728 date-cell-depth targets, now two dates per worker.
They lie in the October 2013–October 2014 validation split, at source origin
indices 4071, 4081, 4091, 4101, 4111, 4121, 4131, and 4142. They are separate
from the 2015 calibration and 2016 assessment dates used for maps.
Initial classifier weights are its 20,000-update
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
and minimum mean of the 19 per-depth bin NLLs for best likelihood. Each depth’s
normalized MSE divides temperature MSE by its training temperature variance
from the baseline release normalization; depths receive equal weight. Raw checkpoints
retain `best-error.pt` and, for classification, `best-likelihood.pt`; EMA adds
`best-error-ema.pt` and `best-likelihood-ema.pt`. Their `model` field contains
the selected weight set, identified by `weights_kind`. Resumable `latest.pt`
contains raw model/optimizer state, full EMA state, and both selection histories.
Qualification checks agreement between ranks for raw and EMA parameters.
The paired final validation uses that arm’s single fifteen-minute reserve
within its cumulative ten-hour budget. The existing initial-checkpoint map job still uses its original
raw checkpoints.

The first map report uses those existing checkpoints while continuation waits.
It compares regression, classifier mode, frozen Samudra epoch-62 outputs, and
OM4 targets at identical query points. Forecast origins, two observed states,
forcing fields, normalization, and the +5-day lead match. Samudra receives
the full global input grid; each specialist receives only its 33×33 crop. The requested calendar origins are
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

Validation before four-GPU submission: all 33 skyline CPU tests passed. A
four-process DDP test compares accumulated gradients, clipped AdamW updates,
and EMA parameters with a full-global-batch reference for both objectives,
using nonuniform importance weights. It also verifies that the intermediate
microbatch does not synchronize gradients. The earlier EMA change passed
three existing checkpoint tests. Checks cover EMA arithmetic and resume, restoration
after exceptions, unchanged optimizer state, and matched validation targets
for both objectives and validation shard changes. Formatting, linting, typing,
schema validation, secret scanning, and license checks passed. Four-GPU training qualification
and the full scientific assessment remain pending; the map qualification passed. The same pinned assessment
protocol is to be repeated after training. For that report, select regression
weights by validation error and classifier weights by validation bin NLL,
considering the selected initial checkpoint and continuation raw/EMA
checkpoints evaluated on the same 9,728 validation targets. Qualification
checkpoints are excluded. Record the selected source run, step, and weight
type; the initial checkpoint remains eligible if continuation worsens validation. Those selections can come from different updates. Claims about
EMA at matched training exposure use the same-update validation comparisons.
