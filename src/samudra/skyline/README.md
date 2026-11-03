<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Temperature point skyline experiment

This experiment predicts one wet cell's temperature five days ahead, conditioned
on depth, geography, and a local crop of the same two observed states and three
forcing fields used by the October 8 quarter-degree Samudra run. Targets and
inputs are five-day averages. All 19 depths participate. There is no rollout.

The baseline is `samudra-v2-quarterdeg-averaged-rust-i2-o1-ar8-211241`, source
commit `6b88f2c0204828ef77d41b3ab2733679615f53e8`. Its checkpoint and resolved
configuration are published under:

`https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/experiments/2026-10-08-samudra-v2-quarterdeg-averaged-rust-i2-o1-ar8-211241/`

The user authorized Torch's local `om4_quarterdeg_v2` fields as equivalent for
this task. The reader checks coordinates, time metadata, timestamps, and all
19 masks against the September averaged release. It uses that release's
normalization and cell areas, and records differences from the local statistics.
These checks do not claim a full payload comparison between releases.

## Model and objectives

One token contains a cell's 80 continuous field values, their asinh transforms,
and 19 wet masks. Learned projections and geographic Fourier features embed
these inputs. Two history embeddings distinguish the input dates. A separate
query contains the target depth, its physical depth in meters, the current
temperature, and its geography. Attention is dense across the observed tokens
and query. No future ocean values or candidate predictions enter the model.

The 33 by 33 crop has 2,178 observed tokens plus one query. Longitude wraps on
the global grid; locations beyond the polar edge are masked, without wrapping
the crop's own edges. Model sizes are configurable:

| Width / layers / heads | Parameters |
| --- | ---: |
| 1,024 / 16 / 8 | 210,067,456 |
| 2,048 / 32 / 16 | 1,657,518,080 |
| 3,072 / 48 / 24 | 5,462,233,088 |

`absolute_ce` uses 2,048 asinh-spaced centers per depth and unweighted hard
cross-entropy. `residual_ce` predicts the five-day change. `regression` predicts
absolute normalized temperature with MSE. There are no inverse-frequency class
weights. Sampling is uniform over depth and proportional to wet-cell area
within each depth. A global date is shared by a configurable number of samples;
the effective batch is not a count of independent ocean realizations.

## Qualification and training

The tracked `configs/temperature_skyline/qualification.json` is an eight-update
engineering check: one GPU, microbatch 1, accumulation 4, effective batch 4,
33 by 33 crops, and two validation dates with two points per depth. Its sampled
training-range support is provisional. It must not be interpreted as a trained
scientific skyline. Non-qualification training refuses sampled-only bounds.

`python -m samudra.skyline.run audit --config CONFIG --output AUDIT_DIR`
scans every training temperature record to establish bounds, with resumable
per-date receipts. `--audit-dates 8` creates a separately named pilot support.

`python -m samudra.skyline.run train --config CONFIG --output RUN_DIR`
trains with fixed optimizer updates, logs W&B and JSONL, and saves optimizer and
sampling state. Explicit resume requires an unchanged schedule. The latest
checkpoint and the best error/likelihood weights are separate. Full validation
uses all valid origins when `validation_dates` is zero. Persisted predictions
allow paired comparisons; regression and classification use identical locations.

Preemptible runs set `preemptible: true` and resume automatically when Slurm's
restart count is positive. An eviction before the first checkpoint restarts from
the original seed. The batch warning requests requeue; each restart restores the
last atomic checkpoint, including optimizer, RNG, and data sampling state.
Interrupted work since that checkpoint is repeated. Use `save_every: 1` for the
eight-update qualification. Torch routing uses the preemption Slurm comment,
`--requeue --signal=B:USR1@300`, and `REQUEUE_ON_USR1=1`.

The first training pair uses `small_absolute.json` and `small_regression.json`:
the 1,024-wide, 16-layer backbone, 20,000 updates, one GPU with microbatch 32 and
no accumulation, 500 warmup updates, and a peak learning rate of 0.0002. The
objectives are absolute-temperature categorical CE and scalar-temperature MSE.
Both use the full training-only support audit. Four updates share each decoded
two-history/target date window (128 sample draws); cells and depths are sampled
independently with replacement. Validation uses the same eight held-out dates
and fixed 64 sampled wet points per depth per date across runs, every 2,000
updates. Latest checkpoints are saved every 500 updates. These are initial
training runs; comparisons with the frozen Samudra baseline require its cache.

The utilization guard samples the assigned GPU by UUID every five seconds,
including validation and checkpoint time. After two minutes of startup and a
five-minute measurement window, a rolling mean below 60% requests a checkpoint
and a failed exit at an optimizer boundary. A failed telemetry query (including
its ten-second timeout) also stops the run. The 60% threshold leaves margin above Torch's reported 50% requirement;
the actual scheduler measurement is external. Telemetry is retained in
`gpu-utilization-restart-N.jsonl`. A stop does not automatically resubmit the
same inefficient configuration. Area-weighted location CDFs are cached to
avoid rebuilding a full-grid sampling table for every crop.

## Baseline and candidate scores

`python -m samudra.skyline.baseline --config CONFIG --baseline-config TRAIN_YAML
--checkpoint BASELINE_PT --output CACHE_DIR` loads the baseline with strict
state-dict matching and caches its first-step temperature outputs. Setting
`baseline_cache` in training configuration adds matched Samudra RMSE and candidate
log densities at the same validation locations.

`python -m samudra.skyline.score --checkpoint MODEL_PT --candidates INPUT.jsonl
--output SCORES.jsonl` scores rows containing `origin`, `row`, `col`, `depth`, and
`temperature_c`. The scorer loads only the observed history, including when the
last available record has no future truth. It reports bin log probability and
log density per degree C. Out-of-support values are explicitly flagged and have
null log scores (zero density), rather than silently becoming edge-bin values.

Raw bin likelihoods across different discretizations are not directly comparable.
Physical-unit density includes bin width and the normalization Jacobian.
Threshold calibration and false-alarm evaluation remain separate scientific
steps. Pointwise scores describe conditional plausibility and are not joint
spatial-field likelihoods.

## Provenance and checks

The Slurm harness verifies a commit archive's SHA-256 and extracts it into
node-local storage mounted read-only. Later workspace edits cannot change a
queued job. Runs record source commit, archive hash, source metadata hash,
normalization, split origins, configuration, model size, and hardware.

Run `pytest tests/test_skyline.py` for CPU checks of split/cadence handling,
periodicity and masking, target isolation, density normalization, backward
passes, resumed sampling, candidate scoring without future truth, and matched
validation locations. GPU qualification, full baseline evaluation, calibration,
and scientific comparisons require their own execution evidence.
