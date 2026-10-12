<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# First-wave execution record

Launched 2026-10-11 EDT / 2026-10-12 UTC. All three new arms completed 70 epochs, and all four common annual
evaluations are complete. The [annual comparison report](RESULTS.md) includes plots and numeric artifacts.
Continuous drift/structure diagnostics are queued following successful qualification; other remaining report
work is listed below. This is an experiment/results record, not intended to merge.

## Runs

| Arm | Beta job | W&B | Source | State at bring-up |
| --- | --- | --- | --- | --- |
| A: existing dynamic MSE | 210664 | [uaei2vbr](https://wandb.ai/ocean_emulators/default/runs/uaei2vbr) | `a20c44484` | Existing final raw and EMA checkpoints verified; no retraining |
| B: plain MSE | 212287 | [lpqa4bfi](https://wandb.ai/ocean_emulators/default/runs/lpqa4bfi) | `d87def314` | Training; online tracking verified |
| C: marginal almost-fair CRPS | 212288 | [mqzdnh28](https://wandb.ai/ocean_emulators/default/runs/mqzdnh28) | `d87def314` | Training; online tracking verified |
| D: structured almost-fair CRPS | 212295 | [k9973p2u](https://wandb.ai/ocean_emulators/default/runs/k9973p2u) | `338ae95d4` | Training; online tracking verified |

All new arms use four B200 GPUs per job, seed 15, 70 epochs, effective batch 32 forecast contexts,
InstanceNorm, bfloat16, existing normalization, Rust loading, and `tau_hfds` (no anomalies). C/D have four
training members per context. The backbone has 84,002,154 parameters; C/D add 13,280 noise-projection parameters.
Two inputs and two outputs per call, with four autoregressive calls, give eight five-day targets per example.
At handoff, B had reached epoch 2 and written raw/EMA checkpoints; C had completed its first training epoch;
D had completed multiple training microbatches. All three had finite training losses and online run URLs.

Allocated wall limits are 12 hours for B and 18 hours each for C/D. The first-wave envelope is approximately
200 GPU-hours including evaluation, with up to eight GPU-hours for qualification. Wall limits are ceilings,
not predicted costs; monitor actual elapsed allocation and progress. Do not shorten the 70-epoch exposure
silently if runtime becomes a problem. Later seeds and experiment waves wait for review.

## Qualification and loss calibration

Job 212277 exercised all original arms on four B200s: three optimizer updates, one-call validation,
two 90-day validation rollouts, and checkpoint writing. Job 212292 repeated D with training-only coefficient
calibration. It used a full production Beta node for a short check because a whole `beta_test` node was
unavailable. Both qualification jobs completed successfully.

| Arm/probe | Job | Peak allocated memory, rank 0 | Outcome |
| --- | --- | --- | --- |
| B | 212277 | 13.16 GB | Finite updates, validation and checkpoint |
| C | 212277 | 39.81 GB | Finite updates; nonzero ensemble spread |
| D, original 0.1/0.1 auxiliaries | 212277 | 54.30 GB | Numerically finite, but difference gradients dominated |
| D, calibrated auxiliaries | 212292 | 54.30 GB | Finite updates; nonzero spread; frozen coefficients |

The calibration used 16 **training** contexts at initialization: the first batch on each rank, a subset of
the 64 training contexts used to estimate transformed-target RMS scales. Measured output-gradient norms were:

| Term | Unweighted objective | Output-gradient L2 norm | Frozen coefficient |
| --- | --- | --- | --- |
| Grid CRPS | 0.427931 | 0.000147616 | 1 |
| Spatial block scores | 0.904904 | 0.000620485 | 0.02379038743674755 |
| Vertical/temporal difference scores | 25.709143 | 0.068039738 | 0.00021695526083931327 |

Each auxiliary is capped at 10% of the grid term's initial output-gradient norm. These are gradients with
respect to the output, not the network parameters. The coefficients stay fixed for the full D run. A
nonzero spread after three updates only verifies that sampling works; it says nothing yet about calibration.

Ten local tests passed, including CRPS formula/gradients, masked block averages, depth boundaries,
checkpoint-preserved noise, matching backbone initialization, member history, checkpoint round trips,
exact plain-MSE parity, and ensemble metric arithmetic. Scoped pre-commit checks passed.

## Data and artifact locations

Campaign root on Beta:

```text
/projects/ny/lz1955/multiscale/jrusak/runs/score-samudra-1deg-20261011
```

- `sources/<full Git commit>/`: immutable source trees, each with a `COMMIT` file.
- `configs/train-{B,C,D}.yaml`: complete launch configs. `outputs/train-*/config.yaml` is the saved runtime config.
- `outputs/train-*/saved_nets/`: raw, EMA, latest and validation-selected checkpoints.
- `outputs/train-*/data-manifest.json`: source bounds, variables, forcing, normalization values, coordinates,
  mask fingerprint and upstream preprocessing attributes.
- `outputs/train-*/{train,validation}-origins.csv.gz`: all eligible windows and target timestamps, with runtime
  assertions that fitting/validation targets are disjoint.
- `outputs/qualification-*/qualification.json`: raw resource and validation checks, not trained-model results.
- `outputs/qualification-D-calibrated/{structured-scales,auxiliary-calibration}.json`: frozen scales, sampled
  training indices, measured loss/gradient magnitudes, and coefficients used for D.
- `logs/` and `allocations/`: Slurm output/error logs and resource records.

The shared data directory is
`/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg`, containing `OM4.zarr`, `OM4_means.zarr`
and `OM4_stds.zarr`. A's former home path is a symlink to this project directory. The store has 4,745 times
on a 180 x 360 Gaussian grid, with upstream preprocessing recorded at commit `42851c09` and
`--skip_spatial_filtering`. Existing normalization is used as requested.

Observed training targets span 1975-01-13 through 2013-10-05; validation targets span 2013-10-15 through
2014-10-05 (all at 12:00). The shared boundary timestamp is a validation **input**, not a validation target.
The held-out source is configured separately for 2014-10-10 through 2022-12-24.

The container digest is pinned in `train_beta.sbatch`. The ARM64 Rust extension is copied from the successful
existing control campaign and its import was verified inside the container. New B/C/D model and training code
is experiment-local; the core `src/samudra` tree remains the pulled main snapshot `6b88f2c02`.

## Monitoring and recovery

Beta login and compute nodes timed out connecting to the external callback receiver. The first B/C launch
attempts, 212283 and 212284, stopped in that notification preflight before model training and were cancelled;
held job 212285 was also replaced. Their registrations were revoked. Their logs and allocations are retained.
Training now has no outbound-notification startup dependency.

Qualification consumed 1.028 allocated GPU-hours (643 and 282 seconds on four GPUs). The cancelled notification
preflights consumed another 0.307 GPU-hours; the held D attempt consumed none.

Durable one-hour reminders on the originating host wake the conversation to inspect Beta over SSH. On every
wake, check Slurm state, current epoch/loss, W&B syncing, checkpoint progress, and allocated GPU-hours; rearm
while active, and retire registrations once jobs finish. Connectivity to HTTPS port 443 was restored and
verified from the login node and all three compute allocations on 2026-10-12. The already-running training
scripts retain reminder-based follow-up; new evaluation jobs attach completion callbacks with hourly fallback.
Resume recoverable interruptions from the latest
completed epoch with the same immutable source and config.

All training had completed by 12:16 UTC on 2026-10-12:

| Arm | Beta job | Epochs | Exit code | Allocated wall time | Allocated GPU-hours |
| --- | --- | --- | --- | --- | --- |
| B | 212287 | 70 | 0 | 9:34:32 | 38.30 |
| C | 212288 | 70 | 0 | 10:51:41 | 43.45 |
| D | 212295 | 70 | 0 | 10:02:06 | 40.14 |

Final raw and EMA checkpoints were written for each arm, and each W&B run reported successful synchronization.
All three training reminders were retired. New training consumed 121.89 allocated GPU-hours; A was reused.
These completion checks do not establish a held-out ranking.

## Annual evaluator qualification

Job **212400** was submitted on 2026-10-12 using evaluator source `0ed45a791` and B's final EMA checkpoint.
It is a short four-GPU qualification with a 20-minute wall limit: four fixed held-out origins, each rolled
forward 360 days, distributed one origin per GPU. It was moved to `beta_test` / `test` while pending to use
the short-test queue. It completed successfully in 5:50 (0.389 allocated GPU-hours). Output is written beneath
`outputs/eval-qualification-B-212400/`; the run must produce all 2,156 per-channel/lead rows and `COMPLETE.json`.
The evaluator requires epoch 70, loads model weights strictly, checks predictions for nonfinite values,
records checkpoint/source/config provenance, and requires online W&B. Ten local tests and scoped pre-commit
checks passed. A verified callback credential is attached to the batch EXIT hook, with a one-hour reminder
covering pending jobs, node loss, and failed notification delivery.

All 2,156 rows were read back: the origin/lead/channel keys are unique, every metric is finite, the seven lead
times and four start dates match the manifest, deterministic spread is zero, CRPS equals MAE, and each
lead/channel rank histogram sums to four origins. The checkpoint reports epoch 70. W&B synchronized at
[he6t0nab](https://wandb.ai/ocean_emulators/default/runs/he6t0nab), and the completion callback was received.
This qualifies execution and data alignment, not model quality; no scientific comparison is claimed from
this short check.

## Common evaluation submissions

The following jobs use the qualified evaluator source `0ed45a791` and final EMA checkpoints. They were
submitted and released on 2026-10-12 with verified callback credentials, EXIT hooks, and one-hour reminders.

| Arm/task | Beta job | Partition | Wall limit | Terminal state |
| --- | --- | --- | --- | --- |
| A: common annual forecasts | 212403 | beta | 1 hour | Completed, exit 0; 2:36 |
| B: common annual forecasts | 212404 | beta | 1 hour | Completed, exit 0; 2:32 |
| D: four-origin, 16-member qualification | 212405 | beta_test | 30 minutes | Completed, exit 0; 2:55 |
| C: common annual forecasts, 16 members | 212406 | beta | 1 hour | Completed, exit 0; 2:49 |
| D: common annual forecasts, 16 members | 212407 | beta | 1 hour | Completed, exit 0; 2:52 |

Each allocation has four B200s. The A/B full evaluations use every prespecified complete 360-day origin;
D's qualification checks ensemble memory use, member evolution and probabilistic outputs before full C/D
evaluation. Job 212406 was submitted with an `afterok:212288` dependency, now cleared following C's successful
training completion. The evaluator also rejects any checkpoint earlier than epoch 70. Report diagnostics
remain pending completion of the common evaluations.

D's qualification job 212405 completed successfully in 2:55 (0.194 allocated GPU-hours), with W&B synchronized
at [e8q1s2v7](https://wandb.ai/ocean_emulators/default/runs/e8q1s2v7). All 2,156 rows are present, unique and
finite for four origins and 16 members. Rank histograms have 17 bins and sum to four for every channel/lead;
coverage values lie in [0, 1], spread is positive, and fair CRPS does not exceed empirical CRPS. The origins,
leads, variables, forcings and normalization match B exactly. Saved 5-day and 360-day target maps and masks
also match B exactly; all saved fields are finite, dry cells are zero, and members 0 and 1 differ. These are
execution/consistency checks, not a conclusion about calibration or physical realism. Its callback was
received and retired, and the full C/D jobs were then submitted with verified completion hooks and hourly
fallback reminders.

### Completed A/B annual evaluations

A and B each produced all **16,170 unique finite rows**: 30 common origins, seven leads and 77 channels.
The final epoch-70 EMA checkpoints loaded strictly. Source timestamps, physical-unit conversion, zero
deterministic spread, CRPS = MAE, rank-histogram totals, finite saved fields and zero dry-cell values all
passed read-back checks. Manifest origins, leads, channels, forcing and normalization match exactly; saved
target maps and masks also match exactly at all three saved leads. W&B synchronized at
[A: 36mor8bf](https://wandb.ai/ocean_emulators/default/runs/36mor8bf) and
[B: xcw8f76x](https://wandb.ai/ocean_emulators/default/runs/xcw8f76x).
Artifacts are in `outputs/eval-full-A-212403/` and `outputs/eval-full-B-212404/` beneath the campaign root.
Their completion callbacks were received and retired.

D's full evaluation also completed with all 16,170 unique finite rows and 16 independently evolving members.
Its manifest choices and saved targets/masks match A/B exactly; its source timestamps, finite saved fields,
zero dry-cell values, positive spread, bounded coverage, empirical/fair CRPS ordering and 17-bin rank totals
passed read-back checks. Distinct saved members verify diversity, not calibration. W&B synchronized at
[D: 5otm4ksi](https://wandb.ai/ocean_emulators/default/runs/5otm4ksi), and artifacts are in
`outputs/eval-full-D-212407/`. Its callback was received and retired. C subsequently completed with all
16,170 rows and successful W&B synchronization at
[djjy0rjx](https://wandb.ai/ocean_emulators/default/runs/djjy0rjx). Its callback was received and retired.
`analyze_score.py` audited all four complete evaluations together: unique declared keys, finite values,
physical-unit conversion, common origins/configuration, target/mask equality, rank totals and distinct stochastic
members. All checks passed. The generated report includes raw metric rows in gzip CSV parts, all-lead/channel
summaries, paired origin-year intervals, rank histograms and fixed-date maps. Two scanner findings in the
evaluation contract were verified as a dataset directory and the evaluator's Git commit; only those exact
file/value findings were marked non-secret in `.secrets.baseline`. Scanner rules remain unchanged.

**Time-coordinate detail:** the source uses a Julian calendar and mostly five-day spacing. Two intervals in
the held-out period are six days: 2016-12-28 to 2017-01-03 and 2020-12-28 to 2021-01-03, all at 12:00.
The evaluator's `lead_days` is the nominal number of five-day steps. Thus a nominal 360-day forecast is
72 source steps and spans 360 or 361 actual days. Nominal 20/40/90/180-day forecasts can likewise span one
additional day. Every scored target matches the intended source index, and A/B use identical timestamps;
this is a labeling distinction rather than a target-alignment error. Reports should label leads as nominal
days and retain actual origin/target timestamps from the CSVs.

## Continuous drift and structure diagnostics

Source `1c76b81ca` adds `diagnose_score.py` and `diagnose_beta.sbatch`. One four-GPU allocation assigns one
GPU to each arm; C/D each evolve four members without resets. All four online W&B runs must initialize before
forecasts begin. Predictions are scored at every output time, with physical moments, wet-pair zonal increments,
adjacent-depth differences, T/S and U/V covariance, temporal increments and fixed-region RMSE. Temperature,
salinity and kinetic-energy bulk means use binary wet masks, cosine latitude and nominal layer thicknesses;
they are explicit proxies, not exact inventories with OM4 partial-cell geometry.

Qualification job **212416** completed three calls/six states per arm in 1:59 (0.132 allocated GPU-hours),
exit 0. All four output markers, 462 unique score rows per arm, finite physical diagnostics, matching truth
statistics, zero dry cells, member diversity and mean/member consistency passed read-back checks. The A/B
first-step physical predictions also match their primary annual outputs. All four W&B runs synchronized:
[A](https://wandb.ai/ocean_emulators/default/runs/tjypjedp),
[B](https://wandb.ai/ocean_emulators/default/runs/wq1ddblu),
[C](https://wandb.ai/ocean_emulators/default/runs/tglclnho),
[D](https://wandb.ai/ocean_emulators/default/runs/f3pmxfjn).
Twelve local tests and scoped code checks passed. Its callback was received and retired.

Full job **212420** completed on production Beta with four B200s in **5:38** (0.376 allocated GPU-hours),
exit 0, using the same immutable source and epoch-70 EMA checkpoints. All four arms produced 598 states
through 2022-12-24, with 46,046 unique finite score rows each. Full read-back checks verified common timestamps,
channels, truth moments and bulk quantities, finite physical diagnostics, saved target/mask equality and
mean/member consistency. The final actual lead is 2,992 days (2,990 nominal days). W&B synchronized at
[A](https://wandb.ai/ocean_emulators/default/runs/ykv0ko06),
[B](https://wandb.ai/ocean_emulators/default/runs/vfhjnj23),
[C](https://wandb.ai/ocean_emulators/default/runs/2d6qhgkc),
[D](https://wandb.ai/ocean_emulators/default/runs/b6z4e9hg).
Outputs are in `outputs/continuous-full-212420/{A,B,C,D}/`; the completion callback was received and retired.
`analyze_continuous.py` audits and summarizes those outputs. Its physical structure, bulk and score assets
are included in `results/`. A post-hoc split by output-block phase identifies larger salinity increments
between calls than within calls, in both stochastic and deterministic arms. It does not establish a cause.

## Persistence and training-only seasonal references

Job **212427** completed on production Beta with four B200s in **5:10** (0.344 allocated GPU-hours), exit 0, from immutable
source `dcdda50e2`. It uses the common annual evaluator's 30 origins, seven leads, all 77 channels, masks
and existing normalization. P holds the latest true input fixed. S uses 73 seasonal bins fitted only to
1975-01-03 through 2013-10-05 source states. Julian-calendar centers are day-of-year 3, 8, ..., 363;
leap years have the same bins and a six-day year-end transition. All 4,745 source timestamps were checked
against this bin definition before submission. Counts, fitting bounds, source and normalization are written
to the reference manifest; 32,340 metric rows are required before marking completion.

Outputs are in `outputs/references-212427/`; W&B synchronized at
[6ul9j3mf](https://wandb.ai/ocean_emulators/default/runs/6ul9j3mf). The compute-node receiver check passed,
the completion callback was received, and its registration was retired. All 32,340 unique finite metric rows
passed read-back checks, including exact target agreement with A/B/C/D, deterministic metric identities and
physical-unit conversion. The fitting manifest contains 2,830 states, with 56 bins of 39 years and 17 of 38.
`analyze_references.py` reproduces the audit, plots and tables. Thirteen local model/numerical tests and scoped
pre-commit checks passed. These deterministic references did not train another neural model.

Remaining report work includes observation metrics where applicable. Checkpoint sensitivities are recorded
below. Proposed next-wave choices in `RESULTS.md` remain provisional. No additional
training seed or later wave has started.

## Remaining first-wave evaluations

Two production Beta jobs were submitted on 2026-10-12 from immutable source `46104028f`, each requesting four
B200s and a one-hour limit. Both have EXIT callbacks, compute-node receiver checks and hourly fallback
reminders. Login-node and compute-node credential checks passed. All checkpoint and export W&B runs
initialized online before forecasting.

| Job | Task | State | Output directory beneath campaign root |
| --- | --- | --- | --- |
| 212432 | A/B/C/D final-raw and validation-selected annual checkpoint sensitivity | Completed, exit 0; 16:35 | `outputs/sensitivity-212432/` |
| 212433 | Replay/export the continuous trajectories for secondary observation scoring | Completed, exit 0; 11:31 | `outputs/continuous-export-212433/` |

The sensitivity job evaluates all eight checkpoint choices with the common 30 origins, seven leads and
16 members for C/D. Primary final-EMA results remain fixed. Final checkpoints must still report epoch 70;
only the explicitly labeled validation-selected role accepts earlier epochs and requires stored selection
provenance. The evaluator records the actual epoch and selection identity, and all ranks wait for online
W&B before forecasting. The original control and B/C/D have different validation-selection policies;
this is a sensitivity check, not another fully matched training comparison.

All eight evaluations completed in 16:35 total allocation time (1.106 GPU-hours). Each has all 16,170 unique
finite metric rows, and all 129,360 rows passed common-target, normalization and physical-unit conversion
checks. Saved target/mask equality, finite fields, stochastic member diversity and rank-histogram totals
also passed. W&B synchronized for every evaluation. The callback was received and retired.

| Arm | Final raw W&B (epoch 70) | Validation-selected epoch | Selected W&B |
| --- | --- | ---: | --- |
| A | [dbf3qjij](https://wandb.ai/ocean_emulators/default/runs/dbf3qjij) | 53 | [cu5dwkre](https://wandb.ai/ocean_emulators/default/runs/cu5dwkre) |
| B | [8m5pfket](https://wandb.ai/ocean_emulators/default/runs/8m5pfket) | 48 | [wzk1fkab](https://wandb.ai/ocean_emulators/default/runs/wzk1fkab) |
| C | [m8yqhwhy](https://wandb.ai/ocean_emulators/default/runs/m8yqhwhy) | 65 | [hwhl5qvp](https://wandb.ai/ocean_emulators/default/runs/hwhl5qvp) |
| D | [75ohh0bq](https://wandb.ai/ocean_emulators/default/runs/75ohh0bq) | 63 | [rk4feqoy](https://wandb.ai/ocean_emulators/default/runs/rk4feqoy) |

The validation-selected weights are EMA: the saved configurations explicitly set `test_using_ema: true`,
and the source saves best-validation checkpoints inside `_test_context`. This was checked in both the
current trainer and A's historical source. Their selection identities differ: A uses 360-day validation
rollout RMSE; B/C/D use the shorter multi-lead policy described in `RESULTS.md`. Selection materially
improves deterministic point scores and narrows C/D's RMSE advantage over A; the CRPS advantage remains.
`analyze_sensitivity.py` writes the audited raw rows, summary tables, paired intervals and figures.

The export reuses the continuous evaluation's initialization, checkpoints, seed and four-member sampling.
It requires bitwise equality with the five saved diagnostic dates from job 212420 and reads back saved Zarr
times and selected physical fields before marking completion. A/B each export one deterministic trajectory;
C/D each export their mean plus all four members. The original diagnostics are retained. There are twelve
stores, approximately 144 GB uncompressed; project storage reported 32.59 TB available before submission.
Secondary observation scoring will use the published five-day products and original per-channel masks,
with member scores kept separate from scores of ensemble means, especially for nonlinear diagnostics.

The export completed in 11:31 (0.768 allocated GPU-hours), with W&B synchronized at
[A](https://wandb.ai/ocean_emulators/default/runs/ap16320s),
[B](https://wandb.ai/ocean_emulators/default/runs/0l5iguor),
[C](https://wandb.ai/ocean_emulators/default/runs/fljqj619) and
[D](https://wandb.ai/ocean_emulators/default/runs/lb2yxp7g). All twelve stores contain 598 times. The runner
verified their timestamps and read back the fixed-date fields before writing completion markers. A separate
post-job audit compared every decompressed score/global CSV, every structure array and every saved field
against job 212420: all were exactly equal. Checkpoint fingerprints, seeds, normalization, depth thicknesses,
origins and final dates also match. Its callback was received and retired. `observations.py` and
`observations_grace.sbatch` prepare the next CPU stage on Grace via Alpha, with exact-timestamp metadata
preflight before spatial scoring.

Thirteen model/numerical tests, eight existing Zarr-writer tests and scoped pre-commit checks passed before
submission. These are evaluation/export jobs within the first-wave allocation; no additional training is
authorized by their callbacks.

## Secondary observational evaluation

Grace job **132091** was submitted through Alpha using immutable source `057e9a61b`, with 32 CPUs, 384 GiB
RAM, no requested GPUs and a six-hour wall limit. Its output is `outputs/observations-132091/`. Slurm's
resource check accepted the request. The verified export from job 212433 is the only prediction input.

The job scores A/B deterministic trajectories plus C/D ensemble means and four members each against the
published five-day DUACS, OISST and ARGO-IAP products. It reapplies the original per-channel wet masks and
requires identical trajectory timestamps. Metadata-only exact-time checks cover both the primary 2015–2022
window and the full common span used by residual-variance diagnostics, before expensive interpolation/scoring.
Individual member metrics remain separate from metrics of the ensemble mean; no mean-field EKE is presented
as mean member EKE. Observation-product locations, attributes, actual shared dates, checkpoint/export
provenance and all model labels are retained in the job manifest.

Online W&B is required before scoring. An EXIT callback and hourly fallback reminder are registered; the
Alpha login-node receiver check passed before release. The script also checks from the compute node.
Compute-node bring-up and the product time-axis preflight remain to be verified. Relevant analysis code was
run on all completed artifacts, and scoped pre-commit checks passed before submission.
