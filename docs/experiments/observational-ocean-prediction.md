<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Improving observational ocean prediction with simulation data

Status: initial problem statement, 2026-09-12. Intended for human researchers and agents.
Unresolved data and evaluation details are listed as TODOs below; this is not yet a fully executable benchmark.

## Scientific objective

How well can we predict future observational ocean states using all the permitted data and compute available to us?
To what extent can model-generated data improve that performance, and which architectures and data strategies are
needed to realize the benefit? In particular, how much do higher-resolution OM4, LLC, and additional simulations
improve the same observational task relative to coarser data?

We have a strong prior that simulation data can help substantially. The research objective is to discover how to
achieve that improvement and measure its extent. The prior is motivation, not evidence of an achieved result.
The leading result should describe the best performance achieved using the available resources. Controlled
comparisons and ablations help explain that result and guide further improvement; they need not precede every
promising experiment or constrain the main search to a fixed experiment matrix.

The ultimate objective is observationally grounded quality after long evolution, with a **100-year rollout** as the
long-term target. This first version uses the implemented roughly eight-year rollout ending in 2022 and its
observational metrics. Short forecasts and model-data validation losses are useful development signals, but the
final evidence comes from the long rollout and observational evaluation.

Budgets, scheduling, agent permissions, and stopping policies belong in separate campaign instructions. This
problem statement does not impose the earlier velocity campaign's model choice, run sizes, or week-long budget.

## Target task and freedom to experiment

Initialize the target model **from observations**, then evolve the ocean continuously under prescribed forcing.
Use OM4 and/or ARCO-ERA5 forcing. The currently available OM4 forcing fields are fluxes; replacing these with
surface-state inputs is planned work. Fix the forcing product, variables, preprocessing, and timestamps for any
direct final comparison. Evaluation-period forcing is a prescribed input, not permission to train on the
corresponding ocean targets.

The target evaluation is a continuous rollout, without resets to future ocean observations. Initialization may
use a documented observation history at the start. The exact initialization products, time support, handling of
unobserved state, and start date still need to be specified. The existing OM4-initialized evaluation is useful
infrastructure and a diagnostic, but it does not by itself implement this observation-initialized target.

Experiments may change or replace the model entirely. Architectures, state representations, input/output
adapters, normalization, losses, training schedules, pretraining, multitask mixtures, auxiliary targets, history
lengths, temporal cadence, crops, spatial scales, and data transformations are open research choices. Other
initializations and tasks are welcome during training or as diagnostics, subject to the training-data rules.
There is no requirement to retain Samudra, its U-Net, shared weights, or velocity-only training.

Match observational coverage and spatial resolution as closely as practical for the final task. The products
have different grids, so this does not prescribe a single common native model grid. Smaller domains, coarser
outputs, and partial-variable tasks are useful experiments. Label their scope; assess overall improvement on
comparable coverage of the complete metric suite. Internal representations may differ, but outputs must support
the same physical observables and documented evaluation transformations.

## Data inventory

The following Torch directories and store metadata were checked read-only on 2026-09-12. This verifies their
presence and basic structure, not every chunk or their suitability for a particular training split. Preserve
source stores and record the provenance of derived datasets.

| Dataset | Torch location | Intended role and current qualification |
| --- | --- | --- |
| OM4 1° | `/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr` | Simulation training source; 180 × 360 horizontal cells |
| OM4 ½° | `/scratch/jr7309/data/om4_halfdeg_v4/OM4.zarr` | Simulation training source; 360 × 720 cells |
| OM4 ¼° | `/scratch/jr7309/data/om4_quarterdeg_v2/OM4.zarr` | Simulation training source; 720 × 1440 cells |
| DUACS, prepared | `/scratch/am16581/data/obs/duacs.zarr` | Observational evaluation and potential initialization; 0.125° grid, 598 five-day maps; velocity fields, no SSH |
| DUACS, raw daily velocity stores | `/scratch/am16581/data/obs_raw/duacs/` | Alternative preprocessing source; dated stores span October 2014–December 2022; training eligibility unresolved |
| OISST, prepared | `/scratch/am16581/data/obs/oisst.zarr` | SST evaluation reference; 0.25° grid, 598 maps |
| Argo/IAP, prepared | `/scratch/am16581/data/obs/argo-iap.zarr` | Ocean heat-content evaluation reference; 0.5° grid, 516 time records |
| CM4 | **TODO: provide Torch path and inventory** | Planned additional simulation source |
| LLC | **TODO: make available and provide Torch path and inventory** | Planned high-resolution simulation source |
| ARCO-ERA5 | **TODO: identify selected store, Torch staging/access path, and variables** | Atmospheric forcing source specified for the target task |

Each OM4 store has 4,745 time records. The three resolutions are regriddings of the same underlying simulation,
not three independent simulated trajectories. Their roots also conventionally contain `OM4_means.zarr` and
`OM4_stds.zarr`; verify the fitting period before reusing normalization statistics.

The observation products also have configured OSN copies under
`s3://m2lines-pubs/Samudra/v2026-07/obs/{duacs,oisst,argo-iap}.zarr`, with endpoint
`https://nyu1.osn.mghpcc.org`. The local Torch paths above are the starting point for this work.

OM4 and DUACS are the current named research sources; CM4 and LLC will be added when provided. OISST and Argo/IAP
are also required evaluation references. Their use as training sources is not implied by their availability.
New data releases should be recorded with variables, dates, grid, temporal averaging, units, and permitted role.

## Temporal separation and permitted data use

Follow the standard Samudra temporal separation across every training source and derived training task. The
reference OM4 configuration specifies:

| Role | Configured date range |
| --- | --- |
| Training | 1975-01-03 to 2013-10-05 |
| Validation | 2013-10-05 to 2014-10-05 |
| Existing rollout configuration | 2014-10-10 to 2022-12-24 |
| Primary observational RMSE window | 2015-01-01 to 2022-12-31, subject to product sampling |

These are the configured bounds; the shared training/validation boundary requires checking the loader's actual
window semantics. Ensure complete input/target windows obey the intended split. Eligible earlier simulation data
may be used when its dates and selection are documented; the 1975 start is a reference setup, not a prohibition
on more pre-cutoff data. Do not move the training cutoff forward to gain access to evaluation-period targets.

Keep validation/evaluation ocean states out of fitting model weights, normalization, climatologies, learned
initializers, and auxiliary reconstruction tasks. Permitted initialization observations at the rollout start
and prescribed forcing during evolution are explicit inference inputs. Record their use separately from training.

**Unresolved DUACS issue:** the local prepared archive starts on 2014-10-20, after the training cutoff and after
the reference rollout start. The daily archive also starts in October 2014. We therefore cannot currently assume
that these stores support observation-supervised training under the intended split. Resolve this TODO without
silently training on the evaluation years. Earlier observations or another explicitly agreed data arrangement
may be needed. The same dates must be reconciled with the observation-based initialization recipe.

Repeated validation use during research is acceptable for this problem statement. A separate hidden-test or
holdout-refresh policy is deferred; other test sets can be introduced later. Still distinguish evaluation-informed
development from independent confirmation when describing results.

## Final evaluation procedure

For a directly comparable final result, record and hold fixed the observation-based initialization protocol,
forcing inputs, rollout interval, observation product versions, physical metric definitions, and scoring rules.
Training choices remain open. Changes to those evaluation conditions produce a separately identified comparison.

1. Initialize from the agreed observations and run continuously through December 2022. Choose a common feasible
   start date once initialization is implemented; retain the 2015–2022 primary scoring window where supported.
2. Save predictions with physical variables, units, coordinates, depth geometry, and actual timestamps. The
   existing evaluator expects sea level (`zos`) and temperature profiles (`thetao`) for the suite below.
3. Apply the observation metric implementation pinned in the source references. It handles observation pairing,
   spatial transformations, temporal aggregation, and physical derived quantities. Native model resolution is
   recorded separately from the scoring grid.
4. Report all primary and annual metric rows, residual-variance diagnostics, and coverage. Compare matching
   periods, definitions, units, depths, and observation support. If native valid footprints differ, quantify
   coverage and provide a common-support comparison before claiming a gain from lower errors.
5. Present improvements and regressions across the suite, with uncertainty where available and a human-readable
   interpretation. Missing metrics or incomplete rollouts are incomplete evidence of overall performance.

The current suite provides these concrete objectives:

| Metric | Observation reference | Units | Desired direction |
| --- | --- | --- | --- |
| Surface geostrophic velocity vector RMSE | DUACS | m/s | Lower |
| Instantaneous surface eddy kinetic energy RMSE | DUACS | m²/s² | Lower |
| SST RMSE | OISST | °C | Lower |
| OHC per-area RMSE, 0–700 m | Argo/IAP | J/m² | Lower |
| OHC per-area RMSE, 700–2000 m | Argo/IAP | J/m² | Lower |
| SST residual-variance map RMSE | OISST | °C² | Lower |
| SST residual-variance pattern correlation | OISST | Dimensionless | Higher, toward 1 |
| Upper-700 m OHC residual-variance map RMSE | Argo/IAP | (J/m²)² | Lower |
| Upper-700 m OHC residual-variance pattern correlation | Argo/IAP | Dimensionless | Higher, toward 1 |

Use `observation_metrics.csv` as the detailed score record and `obs/metrics_table` in W&B when available. Primary
RMSE rows use `period_kind=primary_complete_years`; residual-variance rows use `full_overlap`. These are distinct
period definitions: residual variance currently uses the full shared rollout/observation overlap, not just the
2015–2022 RMSE window. Keep that overlap matched across candidates.

The current primary RMSE aggregation is the square root of the mean annual MSE, with equal calendar-year weights.
Its 95% calendar-year block-bootstrap intervals describe temporal variation, not seed-to-seed training variation.
Report repeated-run variation when measured; do not treat overlapping maps or cells as independent repetitions.
Residual-variance rows do not automatically inherit the RMSE confidence intervals.

The scorer derives geostrophic velocity from sea level rather than substituting a model's total ocean currents.
It calculates residual variance on the native grid before regridding the variance map. Keep these semantics
consistent. Report OHC bathymetry disagreement and partial-column coverage alongside OHC scores.

There is no fixed 3% improvement requirement, 10-day selection objective, mandatory seed count, or composite
reward in this problem statement. The score is a vector of physical metrics. Human assessment should make
tradeoffs explicit; quantities with unlike units should not be added into an unexplained scalar. Lower training
loss alone does not establish observational improvement, and an eight-year result does not establish 100-year
stability or skill.

## Best performance and supporting comparisons

The primary result is the strongest full-task performance achieved with all permitted data and the compute
allocated to the campaign. Record those resources approximately so the result has context. A larger successful
model or training run remains useful even when it exceeds the resources used by a reference experiment.

The following comparisons are useful diagnostics and explanations, not a required sequence or a prerequisite
for pursuing the strongest candidate:

- Compare with a pinned existing Samudra checkpoint and the best previously evaluated candidate, using the same
  target evaluation. Reevaluate references when feasible; label historical OM4-initialized scores separately.
- Train a candidate architecture with coarse OM4 alone and with additional data to estimate the contribution of
  that data. Comparable compute and target-data exposure answer different questions; report which is matched.
- Compare fine data with a coarsened version of the same source, with matched temporal sampling and extent, to
  distinguish fine spatial information from additional simulation diversity.
- Examine scaling with model capacity, training compute, data amount, resolution, and source mixture to identify
  the most productive next experiment.
- Use regional errors, spectra, variability, drift, short forecasts, initialization ablations, and forcing
  ablations to explain successes or failures and improve the next candidate.

Select concrete reference checkpoints and record their provenance when starting a campaign. A useful scientific
report can document mixed or negative outcomes while preserving the broader objective of finding better models.

## Per-experiment evidence

Maintain a short Markdown account linked to Git and machine-readable artifacts. A directory such as the following
is sufficient; file formats may follow existing tooling rather than a new schema:

```text
experiment-id/
  README.md                  # Question, method, results, interpretation, next step
  manifest.json              # Code/data/evaluation identities and artifact pointers
  config.yaml                # Resolved training and model configuration
  learning-curves.csv         # Or the existing JSONL/W&B record
  observation_metrics.csv    # Full evaluation output, including annual rows and coverage
  comparisons.csv            # Deltas versus named references, when computed
```

Record the experiment's hypothesis or performance objective; code commit and any patch; container/environment
identity; data versions, paths, preprocessing and split; architecture and training strategy; seeds and checkpoint
lineage; initialization and forcing; evaluator revision and settings; actual rollout completion; and approximate
hardware, compute, and elapsed time. Include coverage, uncertainty, metric tradeoffs, failures, and limitations.

Keep durable pointers to checkpoints and saved rollouts rather than committing large arrays to Git. A result
should be traceable from a reported number to the checkpoint, input data, and evaluation configuration. Mark
partial-task results and incomplete runs explicitly. An agent resuming work should be able to recover both what
was tried and why the next experiment was chosen.

## TODOs

- [ ] **Observation initialization:** specify products, available fields, observation history, start date,
  preprocessing, and how missing surface/subsurface state is inferred. The supplied DUACS store has velocities
  but no SSH; it does not directly supply the sea level required by the existing full-suite scorer. Specify
  training provenance for any learned initializer and separate initialization from later observational updates.
- [ ] **DUACS training and temporal split:** resolve the post-cutoff-only local archive. Verify complete window
  boundaries and the fitting periods of supplied statistics across sources. Do not inherit the velocity
  campaign's later training split into this task.
- [ ] **Surface-state forcing:** replace current OM4 flux inputs with the intended surface-state inputs. Identify
  OM4/ARCO-ERA5 variables, units, spatial/temporal transformations, coverage, and Torch access paths; version the
  resulting evaluation forcing protocol. Until then, label flux-forced experiments explicitly.
- [ ] **LLC availability:** make the selected dataset available on Torch and record its path, version, fields,
  dates, grid geometry, cadence, and permitted training period. Plan derived tasks without requiring global
  full-resolution evolution.
- [ ] **CM4 availability:** provide the same inventory and training eligibility information for CM4.
- [ ] **Executable final protocol and references:** finish observation initialization, pin evaluation data and
  forcing, verify the common scoring support, and identify concrete reference checkpoints. The metrics are
  implemented; this observation-initialized benchmark still requires that integration.
- [ ] **100-year evaluation:** define forcing over that horizon and how observational metrics, drift, variability,
  and stability will be assessed when a century of matched observations is unavailable. Keep this future protocol
  separate from the initial eight-year result.

## Source references

Requirements come from the discussion in Codex thread `01a09696-19e8-7ea3-86f8-a212d8a01eaa`, generalizing the
velocity-transfer effort in thread `01a09223-d781-70f1-adaf-bbeb2dd25e04`. This statement incorporates the later
decisions to initialize from observations, evaluate long rollouts, allow replacement architectures, and prioritize
best achievable performance over a prescribed ablation campaign.

Implementation references below were inspected at commit `55fe01dde9739b1967ecfe9b81a074ef0a61e19a`. These pin
the inspected definitions; they do not assert that every checkout has this implementation or that its metrics
are immutable forever. Record any evaluator update and rescore comparisons consistently.

- [OM4 temporal ranges and data configuration](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/om4.yaml)
- [Observation products and primary scoring window](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/obs.yaml)
- [Metric definitions and output table](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/report.py)
- [Observation pairing and physical quantities](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/comparisons.py)
- [Aggregation, coverage checks, and uncertainty](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/kernels.py)
- [Saved-rollout observation evaluation](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/run.py)
