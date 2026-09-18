<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Predicting the ocean interior from surface observations

Status: research direction revised 2026-09-17. Intended for human researchers and agents.
The task and candidate learning strategy are described below; dataset preparation and the exact evaluation
protocol remain TODOs. This document does not claim an implemented benchmark or demonstrated skill.

## Scientific objective

How well can we infer the ocean interior from surface observations and predict its future evolution, using all
the permitted data and compute available? To what extent does simulation data improve prediction of future
observed interiors, and which architectures and data strategies realize that benefit? In particular, what do
finer-resolution OM4 data, LLC, and additional model sources contribute beyond coarser data?

We have a strong prior that simulation data should help substantially. The objective is to discover how to
achieve that improvement and measure its extent. Lead with the best performance achieved; controlled comparisons
help explain the result and guide further experiments without imposing a rigid experiment sequence.

Take as a working premise Laure's concern that prescribed atmospheric forcing makes surface prediction too easy
to serve as the main scientific test, while removing forcing limits useful evolution. Keep atmospheric forcing
and make **future observed interior temperature and salinity the primary evaluation targets**. Surface skill is
supporting evidence. We do not need to resolve this premise through a forcing-ablation campaign first.

Historical ocean reconstruction and eventual prediction over decades remain the broader motivation. The immediate
task is repeated forecasts at monthly to seasonal leads. A continuous eight-year rollout and a 100-year rollout
are possible later diagnostics and extensions, not the acceptance criteria for this task. Short-lead success
alone will not establish long-term stability or climate skill.

Budgets, scheduling, and agent execution instructions belong in separate campaign instructions.

## Target task

At a forecast origin `t`, use a documented history of surface observations ending at `t` to initialize an ocean
state. Evolve that state under prescribed atmospheric forcing and predict interior temperature and salinity at
future times. Evaluate against profiles from held-out future periods.

- **Initialization inputs:** observed surface fields and their history, masks, coordinates, and static geometry.
  SST and sea level are candidate fields; the exact products, history length, and available variables must be
  pinned. Observed interior profiles are training labels and evaluation targets, not required inference inputs.
- **Forcing:** OM4 and/or ARCO-ERA5 atmospheric inputs, with a common documented protocol for comparisons.
  Realized future atmosphere defines a conditional hindcast; it does not establish a forecast with unknown
  future atmospheric conditions. Current OM4 forcings are fluxes; surface-state forcing is a TODO.
- **Predictions:** an evolving surface and interior state with temperature/salinity available at scoring locations
  and depths. Architecture, internal representation, spatial grid, and additional state variables remain open.
- **Forecast boundary:** no surface or interior observations after `t` enter a forecast as ocean-state inputs.
  New forecast origins can use newly available surface observations. These repeated forecasts are distinct from
  one continuously evolving trajectory.
- **Initial lead proposal:** evaluate roughly one month ahead, then three and six months. Exact lead windows,
  origins, domain, depth range, and aggregation weights must be fixed before comparing results.

Monthly lead time does not imply a monthly mean target. Individual profiles have observation times. Prefer
predictions at those times, for example through a transition conditioned on the exact lead or a defined temporal
interpolation protocol. A transition from January 1 directly to January 29 can be one learned model application;
it need not be 28 daily steps. A monthly transition rolled out for six months requires six coarse steps.

If a model instead predicts monthly averages, explicitly define how those averages are compared with profile
samples and quantify the mismatch caused by unresolved temporal variability. Do not silently equate an
instantaneous profile, a five-day mean, and a monthly mean. Pretraining targets must also document their temporal
support; a five-day-averaged simulation archive cannot supply instantaneous truth simply by changing its labels.

## Candidate learning strategy

The starting proposal has two separately pretrained components followed by joint fine-tuning. It is a research
strategy, not a restriction on competing architectures.

1. **Surface-to-interior initialization.** Train an initializer on paired surface histories and full ocean states
   from eligible OM4 and/or GLORYS data. It estimates the unobserved interior and combines it with the surface to
   form an initial state. Allow observation-like masks and source adaptations during training. A surface history
   may not identify a unique interior; ensembles or probabilistic representations are valid choices.
2. **Coarse evolution.** Train a dynamics model on full-state transitions from the same permitted sources,
   conditioned on forcing and the chosen time interval. It evolves interior and surface together. Include the
   state needed for useful evolution; temperature and salinity need not be the only internal variables.
3. **Connect the components.** Train or validate evolution from the initializer's inferred states, rather than
   assuming performance from exact simulation states transfers unchanged. This exposes the dynamics model to
   initialization errors before observational fine-tuning.
4. **Joint observational fine-tuning.** Initialize from observed surface history and supervise predicted surface
   fields and interior profiles in the training period. Interior observations remain labels. Sparse profile
   losses can update both components through differentiable sampling of predicted fields; they do not require
   filling an observed global interior grid. Simulation replay and auxiliary interior reconstruction can help
   preserve physical structure. Loss weights and freezing schedules are experiment choices.

Surface-only fine-tuning can be an experiment, but improved surface predictions do not demonstrate improved
interiors. The primary profile evaluation applies regardless of which losses are used in training. Likewise,
GLORYS agreement measures agreement with a reanalysis, not independent observational success.

Architectures, normalization, model size, source mixtures, additional training tasks, temporal cadence, crops,
and spatial resolution are open. Models can be replaced entirely. Match observational support where practical,
and compare candidates on a common evaluation footprint rather than prescribing one internal grid.

## Data and access

Torch paths below were checked on 2026-09-12 for presence and basic metadata, not re-audited for this revision.
They are access pointers, not assertions that each store already supports this task or covers its training period.
Record versions, fields, units, spatial and temporal support, masks, source ancestry, and permitted dates for every
selected dataset and derivative.

| Dataset | Access pointer | Role and qualification |
| --- | --- | --- |
| OM4 1° | `/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr` | Initializer and dynamics pretraining |
| OM4 ½° | `/scratch/jr7309/data/om4_halfdeg_v4/OM4.zarr` | Finer-grid pretraining source |
| OM4 ¼° | `/scratch/jr7309/data/om4_quarterdeg_v2/OM4.zarr` | Finer-grid pretraining source |
| GLORYS | [Copernicus product](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description); Torch path TODO | Candidate paired surface/interior and dynamics pretraining source; assimilating reanalysis |
| DUACS, prepared | `/scratch/am16581/data/obs/duacs.zarr` | Surface data; inspected store has velocities, no SSH, and starts in October 2014 |
| DUACS, daily velocity archive | `/scratch/am16581/data/obs_raw/duacs/` | Alternative preprocessing; inspected archive starts in October 2014 |
| OISST, prepared | `/scratch/am16581/data/obs/oisst.zarr` | SST input/target candidate; check training-period availability and averaging |
| Argo/IAP, prepared | `/scratch/am16581/data/obs/argo-iap.zarr` | Monthly gridded T/S; possible training supervision and large-scale OHC diagnostics |
| Individual Argo profiles | [Argo access](https://argo.ucsd.edu/data/data-from-gdacs/); Torch staging TODO | Candidate primary interior labels and evaluation observations |
| EN4 profiles | [Met Office EN4](https://www.metoffice.gov.uk/hadobs/en4/); Torch staging TODO | Candidate broader collection of interior profiles; overlaps Argo |
| LLC and CM4 | Selected releases and Torch paths TODO | Additional simulation sources when available |
| ARCO-ERA5 | Selected store, fields, and Torch access TODO | Atmospheric forcing |

The three OM4 archives are regriddings of the same underlying simulation, not independent simulated trajectories.
Each has 4,745 records. Distinguish the effect of retaining spatial detail from adding a different simulation or
higher-resolution native dynamics. Verify fitting dates before reusing supplied means and standard deviations.

Configured observation copies also exist under
`s3://m2lines-pubs/Samudra/v2026-07/obs/{duacs,oisst,argo-iap}.zarr`, using endpoint
`https://nyu1.osn.mghpcc.org`. The gridded `argo-iap` store is not an archive of individual Argo profiles.

### Interior observation choice

Use individual quality-controlled profiles to avoid making a spatially smoothed global reconstruction the sole
reference. EN4 provides both profile files and monthly objective analyses; select the profile product for this
purpose. Monthly packaging of profiles does not make the measurements monthly means.
[EN4 documentation](https://www.metoffice.gov.uk/hadobs/en4/)

For Argo, the initial evaluation proposal is delayed-mode adjusted temperature, salinity, and pressure, with good
QC flags and an explicit uncertainty policy. Retain original measurements, observation times, positions, and
identifiers alongside processed arrays. Pin the release and filtering rules.
[Argo profile guidance](https://argo.ucsd.edu/data/how-to-use-argo-files/)

EN4 incorporates Argo and other archives. Deduplicate observations and retain source identifiers if combining
collections; do not describe EN4 and Argo as independent validation datasets.
[EN4 product guide](https://www.metoffice.gov.uk/hadobs/en4/EN.4.2.2_Product_User_Guide_v1.0.pdf)

[OceanDepths](https://arxiv.org/abs/2608.16373) is a candidate preparation shortcut: it pairs satellite surface
products, EN4 profiles, and GLORYS. Inspect its weekly alignment, vertical interpolation, source overlap, and
split semantics before adoption. Its [public dataset](https://huggingface.co/datasets/ESA-philab/OceanDepths)
is not yet a verified Torch input for this task.

## Temporal separation and data provenance

Preserve the existing Samudra separation in time across pretraining, fine-tuning, auxiliary tasks, and fitted
preprocessing. The reference configuration has these bounds:

| Role | Reference dates |
| --- | --- |
| Training | 1975-01-03 to 2013-10-05 |
| Validation | 2013-10-05 to 2014-10-05 |
| Candidate held-out target years | 2015–2022 |

These are reference bounds, not a completed profile split. Resolve the shared train/validation boundary and verify
whole history/target windows, including averaging and interpolation support. Earlier eligible data are permitted;
do not move the training cutoff forward to accommodate a convenient archive. A test forecast may consume its
permitted pre-origin surface history as inference input without fitting on that history.

Keep held-out ocean targets out of weights, normalization, climatologies, learned initializers, and source-derived
training labels. GLORYS assimilates ocean observations: a held-out profile is not independent if its information
already entered a reanalysis used for training. Apply the temporal exclusion across all sources and audit product
time support around boundaries. Future atmospheric forcing remains a separately declared conditional input.
[GLORYS product description](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)

Resolve pre-cutoff surface coverage and the lack of SSH in the inspected DUACS store before implementing
observation-supervised training. Check whether retrospective surface products use observations beyond a nominal
forecast origin; pin either a causal input construction or explicitly label a retrospective reconstruction task.
A separate hidden-test governance policy remains outside this problem statement, but repeated evaluation-informed
development must be distinguished from independent confirmation.

## Evaluation and success criteria

The primary result is prediction of **unseen future interior observations from surface-initialized states**.
Initializer accuracy at lead zero is a separate diagnostic. A good reconstruction alone does not demonstrate
skill in evolution, and improvement confined to surface metrics does not satisfy the objective.

For each candidate, hold fixed the surface input protocol, forcing, forecast origins and leads, profile collection
and QC, depth/domain coverage, sampling operator, and aggregation. Save the matched forecast/observation records
so scores can be reproduced. Evaluation requires new profile matching and scoring; the existing gridded
observation evaluator does not implement this protocol by itself.

| Quantity | Role | Proposed reporting |
| --- | --- | --- |
| Future profile temperature error | Primary | RMSE and bias in °C by lead, depth band, and region; lower RMSE and bias magnitude |
| Future profile salinity error | Primary | RMSE and bias in the pinned salinity convention/units, with the same breakdowns |
| Skill versus reference predictions | Primary interpretation | Compare climatology and persistence of the inferred interior anomaly on identical samples |
| Profile reconstruction error at initialization | Diagnostic | Separate lead-zero T/S errors from future-lead scores |
| Surface SST and sea-level/velocity skill | Supporting | Observational errors with compatible time support |
| OHC, variability, drift, and stability | Supporting/longer-term | Gridded-product comparisons and physical diagnostics with coverage stated |

Define the profile sampling operator explicitly: spatial and vertical interpolation, pressure/depth conversion,
temperature/salinity conventions, time matching, and wet-cell treatment. Do not score missing depths as zero or
extrapolate below supported bathymetry. Preserve a reference to the original profiles when deriving filtered or
averaged diagnostics.

Aggregate so densely sampled casts, depths, or regions do not silently dominate. Pin depth bands and weights,
report profile counts and coverage, and use identical accepted observations for direct comparisons. Estimate
uncertainty with blocks that respect float/time correlations rather than treating every depth sample as
independent. Exact weighting and uncertainty procedures are TODOs, not existing implementation claims.

Less-smoothed observations retain variability that may be unresolved or unpredictable at a given lead. Report
that limitation and the model's spatial/temporal support. Sparse profile scores characterize performance where
observations exist; they do not independently establish global heat-content accuracy. Keep gridded IAP/OHC
comparisons as complementary large-scale diagnostics.

There is no prescribed scalar reward or percentage threshold. Report the vector of T/S scores, uncertainty,
coverage, and tradeoffs. Use best achievable performance as the headline; observation-only training, coarse-only
versus richer simulation training, and compute/data scaling are useful supporting comparisons for attributing
improvement. Their experiment order and budgets remain flexible.

## Per-experiment evidence

Keep a short Markdown account linked to Git and machine-readable artifacts. Record the question, code commit,
resolved configuration, checkpoint lineage, model and training strategy, source versions/paths, date splits,
preprocessing, forcing, input history, forecast origins/leads, evaluator revision, and approximate compute.
Include results, uncertainty, coverage, failures, interpretation, and the next proposed experiment.

Save a manifest, per-profile prediction/observation matches (including source IDs and QC), aggregated metrics,
and deltas versus named references in CSV, JSON, Parquet, or existing equivalent formats. Keep durable pointers
to checkpoints and output arrays rather than committing them to Git. Label incomplete runs and partial-domain
results. A human or agent should be able to trace a score back to its model, inputs, and accepted observations.

## TODOs before an executable campaign

- [ ] **Surface initialization:** choose products, variables, history, masks, and geometry; obtain eligible
  pre-cutoff coverage and SSH if selected. Pin how inferred interior state connects to the dynamics model.
- [ ] **Profile dataset:** choose Argo, EN4 profiles, or a documented combination; stage on Torch and record QC,
  duplicates, versions, units, depth support, and date coverage. Assess OceanDepths as a preparation option.
- [ ] **Time target:** choose endpoint/exact-lead predictions versus means; specify matching and interpolation,
  and verify the temporal support available in simulation and observation training sources.
- [ ] **Evaluation contract:** pin forecast origins, one/three/six-month leads or alternatives, test dates,
  domain, depth bands, sampling operator, weights, uncertainty procedure, and reference predictors.
- [ ] **Splits and ancestry:** verify full window boundaries, fitted statistics, retrospective input support,
  and exclusions across raw observations, gridded products, GLORYS, and simulation-derived training tasks.
- [ ] **Surface-state forcing:** replace current OM4 flux inputs with intended atmospheric surface-state inputs;
  identify OM4/ARCO-ERA5 fields, transformations, time support, and Torch access. Label interim flux experiments.
- [ ] **Additional model data:** inventory GLORYS, LLC, and CM4 paths, fields, geometry, cadence, dates, and
  permitted roles. LLC availability remains an explicit dependency, not an assumed completed transfer.
- [ ] **Training and scoring implementation:** implement separate pretraining, connected fine-tuning, sparse
  profile losses, and matched-profile evaluation, or identify reusable implementations with verified semantics.

## Source references

This direction follows the discussion in task `01a09696-19e8-7ea3-86f8-a212d8a01eaa`, building on the original
velocity-transfer effort in `01a09223-d781-70f1-adaf-bbeb2dd25e04`, and the subsequent two-component model and
profile-evaluation discussion. The exact open choices above should be resolved in campaign instructions.

- [Meeting slides](https://docs.google.com/presentation/d/1WjVueoj0ibFlMIyf3RsEMsthqp7I_4-CTE-tJQtYz3I/edit)
- [Meeting notes](https://docs.google.com/document/d/1dttyhoUtmnU4bD-hl73mcfbHnaq7rTrdKJ0q23r5qc8/edit)
- [Literature review: ocean and atmospheric precedents](observational-ocean-literature-review.md)
- [Reference OM4 split and configuration](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/om4.yaml)
- [Existing gridded observation products](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/obs.yaml)
- [Existing gridded metric definitions](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/report.py)
- [Existing observation comparisons](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/comparisons.py)

The pinned code references describe reusable infrastructure, not the proposed profile benchmark. Record and
validate any implementation adopted by a campaign against its final evaluation contract.
