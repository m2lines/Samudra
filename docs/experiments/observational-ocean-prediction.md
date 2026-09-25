<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Predicting ocean state from surface observations

Status: research direction revised 2026-09-18. Intended for human researchers and agents.
The task and candidate learning strategy are described below; dataset preparation and the exact evaluation
protocol remain TODOs. This document does not claim an implemented benchmark or demonstrated skill.

## Scientific objective

How well can we infer ocean state from surface observations and predict its future evolution, using all
the permitted data and compute available? To what extent does simulation data improve prediction of future
observed interior and surface fields, and which architectures and data strategies realize that benefit? What do
finer-resolution OM4 data, LLC, and additional model sources contribute beyond coarser data?

We have a strong prior that simulation data should help substantially. The objective is to discover how to
achieve that improvement and measure its extent. Lead with the best performance achieved; controlled comparisons
help explain the result and guide further experiments without imposing a rigid experiment sequence.

Keep **future observed interior temperature and salinity as primary evaluation targets**, and also predict
**daily sea surface height (SSH) and sea surface temperature (SST)** for observational scoring with the existing
metrics. Forcing consists of selected ERA5 atmospheric fields; OISST and DUACS are not forecast-time forcings.
In particular, SSH is a prediction target, not a prescribed input from the atmosphere. Surface prediction is a
meaningful part of this task because the future observed surface fields are not supplied to the model.

Pretraining on model outputs also supervises interior velocities. Retain these predictions during observational
fine-tuning to investigate whether the model can produce plausible velocity fields alongside the observed
quantities, even without gold observational interior velocity targets.

Historical ocean reconstruction and eventual prediction over decades remain the broader motivation. The immediate
task combines daily surface predictions with repeated interior forecasts at monthly to seasonal leads. A continuous
eight-year rollout and a 100-year rollout are possible later diagnostics and extensions, not the acceptance
criteria for this task. Short-lead success alone will not establish long-term stability or climate skill.

Budgets, scheduling, and agent execution instructions belong in separate campaign instructions.

## Target task

At a forecast origin `t`, use a documented history of surface observations ending at `t` to initialize an ocean
state. Evolve that state under prescribed atmospheric forcing and predict daily SSH/SST together with interior
temperature, salinity, and velocities. Evaluate the observed quantities against held-out surface products and
interior profiles; assess interior velocity plausibility separately.

- **Initialization inputs:** observed surface fields and their history, masks, coordinates, and static geometry.
  SST and sea level are candidate fields; the exact products, history length, and available variables must be
  pinned. Observed interior profiles are training labels and evaluation targets, not required inference inputs.
- **Forcing:** selected ERA5/ARCO-ERA5 atmospheric fields for observational forecasts, with compatible OM4
  forcing for model-data pretraining and a common documented protocol for comparisons. OISST, DUACS, and
  forecast-time ocean SSH/SST or observed ocean velocities are not forcing variables. Eligible pre-origin surface
  observations may initialize the state, and training-period future observations may serve as targets.
  Realized future atmosphere defines a conditional hindcast; it does not establish a forecast with unknown
  future atmospheric conditions. Current OM4 forcings are fluxes; surface-state forcing is a TODO.
- **Predictions:** daily SSH and SST, plus interior temperature/salinity available at scoring locations and depths.
  Retain interior horizontal velocity components (`uo`, `vo`) at the declared interior state times throughout
  pretraining, observational fine-tuning, and inference. Architecture, internal representation, spatial grid,
  and additional state variables remain open.
- **Forecast boundary:** no surface or interior observations after `t` enter a forecast as ocean-state inputs.
  New forecast origins can use newly available surface observations. These repeated forecasts are distinct from
  one continuously evolving trajectory.
- **Initial lead proposal:** evaluate interior profiles roughly one month ahead, then three and six months,
  with daily SSH/SST outputs through the forecast interval. Exact lead windows, origins, domain, depth range,
  and aggregation weights must be fixed before comparing results.

Monthly lead time does not imply a monthly mean target. Individual profiles have observation times. Prefer
predictions at those times, for example through a transition conditioned on the exact lead or a defined temporal
interpolation protocol. A transition from January 1 directly to January 29 can be one learned model application;
it need not be 28 daily steps. A monthly transition rolled out for six months requires six coarse steps.

Daily surface output is a separate requirement from the internal integration cadence. A model with coarse
interior transitions can use a daily surface decoder, direct predictions at daily leads, or another documented
output strategy. It must produce the daily SSH/SST fields needed for scoring without consuming post-origin ocean
observations. Daily outputs do not require backpropagation through a daily full-interior rollout.

If a model instead predicts monthly averages, explicitly define how those averages are compared with profile
samples and quantify the mismatch caused by unresolved temporal variability. Do not silently equate an
instantaneous profile, a five-day mean, and a monthly mean. Pretraining targets must also document their temporal
support; a five-day-averaged simulation archive cannot supply instantaneous truth simply by changing its labels.

## Candidate learning strategy

The starting proposal has two separately pretrained components followed by joint fine-tuning. It is a research
strategy, not a restriction on competing architectures.

1. **Surface-to-interior initialization.** Train an initializer on paired surface histories and full ocean states
   from eligible OM4 and/or GLORYS data. It estimates the unobserved interior and combines it with the surface to
   form an initial state, including temperature, salinity, and interior velocities. Allow observation-like masks
   and source adaptations during training. A surface history may not identify a unique interior; ensembles or
   probabilistic representations are valid choices.
2. **Coarse evolution.** Train a dynamics model on full-state transitions from the same permitted sources,
   conditioned on forcing and the chosen time interval. Supervise temperature, salinity, SSH/SST, and interior
   velocities against the source's available fields and time support. The model evolves interior and surface
   together and provides daily surface outputs for the target task; internal state cadence remains flexible.
3. **Connect the components.** Train or validate evolution from the initializer's inferred states, rather than
   assuming performance from exact simulation states transfers unchanged. This exposes the dynamics model to
   initialization errors before observational fine-tuning.
4. **Joint observational fine-tuning.** Initialize from observed surface history and supervise daily SSH/SST and
   interior T/S profiles in the training period. Interior observations remain labels. Sparse profile losses can
   update both components through differentiable sampling of predicted fields; they do not require filling an
   observed global interior grid. Keep predicting interior velocities while optimizing the observed quantities.
   There are no gold interior velocity observations in this proposal, so observational batches have no direct
   interior velocity target; missing labels are not zeros. One option is to mix eligible model-output examples
   into fine-tuning, retaining supervised velocity reconstruction/forecast losses to discourage forgetting.
   Simulation replay, auxiliary tasks, loss weights, mixing ratios, and freezing schedules are experiment choices.

Surface-only fine-tuning can be an experiment, but improved surface predictions do not demonstrate improved
interiors. The primary profile evaluation applies regardless of which losses are used in training. Likewise,
GLORYS agreement measures agreement with a reanalysis, not independent observational success.

Architectures, normalization, model size, source mixtures, additional training tasks, internal temporal cadence,
crops, and spatial resolution are open. Models can be replaced entirely. Match observational support where
practical, and compare candidates on a common evaluation footprint rather than prescribing one internal grid.

## Data and access

Torch paths below were checked on 2026-09-12 for presence and basic metadata, not re-audited for this revision.
They are access pointers, not assertions that each store already supports this task or covers its training period.
The full-range OSN and Empire AI (EAI) copies were published and audited on September 18–21, 2026, as
detailed below. Record versions, fields, units, spatial and temporal support, masks, source ancestry, and
permitted dates for every selected dataset and derivative. Availability of later dates does not change the training cutoff.

| Dataset | Access pointer | Role and qualification |
| --- | --- | --- |
| OM4 1° | `/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr` | Initializer and dynamics pretraining |
| OM4 ½° | `/scratch/jr7309/data/om4_halfdeg_v4/OM4.zarr` | Finer-grid pretraining source |
| OM4 ¼° | `/scratch/jr7309/data/om4_quarterdeg_v2/OM4.zarr` | Finer-grid pretraining source |
| GLORYS | [Copernicus product](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description); Torch path TODO | Candidate paired surface/interior and dynamics pretraining source; assimilating reanalysis |
| DUACS, legacy prepared | `/scratch/am16581/data/obs/duacs.zarr` | Five-day surface metric data; inspected store has velocities, no SSH, and starts in October 2014 |
| DUACS, daily velocity archive | `/scratch/am16581/data/obs_raw/duacs/` | Alternative preprocessing; inspected archive starts in October 2014 |
| OISST, legacy prepared | `/scratch/am16581/data/obs/oisst.zarr` | Five-day SST metric reference; use the full-range store below for daily data |
| Argo/IAP, legacy prepared | `/scratch/am16581/data/obs/argo-iap.zarr` | Monthly gridded T/S; full-range copy below extends coverage |
| Individual Argo profiles | [Argo access](https://argo.ucsd.edu/data/data-from-gdacs/); Torch staging TODO | Candidate primary interior labels and evaluation observations |
| EN4 profiles | [Met Office EN4](https://www.metoffice.gov.uk/hadobs/en4/); Torch staging TODO | Candidate broader collection of interior profiles; overlaps Argo |
| LLC and CM4 | Selected releases and Torch paths TODO | Additional simulation sources when available |
| ERA5 surface | Verified OSN and EAI paths below; no Torch copy claimed | Eight hourly atmospheric fields from 1993; model forcing adapter still required |

The three OM4 archives are regriddings of the same underlying simulation, not independent simulated trajectories.
Each has 4,745 records. Distinguish the effect of retaining spatial detail from adding a different simulation or
higher-resolution native dynamics. Verify fitting dates before reusing supplied means and standard deviations.

The legacy observation copies configured by the existing evaluator also exist under
`s3://m2lines-pubs/Samudra/v2026-07/obs/{duacs,oisst,argo-iap}.zarr`, using endpoint
`https://nyu1.osn.mghpcc.org`. Their DUACS/OISST fields use centered five-day means on OM4 timestamps. Preserve
daily observations and predictions, and construct explicitly matched derivatives when using those legacy products.

### Native-cadence observations and ERA5 on OSN and EAI

Task `01a0aad5-4372-71c0-b1d3-b121e0408166` published OISST/IAP and the original DUACS velocities on
2026-09-18, extended DUACS with SSH on 2026-09-20, and completed ERA5 publication and its independent audit
on 2026-09-21. Each product is one consolidated Zarr store retaining its upstream grid and native cadence,
with coordinate/schema standardization and no additional temporal averaging, regridding, or filtering.
The providers' own analysis/interpolation remains part of the products; IAP contains monthly gridded T/S,
not individual Argo profiles. These are frozen inventories, not automatically updated archives.

| Dataset | OSN S3 location | EAI Grace location (access via `ssh alpha`) |
| --- | --- | --- |
| DUACS, velocities and SSH | `s3://emulators/jr7309/data/full_range/duacs.zarr` | `/mnt/home/jrusak/data/obs_full_range/prepared/duacs.zarr` |
| OISST | `s3://emulators/jr7309/data/full_range/oisst.zarr` | `/mnt/home/jrusak/data/obs_full_range/prepared/oisst.zarr` |
| Argo/IAP | `s3://emulators/jr7309/data/full_range/argo-iap.zarr` | `/mnt/home/jrusak/data/obs_full_range/prepared/argo-iap.zarr` |
| ERA5 surface | `s3://emulators/jr7309/data/full_range/era5-surface.zarr` | `/mnt/home/jrusak/data/obs_full_range/prepared/era5-surface.zarr` |

| Dataset | Frozen coverage and cadence | Native grid | Fields and use |
| --- | --- | --- | --- |
| DUACS, release `202411` | 1993-01-01–2026-01-16; 12,069 daily records at 00:00 UTC | 0.125°; 1440 × 2880 | `adt`, `sla`, `ugos`, `vgos`, `ugosa`, `vgosa`; direct SSH initialization/labels and geostrophic velocity/EKE reference |
| OISST v2.1 final | 1981-09-01–2026-09-03; 16,439 daily records at 12:00 UTC | 0.25°; 720 × 1440 | `sst`; daily surface labels/reference and eligible pre-origin initialization |
| IAP/CZ16 | 1960-01-01–2023-09-01; 765 monthly records, labeled at month start | 0.5°; 360 × 720; 41 depths | `temp`, `salt`; gridded interior supervision and OHC diagnostics |
| ERA5 surface, final only | 1993-01-01 00:00–2026-06-30 23:00 UTC; 293,616 hourly records | 0.25°; 721 × 1440 | Eight fields below; atmospheric forcing, with ERA5T excluded |

DUACS SSH comes from the Copernicus Marine product `SEALEVEL_GLO_PHY_L4_MY_008_047`, dataset
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`, release `202411`. The extension preserves the
existing four velocity arrays and coordinate bytes. It adds `adt` and `sla` to the **same** local/remote
`duacs.zarr` after exact time/latitude/longitude matching; original velocity provenance remains recorded.
Select and document ADT versus SLA, reference datum/climatology, and alignment with simulated SSH before
using either as an input or target; their availability does not implement a direct-SSH metric.

ERA5 was streamed from the public
[ARCO ERA5 store](https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zmetadata).
Its final-data validity bound, rather than the source array's allocated time axis, fixes the end date above.
The retained fields are `10m_u_component_of_wind`, `10m_v_component_of_wind`, `2m_temperature`,
`2m_dewpoint_temperature`, `surface_pressure`, `surface_solar_radiation_downwards`,
`surface_thermal_radiation_downwards`, and `total_precipitation`. Decoded values, masks, units, and provider
time labels are preserved. Radiation and precipitation retain their provider accumulation units; they have
not been converted to flux rates. Define accumulation windows, conversions, interpolation, and the
OM4/ERA5 forcing interface before training. This archive is atmospheric reanalysis, not an operational
forecast; retain the stated conditional-forcing interpretation when supplying future atmospheric fields.

Use S3 endpoint `https://nyu1.osn.mghpcc.org` with credentials authorized for the `emulators` bucket, or the
configured rclone prefix `nyu-osn:emulators/jr7309/data/full_range`. Authenticated access was verified;
anonymous requests returned HTTP 403 in the September 18 check. All four **prepared stores persist locally**.
Source downloads under `obs_full_range/raw/{duacs,oisst,argo-iap}/` were removed after verification on
2026-09-20 with user authorization, freeing 355,482,812,609 bytes; new SSH/ERA5 source blocks streamed through
memory. Inventories, provenance, block-verification receipts, reports, and failed-job logs remain available.

| Newly published store | Stored bytes | Files including consolidated metadata | Successful publication / independent audit jobs |
| --- | ---: | ---: | --- |
| Combined DUACS | 1,305,414,878,529 | 193,176 | `93663` / `93702` |
| ERA5 surface | 5,497,241,311,236 | 1,957,475 | `93554` / `94809` |

Each OSN store has sibling `PRODUCT.inventory.json` and `PRODUCT.SUCCESS.json` records. The immutable
original DUACS inventory is also retained as `duacs.inventory.previous.json`; the local combined inventory is
`manifests/duacs-with-ssh.json`. Processing used commit
[`e562fbe008c9ac7e81c2a0b28f970152b283b224`](https://github.com/m2lines/Samudra/commit/e562fbe008c9ac7e81c2a0b28f970152b283b224).
The scripts are in [PR #884](https://github.com/m2lines/Samudra/pull/884).

Verification evidence under `/mnt/home/jrusak/data/obs_full_range/reports/`:

- `duacs-ssh-publication-20260920.json`: combined-store inventory, metadata, size/count and success record
  match local/remote; publication read-back found zero differences across 193,175 payload files.
- `era5-surface-publication-20260921.json`: matching local/remote inventory, metadata, size/count and success
  record; publication read-back found zero differences across 1,957,474 payload files.
- `publication-completion-20260918.json`: original DUACS/OISST/IAP publication audit, also described in the
  [original report](https://github.com/m2lines/Samudra/blob/04d9dbf46f7c24646b2e487a7143e90a36b688d3/docs/observations-full-range-results.md).
- `source-cleanup-20260920.json`: verified removal of the old raw downloads, with prepared copies retained.

Every new SSH and ERA5 output block was read back and compared exactly against all decoded source values
and masks before checkpointing. Final validation checked timestamps, coordinates, schema/units, chunk presence,
receipts, and decoded samples. Publication performed full `rclone check --download`, then byte-verified
consolidated metadata and success records; the separate audit checked inventory, metadata and size/count.
Original velocity/OISST/IAP source-value comparisons were sampled, not exhaustive full-store source comparisons.
These checks establish data preparation and transfer integrity, not scientific forecast skill.

Use the full-range OISST and DUACS stores for daily SST/SSH/geostrophic references with explicit treatment of
00:00 versus 12:00 time labels. Individual profile preparation remains a separate TODO. OISST/DUACS are
observation inputs before initialization and labels/references afterward, never forecast-time forcings.

### Quarter-degree OM4 snapshots on EAI

The single selected quarter-degree variant is the **unfiltered snapshot** release documented in
[PR #880](https://github.com/m2lines/Samudra/pull/880). The OSN directory is
`s3://m2lines-pubs/Samudra/v2026-09/om4_quarterdeg_snapshots/`, with anonymous access at the same OSN endpoint.
Its retained local copy is `/mnt/home/jrusak/data/om4/v2026-09/om4_quarterdeg_snapshots/`.
Both contain `OM4.zarr`, `OM4_means.zarr`, and `OM4_stds.zarr`; averaged and filtered quarter-degree variants
were not copied in this transfer.

The main store has 4,745 five-day snapshots labeled 1958-01-06 00:00 through 2023-01-01 00:00,
19 depths, and a 720 × 1440 Gaussian latitude/longitude grid (`grid_type=gaussian`, source grid
`gaussian_grid_720_by_1440`). Upstream preparation conservatively regridded the OM4 source and skipped spatial filtering. Ocean state (`thetao`, `so`, `uo`, `vo`, `zos`) is instantaneous;
`hfds`, `wfo`, `tauuo`, and `tauvo` remain five-day-mean forcings over the transition. Snapshot labels mark
interval upper bounds and are 2.5 days later than the matching averaged-state labels. Preserve the stored
Julian calendar and verify normalization fitting dates against the experiment split before using the supplied
statistics. This is the same underlying OM4 trajectory, not an additional independent simulation.

Grace copy job `94565` completed on 2026-09-21, retaining **1,594,021,376,137 bytes in 385,330 files**.
Full `rclone check --download` found zero differences across 385,327 payload files; the three consolidated
metadata files were checked for source stability and installed last. Source/local byte totals and file counts
matched. This was a byte copy with no new scientific preprocessing. Evidence:
`/mnt/home/jrusak/data/om4-v2026-09-transfer/reports/94565/SUCCESS.json`.

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

The full-range surface stores now include daily SSH and provide pre-cutoff coverage; select eligible windows
and align SSH reference conventions before using it as an input or direct label. Check whether retrospective
surface products use observations beyond a nominal forecast origin; pin either a causal input construction or explicitly label a retrospective reconstruction task.
A separate hidden-test governance policy remains outside this problem statement, but repeated evaluation-informed
development must be distinguished from independent confirmation.

## Evaluation and success criteria

The primary interior result is prediction of **unseen future interior observations from surface-initialized states**.
Initializer accuracy at lead zero is a separate diagnostic. A good reconstruction alone does not demonstrate
skill in evolution. Daily SSH/SST prediction and surface metric reporting are also required; improvement confined
to the surface does not establish interior skill.

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
| Daily SST skill | Required surface evaluation | Existing OISST SST error and variability metrics with documented temporal matching |
| Daily SSH skill | Required surface evaluation | Existing DUACS surface geostrophic velocity/EKE metrics derived from SSH; direct SSH errors using the available ADT/SLA after implementing the scoring adapter |
| Interior velocity fields | Retained output; plausibility diagnostic | Velocity distributions, kinetic energy, spectra/coherence, and held-out model-data skill as useful; no claim of gold observational verification |
| OHC, variability, drift, and stability | Supporting/longer-term | Gridded-product comparisons and physical diagnostics with coverage stated |

For the existing evaluator, expose SSH as `zos` and provide the SST field through its documented temperature
interface (currently `thetao` at the top level), or implement an explicit adapter. The DUACS metrics derive
surface geostrophic velocity from SSH; they do not validate the model's directly predicted interior `uo`/`vo`.
The existing scorer requires exact timestamps. Preserve daily outputs and compare with daily references where
available; for the prepared five-day products, explicitly match averaging support and timestamps before scoring.
Record which daily and matched-cadence metrics are reported rather than silently treating them as equivalent.
For repeated forecast origins, specify which leads enter each surface time series before reusing these metrics.

Assess whether velocity fields remain plausible after fine-tuning and whether model-data replay helps retain
them. Diagnostics and model-data comparisons can support that assessment, but better SSH/SST or T/S scores alone
do not demonstrate accurate interior velocities. Velocity diagnostics do not replace observational scores.

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

There is no prescribed scalar reward or percentage threshold. Report T/S and daily surface scores, velocity
diagnostics, uncertainty, coverage, and tradeoffs. Use best achievable performance as the headline; observation-only
training, coarse-only versus richer simulation training, and compute/data scaling are useful supporting comparisons
for attributing improvement. Their experiment order and budgets remain flexible.

## Per-experiment evidence

Keep a short Markdown account linked to Git and machine-readable artifacts. Record the question, code commit,
resolved configuration, checkpoint lineage, model and training strategy, source versions/paths, date splits,
preprocessing, atmospheric forcing fields, input history, forecast origins/leads, daily output and internal state
cadences, velocity supervision/replay policy, evaluator revision, and approximate compute.
Include results, uncertainty, coverage, failures, interpretation, and the next proposed experiment.

Save a manifest, per-profile prediction/observation matches (including source IDs and QC), aggregated metrics,
and deltas versus named references in CSV, JSON, Parquet, or existing equivalent formats. Keep durable pointers
to checkpoints, daily SSH/SST, interior velocity outputs, and their diagnostics rather than committing arrays to Git.
Label incomplete runs and partial-domain results. A human or agent should be able to trace a score back to its
model, inputs, and accepted observations.

## TODOs before an executable campaign

- [ ] **Surface initialization:** choose products, variables, history, masks, and geometry; select eligible
  pre-cutoff windows from the full-range stores, including the available DUACS SSH fields. Pin how inferred
  interior state connects to the dynamics model.
- [ ] **Daily surface outputs and data:** integrate the available full-range OISST SST and DUACS ADT/SLA labels;
  select SSH reference conventions and implement direct SSH scoring; define daily readout from the chosen internal
  cadence, time-label handling, and adapters for existing metrics. Preserve daily outputs alongside any five-day matched derivatives.
- [ ] **Profile dataset:** choose Argo, EN4 profiles, or a documented combination; stage on Torch and record QC,
  duplicates, versions, units, depth support, and date coverage. Assess OceanDepths as a preparation option.
- [ ] **Time target:** choose endpoint/exact-lead predictions versus means; specify matching and interpolation,
  and verify the temporal support available in simulation and observation training sources.
- [ ] **Evaluation contract:** pin forecast origins, one/three/six-month leads or alternatives, test dates,
  domain, depth bands, profile sampling, daily surface scoring, weights, uncertainty, and reference predictors.
- [ ] **Splits and ancestry:** verify full window boundaries, fitted statistics, retrospective input support,
  and exclusions across raw observations, gridded products, GLORYS, and simulation-derived training tasks.
- [ ] **Surface-state forcing:** replace current OM4 flux inputs with intended atmospheric surface-state inputs;
  adapt the eight archived ERA5 fields to the model interface and pin transformations, accumulation/time support,
  and compatibility with OM4 pretraining. EAI/OSN copies are available; Torch staging is optional. Exclude
  forecast-time ocean SSH/SST and observed velocities from the forcing schema, including OISST/DUACS. Label interim flux experiments.
- [ ] **Additional model data:** inventory GLORYS, LLC, and CM4 paths, fields, geometry, cadence, dates, and
  permitted roles. LLC availability remains an explicit dependency, not an assumed completed transfer.
- [ ] **Training and scoring implementation:** implement separate pretraining, connected fine-tuning, sparse
  profile and daily surface losses, and matched evaluation, or identify reusable implementations with verified
  semantics. Retain interior velocity predictions and choose any model-data replay/auxiliary velocity losses;
  record plausibility diagnostics and forgetting relative to the pretrained model.

## Source references

This direction follows the discussion in task `01a09696-19e8-7ea3-86f8-a212d8a01eaa`, building on the original
velocity-transfer effort in `01a09223-d781-70f1-adaf-bbeb2dd25e04`, and the subsequent two-component model and
profile-evaluation discussion. The exact open choices above should be resolved in campaign instructions.

- [Meeting slides](https://docs.google.com/presentation/d/1WjVueoj0ibFlMIyf3RsEMsthqp7I_4-CTE-tJQtYz3I/edit)
- [Meeting notes](https://docs.google.com/document/d/1dttyhoUtmnU4bD-hl73mcfbHnaq7rTrdKJ0q23r5qc8/edit)
- [Literature review: ocean and atmospheric precedents](observational-ocean-literature-review.md)
- [Full-range OSN/EAI observations and publication audit](https://github.com/m2lines/Samudra/blob/04d9dbf46f7c24646b2e487a7143e90a36b688d3/docs/observations-full-range-results.md)
- [Reference OM4 split and configuration](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/om4.yaml)
- [Existing gridded observation products](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/configs/data/obs.yaml)
- [Existing gridded metric definitions](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/report.py)
- [Existing observation comparisons](https://github.com/m2lines/Samudra/blob/55fe01dde9739b1967ecfe9b81a074ef0a61e19a/src/samudra/metrics/comparisons.py)

The pinned code references describe reusable infrastructure, not the proposed profile benchmark. Record and
validate any implementation adopted by a campaign against its final evaluation contract.
