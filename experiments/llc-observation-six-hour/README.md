<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# LLC + observations at six-hour cadence

This is an experiment record, not intended to merge. The planned first run trains the
surface-initialized deterministic ocean model jointly on coarsened LLC4320
snapshots and observed SST/SSH, with ERA5 atmospheric conditioning. OM4 and
monthly profile supervision are excluded from this first run. No training has
started. Here OBS refers to the observation source (OISST SST and DUACS SSH);
the model and harness come from [#960](https://github.com/m2lines/Samudra/pull/960)
and [#955](https://github.com/m2lines/Samudra/pull/955).

## First run

| Choice | Initial setting |
| --- | --- |
| Grid and state | Existing 1-degree coarsened LLC grid; all 51 LLC depths (0.5–945.595 m); U/V/temperature/salinity plus SSH, 205 channels |
| Initialization | One week: 29 surface/ERA5 frames spaced six hours apart; infer two physical states at t−6h and t |
| Observation inputs | Daily SST/SSH occupy their available daily slots; missing six-hour slots remain masked |
| Dynamics | Fixed six-hour processor; 12 autoregressive steps, a three-day forecast |
| LLC targets | Instantaneous snapshots at every six-hour step, all state fields |
| Observation targets | SST and SSH at steps 4, 8, 12; gradients pass through all intermediate processor calls |
| Atmosphere | The same eight ERA5 variables for both sources, including prescribed future weather; this is a hindcast experiment |
| Architecture | Full deterministic OBS initializer and processor U-Nets, widths [280, 380, 480, 520], affine InstanceNorm, capped GELU, 3×3 output heads |
| Source schedule | Alternate LLC and observation optimizer updates, 50/50; initialize weights from scratch |
| Budget | 2,000 optimizer updates; effective batch eight windows per update; AdamW, LR 1e-4, weight decay 0.01, gradient clip 1, EMA half-life 100 updates |
| Forecast loss | LLC: equal weight per U/V/T/S/SSH group, averaging depths within group; observations: equal SST/SSH weight; wet-area-weighted normalized MSE |
| Auxiliary loss | LLC initializer reconstruction weight 0.1; masked surface completion weight 0.1 for both sources |
| Exclusions | No monthly targets or monthly reconstruction on OBS; no replay; no observation-only cooldown |

The processor consumes the two most recent physical states and predicts the
next six-hour state, then shifts that pair forward. Both sources provide only
SST/top-layer temperature and SSH as ocean inputs to the initializer.
Reconstruction scores the two inferred LLC states, excluding the supplied
surface channels. Completion artificially hides surface regions in the history
and scores originally available values in the final two inferred frames.

The ERA5 fields are 10 m east/north wind, 2 m temperature, 2 m dewpoint, surface
pressure, downward surface solar radiation, downward surface thermal radiation,
and total precipitation.

Use observation-training statistics for the two observed channels and LLC-training
statistics for the remaining state channels. LLC top-layer temperature and Eta
use those same SST/SSH scales. The initial convention compares top-layer
temperature with SST and absolute Eta with ADT; quantify SSH datum offsets
before launch and report them, since normalization does not remove that mismatch.
Use one ERA5 normalization for both sources. Retain physical wet masks separately from observation availability and
loss support. Dense LLC surface history remains six-hourly; do not impose the
observation product's daily sampling on it.

The proposed LLC split is training through 2012-09-13 and validation from
2012-09-14 through 2012-11-15. Exclude this validation interval from observation
training too. A sample's complete history and forecast must fit its permitted
split and avoid exclusions. Retain the OBS branch's separate observation
validation (2013-11 through 2014-07) and test (2015-01 through 2022-12) years.
Fit normalization on training support only.

The first preparation must record the actual native LLC depth centers/thicknesses,
field order, units, time support and statistics. The coarsened store lacks explicit
physical depth coordinates. Use its existing compact arrays and [#1034's grouped
reader](https://github.com/m2lines/Samudra/pull/1034); do not expand the archive into 205 physical arrays.

The sampler currently uses midnight daily analysis slots. Preparation must
record how each product timestamp maps to those slots and finalize the ERA5
state/accumulation support rules. Daily products are analyses with differing nominal timestamps (OISST noon,
DUACS midnight), not measurements at every six-hour step. Record a common daily
analysis-slot convention and retain original timestamps. Do not turn daily
products into artificial subdaily labels. ERA5 state fields and accumulated
fields require separate time-support rules and explicit unit conversion.

## Qualification and reporting

Before a production run, verify data coverage, masks, normalization, temporal
alignment and split exclusions. Run synthetic tests for 29-frame initialization,
sparse daily targets, gradient flow through unsupervised steps, source alternation
and checkpoint/resume. Then measure GPU memory, data wait and update time with
both real sources before sizing the allocation.

Report LLC validation by field, depth and lead, and observation SST/SSH at daily
leads, against matched persistence. Keep a fixed validation cohort and preserve
raw/EMA checkpoint identity. A longer 30-day rollout is a diagnostic after the
short forecast works; it adds no monthly training supervision. Record job IDs,
config and code hashes, data proofs, W&B URL, completed optimizer updates and
source exposure here. No scientific result is claimed by this design document.

## Questions for adding OM4

1. **Vertical representation.** LLC has 51 levels in the upper 946 m; current OM4
   has 19 levels extending to 6 km. Choose a shared shallow physical state,
   conservative remapping with masks, or a latent state with source-specific
   depth decoders. A latent state is the preferred longer-term direction. Define
   layer bounds and how deep OM4 information influences the shared dynamics.
2. **Time step and recurrent history.** OM4 snapshots are available at five-day
   spacing. Decide whether processor steps vary directly or fine steps are
   composed to reach OM4 targets. Condition on elapsed time, and specify the
   time separation of both recurrent states. An OM4 sample cannot supply missing
   six-hour snapshot history without an explicit inference scheme.
3. **Temporal support.** Use snapshot stores for all LLC experiment fields and
   identify the corresponding OM4 snapshot products. Keep interval-mean products
   distinguishable. Monthly supervision returns only after variable-step
   rollouts and duration-weighted averaging are implemented and tested.
4. **Atmospheric conditioning.** ERA5 is a convenient common covariate but is
   not either simulation's exact native forcing. Compare it with native fluxes,
   define instantaneous versus accumulated/averaged fields, and distinguish
   historical prescribed forcing from genuinely forecastable inputs. CM4's
   free-running atmosphere would need a separate treatment.
5. **Physical definitions and SSH reference.** Check LLC Eta, OM4 zos and DUACS
   ADT reference levels; SST versus model top-layer temperature; velocity
   rotation/staggering; salinity/temperature conventions; tides and filtering.
   Decide whether datum/bias adapters or explicit anomaly conventions are needed.
6. **Grid, masks and losses.** Carry actual coordinates, cell areas and depth wet
   support. Choose source/grid-specific readouts and area/volume weighting.
   Define comparable normalization and losses without allowing 51 LLC levels to
   dominate 19 OM4 levels or two observed surface fields.
7. **Exposure and evaluation.** LLC covers about 14 months while OM4 and OBS
   cover decades. Set source probabilities and matched optimizer/data budgets,
   prevent overlapping-calendar holdout leakage, and use source-specific
   persistence controls. Separate forcing mismatch from transfer benefit.
8. **Repairs and provenance.** Preserve LLC repair information and compare clean
   support where possible. A repair flag does not invalidate an entire map.
   Repaired native forcing does not enter a run conditioned exclusively on ERA5.

## Branch provenance and current status

The experiment branch combines deterministic OBS #960 (`390437693`), the updated
OBS harness #955 (`32d3f063e`), main including LLC #971, and compact reader #1034
(`8457d6596`). The first merge reconciles the rebased harness with the deterministic
extension. The branch implements the six-hour source adapter, sparse daily loss, compact
LLC reads, validation against persistence, and the existing distributed training,
prefetch and checkpoint loop. The runnable [training preset](train.yaml) retains
the 2,000-update budget. Validation uses eight fixed, evenly spaced origins per
source every 100 updates, at 24/48/72 hours. Selection uses EMA observation
normalized RMSE averaged over channels and daily leads, after pooling squared
error across origins. Physical RMSE uses the same pooling order.

LLC's persistence baseline has the true full initial state, while the model
receives surface history. Label it **full-state persistence** for interior
comparisons. Observation persistence uses the last daily surface analysis;
model and persistence are scored on identical joint origin/target support.

Local qualification passed 65 tests across the new snapshot/runtime suites and
existing observation harness/prefetch tests. This includes exact four-update
CPU training equivalence between uninterrupted and 2+2 resumed execution.
Adversarial tests independently check time/channel ordering, forcing intervals,
sparse observation availability and exclusion boundaries. Pre-commit checks pass.

The Grace preparation pilot, job `208388`, completed under standard QoS. Native
LLC Z/drF were read from Engaging and the target Gaussian grid from the published
Samudra grid archive; its centers exactly match the existing surface products.
The [ERA5 pilot audit](qualification/era5-pilot.json) compares four six-hour bins
against an existing daily product on 1993-01-01. Daily-mean differences are about
0.0011 K for temperature and 0.0125 W m−2 for downward solar radiation. The new
ERA5 path uses the repository's ESMF conservative remapper; the existing daily
surface products retain their preparation provenance. Tiny negative solar values
are retained, with a minimum of −0.000069 W m−2 in this sample.

Preparation array `208394` covers 1993 through July 2014; its concurrency was
increased from eight to 22 CPU tasks after the pilot. Normalization job `208395`
depends on successful preparation. The additional cache is roughly 50 GiB; LLC
fields remain in the original compact store. No GPU training has started yet.

The [EAI launcher](eai.sbatch) uses a four-GPU Beta allocation, standard QoS and
online W&B. Its [qualification gate](run_eai.sh) checks eight updates split into
four plus a resumed four, both sources and daily validation before production.
Qualification is bound to actual runtime source hashes, both configs, prepared
data proof, Torch/CUDA versions and world size. A short test-queue qualification
can share this gate with production; only one allocation owns qualification.

The source stores on EAI are
`/projects/ny/lz1955/multiscale/jrusak/data/coarse_LLC4320/LLC4320_samudra_1deg.zarr`
and the OISST, DUACS and ERA5 stores under
`/projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/prepared/`. A live check on
2026-11-20 found approximately 14 TiB available on the project filesystem and
idle Grace CPU nodes; Alpha GPU nodes were mixed, reserved or draining. Free
filesystem capacity is not a per-user quota or a GPU start-time guarantee. Choose
compute after qualification and a fresh scheduler check, using standard QoS.
