<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Proposal: fine-tune deterministic D on coarsened observations

22 September 2026. **Approved for implementation and pilot execution; not yet launched.**

Progress snapshot due **23 September 2026, 10:00 a.m. America/New_York (14:00 UTC)**.
Delivering that snapshot completes the requested goal; ongoing runs may continue.

**Recommendation:** reuse the existing D initializer and autoregressive evolution weights, retain their
one-degree grid and five-day step, and fine-tune against coarsened OISST/DUACS and monthly IAP T/S.
Add a small learned atmospheric-input adapter. Let fine-tuning learn remaining product biases and
reconstruction differences. This is an analysis-trained transfer pilot, with retrospective surface inputs
and prescribed ERA5 atmosphere. It deliberately postpones daily outputs, individual profiles, operational
availability, and three/six-month forecasts from the broader PR #881 objective.

## Which D and what can be reused

This is arm D of the [wave-3 plan](https://github.com/m2lines/Samudra/blob/21d8edb3426d119456dd5249bcb87265f959b454/docs/experiments/surface-wave3-plan.md),
not the later single-field deterministic residual control paired with diffusion.

| Component | Verified contract | Proposed treatment |
| --- | --- | --- |
| D initializer | 121,684,010 parameters; ConvNeXt U-Net widths 256/384/512/768; 19 surface/forcing frames spanning 90 days between labels | Keep architecture, history, channel order and all weights |
| Initializer inputs | SST, SSH, two surface masks per frame; three OM4 forcings per frame; geography and annual sine/cosine; 138 concatenated channels | Supply observed SST/ADT and adapted ERA5; keep 138 channels |
| Initializer outputs | Two 77-channel states: T/S/U/V at 19 depths plus SSH; final two SST/SSH frames copied exactly | Keep outputs, surface copying and the physical-state interface |
| Evolution | Separate autoregressive U-Net; two states plus three forcings and five context channels; 162 inputs, 77 outputs | Keep pretrained weights and five-day step; fine-tune jointly after initialization adaptation |
| Grid / geometry | OM4 `om4_onedeg_v3`, 180 × 360 Gaussian latitude/longitude, existing 19 depths and ocean masks | Remap observations to these actual coordinates and retain the existing state mask |

Starting checkpoint: `/scratch/jr7309/runs/2026-09-22-initializer-wave3-primary/D/best.pt`.
Its live SHA-256 matches the published lineage: `22e629f41c418fa93ee3ec0a3257d80d71d77b02b2fe1cbd89eae04507d22905`.
It includes the initializer and original evolution weights. The producer is
[`96aea56c`](https://github.com/m2lines/Samudra/commit/96aea56c92d82aa18255942b3443b87ed510a589).
The already jointly adapted alternative is `/scratch/jr7309/runs/2026-09-22-initializer-wave3-joint/D/best.pt`,
also present and checksum-matched. Use original D for the primary transfer lineage; do not mix its
initializer with adapted dynamics silently. D-joint can be a later warm-start comparison.

**What is not a drop-in:** the current loader/loss assume dense OM4 full-state labels. They need an
observation loader, validity-aware losses, and monthly aggregation. The eight archived ERA5 fields are
not the three physical fluxes D expects. Those are adapter/training-harness changes, not reasons to
repeat 121M-parameter pretraining. This conclusion follows from code and checkpoint provenance; an
actual strict-load/forward compatibility test remains part of implementation qualification.

## Fixed data choices

Use the full-range stores from PR #881 under `emulators/jr7309/data/full_range/` on OSN or
`/mnt/home/jrusak/data/obs_full_range/prepared/` on EAI. Prepare a compact derived dataset before moving
it to the training machine; the legacy 2014–2022 metric stores cannot supply the training period.

| Quantity | First-pass preparation |
| --- | --- |
| SST | OISST Celsius; area-weighted remapping to the one-degree model cells, then five-day averages |
| SSH | DUACS **ADT in meters**, not SLA; same remapping and averaging. No separate bias-correction project, full-record detrending, or test-period climatology fitting |
| Interior T/S | Monthly IAP, area-weighted horizontal coarsening; interpolate its native depth values to the 14 model centers from 2.5 to 1850 m. No extrapolation across missing depths or below valid bathymetry |
| Temperature semantics | Use IAP Celsius directly, as the existing OHC comparison does; retain the existing shallowest-model-level SST convention. Document this approximation rather than delaying for a temperature-definition overhaul |
| Salinity semantics | Given the stored Absolute Salinity metadata, convert once to Practical Salinity using pressure/location before remapping. This is a small deterministic conversion, not a model-development project |
| Atmosphere | All eight already archived ERA5 fields; five-day means of state variables and properly converted radiation/precipitation rates; fit input standardization on training data only |
| Geometry / missingness | Keep the OM4 state mask. Supervise only where both model and observation are supported; require at least 90% valid source wet area in a coarse cell. Primary scores cover 60°S–60°N to defer ice-region decisions |

The salinity conversion is [the standard TEOS-10 `SP_from_SA` operation](https://www.teos-10.org/pubs/gsw/html/gsw_SP_from_SA.html),
using pressure derived from depth/latitude. Save the source convention and converted values separately.
For temperature, apparent statistical bias between products is left for fine-tuning to learn.

Retain each native product as the reference. The one-degree derivative is a training and matched-scale
scoring product, not a replacement for the original observations. Use explicit spherical cell bounds,
periodic longitude, and wet-fraction weights; striding every Nth native cell is not the proposed coarsening.

For observed surface gaps, fill input values with training-period climatology and set the existing mask
channels to zero for that frame/cell. This requires accepting per-frame masks in `HistoryInitializer`,
but no weight shapes change. Its hard-copy path must copy the filled input rather than NaN; such filled
values are never scored as observations. Retain the separate static output mask. Initialization SST/SSH
loss is zero by construction at supplied cells, so it is not a useful adaptation objective.

Preserve source ocean normalization inside the transferred model as part of its checkpoint interface;
do not replace those numbers and assume the old weights implement the same function. Use separate
training-observation scales for loss balancing. Freeze BatchNorm running statistics for this small-data
pilot, while allowing affine parameters to fine-tune; the source network uses BatchNorm.

## Learn the forcing mapping while retaining the networks

Add a pointwise `8 → 32 → 3` MLP, implemented as two 1×1 convolutions with a nonlinear activation,
shared across time and between initializer history and future evolution. Its inputs are standardized
ERA5 fields; its outputs occupy the original three normalized forcing slots. **Treat these as learned
conditioning features, not physically verified wind stress and net heat flux.** This avoids requiring
new ERA5 flux downloads or a bulk-flux model in the first pass.

Initialize the final adapter layer to zero. Before adaptation, it exactly reproduces the original
networks supplied with zero normalized forcing; no claim of useful zero-shot ERA5 skill is implied.
Train the adapter against ocean losses, then unfreeze the existing networks. It has only 387 parameters;
the old initializer and evolution parameter tensors can load strictly and remain unchanged in shape.
Future forcing contains only ERA5, never observed future SST/SSH or IAP.

This is a reasonable first hypothesis, not a guarantee that three conditioning features are enough.
If validation and a small-set fitting check show that bottleneck matters, expand to eight forcing channels
in the networks. That changes D from 138 to 233 inputs and evolution from 162 to 167. **In this code it
changes the whole first ConvNeXt block**, including its expansion width, normalization and skip projection,
not just a single input convolution. Retain all later blocks and heads; retrain the new first blocks and
fine-tune the rest. Even that fallback does not inherently require full OM4 pretraining from scratch.

## Match the monthly supervision without inventing daily interiors

Use explicit five-day intervals; do not repeat a monthly IAP field as six supposedly instantaneous labels.
For the observational loader, label each bin by its end and supply seasonal context at its midpoint.
Nineteen consecutive bins retain D's 90-day center-to-center history (95 days including averaging support).

**Reconstruction stage:** run D at the bins covering a calendar month and compare the overlap-weighted
monthly mean of its current-state reconstructions with that month's IAP field. Inputs may extend through
that month because this stage is reconstruction, not forecasting. The previous reconstructed state is
retained for the dynamics interface, without inventing a second independent monthly target.

**Forecast stage:** initialize once at the beginning of each target month from preceding surface history.
Predict successive five-day bins using only the inferred ocean states and prescribed ERA5 for each
predicted interval. Six or seven steps cover the calendar month; aggregate predictions using interval
intersection with the month before comparing with IAP. Seven steps require changing the wrapper's
hard-coded six-step loop, not the autoregressive network weights. Partial boundary bins use a documented
piecewise-constant five-day representation; this is a temporal-resolution approximation, not exact daily
reconstruction. Future surface losses compare predicted bins with observations averaged over those same bins.

This yields five- through approximately 30/35-day surface predictions and a next-calendar-month mean
interior target. It does not claim instantaneous interior skill at a one-month lead, or daily surface skill.
Use the same temporal operator for reference predictors. Keep native daily products available for later
validation without presenting interpolated five-day output as a learned daily forecast.

Use a masked, area-weighted loss with fixed group weights:

`L = 0.4 L_T_month + 0.4 L_S_month + 0.1 L_SST_5day + 0.1 L_ADT_5day`.

Scale each channel error using a training-only observational scale, average within each variable, then
apply group weights. Exclude the copied surface temperature from interior T loss. During reconstruction,
use only the T/S terms renormalized to equal weight. No velocity or deep-ocean zero targets. Keep all 77
state channels for checkpoint/dynamics compatibility, but make no new observational accuracy claim for
U/V or depths below 2000 m. No simulation replay is needed for the primary pilot; add it only as a
separately labeled retention experiment if later justified.

## Dates, training and interpretation

Treat this explicitly as **retrospective analysis-initialized, ERA5-conditioned prediction**. Accept
provider smoothing/look-ahead for this first-cut comparison, equally across candidates. A nominally
pre-origin DUACS map can contain future information; this choice permits an adaptation experiment,
not a causal forecast claim. Do not attempt an operational-vintage data project in this pilot.

Preserve the inherited training cutoff before 5 October 2013. For a conservative monthly pilot, use
training target months **May 1993–July 2013**, validation **November 2013–July 2014**, and test
**January 2015–December 2022**. The earlier training end leaves room for monthly support, a seventh
forecast bin and DUACS's documented six-week mapping window before the inherited cutoff. Audit the
actual support of every generated sample; these dates alone do not certify all upstream dependencies.
History-only observations before a validation/test origin are allowed. These monthly origins intentionally
differ from the OM4 wave's 99 origins. There are 243 candidate training months, not millions of independent
examples merely because each map has many cells; count accepted months and cells after preparation.

Proposed bounded optimization, with seed 1729, AdamW, clipping at one, effective batch eight and BF16
convolutions/FP32 reductions:

1. Qualify preprocessing, strict checkpoint loading, source-path equivalence, masks and monthly weights.
   Fit a small training subset to establish that gradients reach adapter and initializer. Save originals.
2. Warm the adapter with D/evolution frozen for up to 200 updates. Then fine-tune D plus adapter on monthly
   reconstruction, up to 1,000 updates or two training hours. Start with learning rates 1e-5 for D and
   1e-3 for the adapter. Keep evolution frozen in this stage.
3. Jointly fine-tune the adapted D, original evolution and adapter on forecast loss, up to 1,000 updates
   or four training hours. Use 1e-5 for pretrained weights, 1e-4 for the adapter, and optionally retain
   0.1 times the reconstruction objective. Fix this choice before production; the default is to retain it.
4. Validate every 100 updates, keep the starting checkpoint eligible, and stop after five non-improving
   checks under the observational selection protocol below. Track training RMSE as a diagnostic. Record actual
   updates and GPU-hours; the clocks are caps, not evidence of convergence. Budget two GPUs for the primary
   pilot, with a separate measured allowance for data preparation and evaluation.

### Observational selection and reporting: approved amendment

Select checkpoints on validation observations using the integrated observational metrics, including
spectral diagnostics. Report SST, geostrophic velocity, EKE, OHC, variability and spatial spectra,
alongside T/S and ADT errors. Preserve individual metric units and component scores; do not sum raw
quantities with incompatible units or select solely on RMSE. Before production, freeze the exact
dimensionless aggregation or deterministic multi-metric ranking, weights, spectral support, validation
cohort and missing-metric policy in the run manifest. Use validation only for selection; reserve the
2015–2022 test period for reporting after selection.

Qualify which metrics have sufficient support in the nine-month validation window. Annual/trend and
long-period temporal spectra cannot be treated as reliable on this short record. Use supported spatial
spectra and matched-cadence diagnostics for selection, record unavailable components explicitly, and
retain longer-period diagnostics for the held-out report. Do not silently fall back to RMSE if the
integrated scorer is unavailable. Respect the one-degree effective resolution, common accepted cells
and forecast lead. Do not interpret jumps between reinitialized forecasts as ocean temporal variability.
Document and test necessary adaptations of the existing scorer before checkpoint selection.

The deadline report will include data coverage, code/checkpoint provenance, job IDs and actual progress,
completed observational comparisons and spectral diagnostics, missing evidence, blockers and next steps.
A running job is not a completed scientific result. Keep runs active beyond the snapshot when their
approved budgets allow; the report itself completes the goal.

Required comparisons are training-only seasonal climatology, persistence of the inferred T/S anomaly,
surface persistence, adapter-only/frozen-core transfer, and full fine-tuning. Reconstruction and forecast
scores must remain separate. Include physical T/S bias and RMSE by depth, anomaly correlation/amplitude,
five-day SST/ADT error, and secondary existing geostrophic/OHC metrics with matched cadence. Use identical
accepted cells and calendar months across methods; quantify paired year-block variation after selection.
Do not require a fresh Argo/EN4 pipeline before this pilot, and do not call IAP agreement independent
profile validation. Existing OHC and velocity diagnostics should retain their established definitions.

To answer the original simulation-benefit question, also run a fresh-weight D/evolution control on the
same observational examples and targets, with its own observation-only normalization and sufficient
validation-controlled training. Report its optimization budget and normalization difference explicitly;
a short fine-tuning-length scratch run that has not converged is not a fair best-observation-only baseline.
Because IAP is a gridded reconstruction, label this the **analysis-only** control rather than a strict
raw-observation-only benchmark.

## Retraining decision

- **No full OM4 restart is required for the recommended pilot.** D and its evolution weights are reusable.
- **New training:** the small forcing adapter; observational fine-tuning of existing D, then evolution.
- **New implementation:** coarsened observation cache, per-frame input validity, masked T/S/surface losses,
  calendar-month aggregation, and a configurable rollout length. The existing dense-OM4 loss is not reusable unchanged.
- **Only if needed:** larger forcing interface requires new first blocks; daily stepping requires additional
  temporal adaptation/readout training; individual-profile claims require a new scorer. None is an initial blocker.

Code evidence: [initializer and copying](https://github.com/m2lines/Samudra/blob/21d8edb3426d119456dd5249bcb87265f959b454/src/samudra/experiments/initializer_models.py#L187),
[pair and rollout](https://github.com/m2lines/Samudra/blob/21d8edb3426d119456dd5249bcb87265f959b454/src/samudra/experiments/initializer_wave.py#L58),
[evolution](https://github.com/m2lines/Samudra/blob/21d8edb3426d119456dd5249bcb87265f959b454/src/samudra/experiments/surface_state.py#L58),
[checkpoint lineage](https://github.com/m2lines/Samudra/blob/21d8edb3426d119456dd5249bcb87265f959b454/docs/experiments/surface-wave3-results/artifacts/lineage-audit.json).
The original workspace report `docs/reports/observational-baseline-2026-09-22.md` supplies grid and product evidence.
