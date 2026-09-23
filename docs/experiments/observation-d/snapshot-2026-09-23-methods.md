<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Observational D snapshot: technical methods

See the [snapshot report](snapshot-2026-09-23.md) for results and next decisions.

## What this pilot establishes

The existing **D initializer and five-day evolution model can be reused**. Strict loading,
zero-adapter equivalence, real-data fitting and complete training/evaluation all ran on
Torch. No OM4 pretraining was repeated. The only added network is an eight-input,
three-output atmospheric adapter (387 parameters); the original ocean tensor shapes
and architecture remain intact. Its outputs are learned conditioning features, not
calibrated physical fluxes. The source-control adapter outputs zero in the original
normalized forcing slots; this is not a claim of physically zero atmospheric flux.

The three comparisons are full fine-tuning, adapter-only with the ocean networks
frozen, and the same architecture initialized from scratch. Scratch uses no OM4
weights or normalization, but its atmospheric inputs include ERA5 reanalysis and its
interior targets are gridded IAP analyses. All arms retain the same model grid and
ocean masks. “Observation-only” describes the ocean
training source, not raw, independent measurements or an operational forecast.

## Data and physical conventions resolved

- **Common grid:** conservatively remap to D's actual 180 × 360 Gaussian coordinates,
  with periodic longitude and at least 90% finite source coverage over each full
  target cell. The native grids differ; a nominal one-degree label is insufficient.
  The [sample audit](materialized-sample-audit.json) found SST/ADT coverage of
  93.95%/94.19% within the model's 60°S–60°N surface domain for the audited month.
- **Time:** OISST, DUACS ADT and ERA5 use explicit five-day means. Nineteen prior
  surface bins feed six or seven forecast bins; calendar overlap weights produce
  the next-month interior prediction. Monthly IAP supervises 14 model depth centers
  through 1850 m. No extrapolated observations below 2000 m.
- **Quantities:** convert IAP Absolute to Practical Salinity with pressure/location.
  Retain the existing direct Celsius temperature approximation and shallow-model SST
  proxy. OHC uses native-layer overlap integrals on each vertical grid, density
  1035 kg/m³ and heat capacity 3850 J/kg/K, with complete-column common support.
  DUACS ADT is the SSH target; surface velocity/EKE come from SSH geostrophy.
- **Operator and missing-data effects measured:** the original [single-month check](coarsening-consistency.json)
  found 0.0656 m/s disagreement between geostrophy of coarsened ADT and directly
  coarsened DUACS velocity. The [full-cohort audit](artifacts/2026-09-23/heldout-operator-audit.json)
  feeds *target-time observed ADT* through the unchanged scoring operator. Its vector
  RMSE is 0.1603 m/s on full scoring support, but about **0.0730 m/s** where the
  observed ADT gradient stencil is complete. The **5.38%** of weighted component/origin
  support touching missing ADT has about **0.620 m/s** RMS disagreement and contributes
  roughly **80% of squared velocity disagreement**. DUACS u/v masks also differ;
  this audit reproduces the scorer's component-wise support exactly. These are
  data/operator effects, not forecast errors or irreducible error floors.
  Even this target-time ADT diagnostic has median day-30 EKE spectral power only
  **24%** of the directly coarsened DUACS reference (nine region/bin comparisons).
  Forecast dynamics further weaken power, but cannot explain the whole deficit.
- **Splits and missingness:** 243 training months (May 1993–July 2013), nine validation
  months (November 2013–July 2014), 96 test months (2015–2022). Normalization and
  seasonal climatologies use training samples only. Climatology fills missing inputs
  with validity masks; filled targets are never scored.

The compact prepared package is **20.385 GiB**, 350 NPZ files including grid and
statistics. All 350 hashes were checked on Torch before fitting. Preparation used
EAI CPU resources, then OSN and the Torch DTN; the raw full-resolution stores were
not copied to Torch. No scratch cleanup or deletion was needed.

## How selection and evaluation work

Checkpoint selection used the frozen validation objective throughout:

`0.5 × mean(five integrated errors / fixed validation-climatology errors)`
`+ 0.5 × mean(27 spatial spectral errors in dex)`.

The five integrated errors are SST, geostrophic velocity, EKE and OHC in 0–700 m
and 700–2000 m. Surface metrics average the matched day-5, day-15 and day-30 bin
errors. Spectra cover SST, ADT and EKE in the North Pacific, Gulf Stream and Agulhas
at those three leads. Every required component must be finite; there is no RMSE-only
fallback. Training loss and depth-resolved T/S errors are diagnostics. **Held-out
scores retain the validation denominators**, and were not used to pick checkpoints,
change weights or extend budgets.

Only three radial spectral bins per region survive the four-cell minimum wavelength:
approximately 600–3000 km. These are **broad-scale spatial spectra**, not mesoscale
or temporal spectral evidence. Nine validation months cannot establish annual or
long-period variability. Independently reinitialized forecasts are not concatenated
into a continuous trajectory. The pilot's monthly 2015–2022 cohort and coarsened
references differ from the legacy native-grid, continuous-rollout metric cohort;
numerical scores should be compared within this pilot.
