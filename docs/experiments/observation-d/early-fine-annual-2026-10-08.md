<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Continuous 365-day evaluation of early/fine models

This diagnostic extends the [constant-rate](early-fine-wave-2026-10-07.md) and
[cooldown](early-fine-cooldown-2026-10-08.md) comparisons to continuous year-long
rollouts. It evaluates their ten already selected checkpoints; it does not
retrain models or select new checkpoints using annual outcomes.

## Evaluation protocol

- Use the established annual cases starting **January 1, 2015, 2018 and 2021**.
  Each has 19 five-day history bins and **73 autoregressive five-day steps**.
  Initialize once from the history. Future surface observations are targets only.
- Prescribe the same future ERA5 forcings through each trained adapter. This
  measures ocean evolution under known forcing, not operational weather forecasts.
- Score the global 1° observation grid at days **5, 15, 30, 90, 180 and 365**:
  SST/ADT and geostrophic velocity errors, regional SST/ADT spatial spectra,
  annual EKE errors/spectra, and calendar-month OHC errors in both depth ranges.
- Annual EKE uses anomalies around each sequence's own annual mean. It is distinct
  from the fixed-lead, independently initialized monthly EKE in the short report.
  Report components explicitly rather than treating the old 30-day composite
  as a 365-day selection criterion.
- Include **each checkpoint's own initialized persistence**: hold its last inferred
  physical initial state fixed throughout the year. This supplies matched
  long-horizon statistics and map controls.
- Produce day-365 maps against date-matched observations, with matched model pairs,
  shared color scales/masks and the existing two-pixels-per-grid-cell convention.
  Show individual origins as well as their aggregate errors; three cases provide
  a limited diagnostic, not a 96-origin annual benchmark.

Literal model names are U-global, U-multitask-early,
U-multitask-early-latent, U-multitask-early-fine,
U-multitask-early-fine-latent, and each corresponding `-cooldown` name.
Their data, architectures, parameter counts and training schedules are defined
in the linked reports. The annual comparison preserves those meanings and the
same integrated-plus-spectral validation-selected weights.

## Execution and checks

The annual data already exist at
`torch:/scratch/jr7309/data/obs-d-annual-instance-v1`.
The new evaluation root is
`torch:/scratch/jr7309/runs/2026-10-08-early-fine-annual`.
Scratch has approximately 0.45 TB free; saved evaluations are expected to use
well under 10 GB. Run every model on the same pinned Torch runtime; the original
U-global selected checkpoint is transferred from Beta with checksum verification.

Before GPU evaluation, a CPU audit verifies checkpoint lineage, monthly-data
manifest/grid/statistics fingerprints, all annual payload hashes/timestamps,
finite forcings, the first 30 days against the prepared monthly samples, and
calendar-month OHC targets. Model and normalization loading are strict. All source
checkpoints are read-only to evaluation jobs.

The annual evaluator adds persistence and permits a relocated monthly-data path
only when its fingerprints match training. Eleven targeted tests pass, including
single initialization through 73 steps, exclusion of future observations, exact
calendar-month overlap weights, and rejection of changed normalization files.
Submission and verified execution status will be recorded here concisely.
