<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Observational D snapshot evidence

See the [snapshot report](../../snapshot-2026-09-23.md) for interpretation.

- `evidence.json.gz`: complete timestamped capture of training manifests, all validation
  checks, selected checkpoints, held-out metrics, per-year/anomaly diagnostics, current
  submissions and Slurm accounting. Decompress with `gzip -dc evidence.json.gz`.
- `summary.json`: selected-checkpoint identities, component scores and paired-year
  diagnostics. Records the SHA-256 of the uncompressed evidence.
- `heldout-metrics.csv` and [component table](heldout-table.md): all seven methods for
  each arm. CSV OHC units are J/m²; the Markdown table uses GJ/m².
- `archive-and-wandb-evidence.json`: immutable selected-checkpoint archive manifests
  and the W&B completion audit. These are metadata, not model weights.
- `heldout/`: component, lead, spectral and monthly T/S plots, in PNG and PDF, with
  provenance. The component plot divides by held-out climatology only for descriptive
  display; the composite score always uses the frozen validation denominators.
- `validation/`: joint-stage validation trajectory with provenance. Update counts
  are not a matched-compute comparison.
- `heldout-operator-audit.json`: target-time observed-ADT operator diagnostic,
  including separate complete/incomplete observed gradient stencils. This is not
  a forecast candidate, and does not change selection. Producer `bb7c3b787`,
  CPU job 18330279; script and source-array hashes are embedded.
- `quota.txt`: final live Torch quota readout.
- `SHA256SUMS`: artifact checksums, excluding the checksum manifest itself.

All results use the fixed 96-origin January 2015–December 2022 cohort. The plotting
script verifies selected checkpoint hashes, complete origins and common spectral
reference curves. The CPU reporter verifies identical observational arrays and
OHC support across methods. Annual EKE anomalies are recomputed within each year;
annual scores are descriptive and do not form an additive decomposition of the
full-cohort score. Eight years are not eight independent experiments.

## Reproduction and retained outputs

Repository scripts:

- `scripts/snapshot_observation_pilot.py`
- `scripts/plot_observation_snapshot.py`
- `scripts/analyze_observation_predictions.py`
- `scripts/audit_observation_operator.py`

Run root on Torch: `/scratch/jr7309/runs/2026-09-22-observation-D`.
Prepared data: `/scratch/jr7309/data/obs-d-pilot`.
Published compact data: `nyu-osn:emulators/jr7309/data/observation_d_pilot/2026-09-22`.
The full prediction NPZ files remain under each arm's evaluation directory on Torch;
the repository contains small metrics/evidence files rather than duplicate arrays.

Training producer: `d8949bb15d9fa92e6c8c94702092360bbe9005ef`.
Evaluation producer: `7a892320332b43e16a5c1c8babfbea660cefc9a7`.
Primary selected forecast and selected-state persistence were produced by
`5b8d391230e9a0b97afd232520335c8036363248` before an anomaly-control timestamp
compatibility fix; their original hashes and producer are preserved in
`primary-evaluation/reused-output-provenance.json` inside the evidence capture.
The fix changed only conversion of a NumPy Unicode scalar to a Python string,
not model weights, training, normalization or metric kernels.

Source D checkpoint SHA-256:
`22e629f41c418fa93ee3ec0a3257d80d71d77b02b2fe1cbd89eae04507d22905`.
Prepared-data SHA256SUMS hash:
`d6e6672ca11aec3491a4fc896d9e99d528075a889cc0e6187b580ef8a52500f3`.
Grid SHA-256:
`ce96360964a6669d0186b9581abaaffc06f660c4e09e0ac813b3dd1b819e5167`.
Frozen selection-reference file SHA-256:
`ba1b2aeedb9523e76d44746ac96e0c9d9d2ffb2aecee60aa0788eb4c37bb1781`.

Selected checkpoint files are named `<arm>-<sha256>.pt` under the run root's
`selected-checkpoints/` directory. Each is 613,575,397 bytes and was copied and
read back with a matching SHA-256. Their exact identities are in `summary.json`
and `archive-and-wandb-evidence.json`.
