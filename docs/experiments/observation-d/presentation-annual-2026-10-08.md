<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Presentation models: matched 30- and 365-day RMSE

This follow-up tests whether differences hidden by short monthly forecasts become
clearer after a continuous year. It evaluates the **same selected checkpoints**
as the presentation, with no retraining or annual checkpoint selection.

## Models

| Model names | Meaning | Original methods/results |
|---|---|---|
| U-global | Global-only U-Net control | [Initial comparison](extent-wave-2026-10-02.md) |
| U-multitask | Replace half the coarse OM4 tasks with regional ¼° tasks | [Initial comparison](extent-wave-2026-10-02.md) |
| U-omit-patch | Omit those regional tasks | [Omission control](extent-ablations-2026-10-03.md) |
| U-patch-loss01 | Multiply the regional objective by 0.1 | [Loss control](extent-initializer-followup-2026-10-04.md) |
| U-aux01 | Global evolution with a 0.01-weight fine-scale variance accessory loss | [Accessory and capacity methods](extent-representation-2026-10-03.md) |
| U-aux01-static, U-aux01-seasonal, U-aux01-shuffled, U-aux01-anomaly | Respectively time-mean spatial maps, seasonal, shuffled, and anomaly accessory targets | [Target controls](extent-accessory-controls-2026-10-04.md) |
| W-global, W-multitask | Wider processor, respectively global-only and regional multitask | [Capacity methods](extent-representation-2026-10-03.md) |
| A-global, A-multitask | Axial-attention processor, respectively global-only and regional multitask | [Capacity methods](extent-representation-2026-10-03.md) |
| U-multitask-early | Add earlier historical coarse OM4 tasks | [Early/fine methods](early-fine-wave-2026-10-07.md) |
| U-multitask-early-latent | Same coarse data, with ten recurrent latent channels | [Early/fine methods](early-fine-wave-2026-10-07.md) |
| U-multitask-early-fine | Earlier OM4 at ¼° through an encoder and decoder around the coarse processor | [Early/fine methods](early-fine-wave-2026-10-07.md) |
| U-multitask-early-fine-latent | Same fine-resolution task with ten recurrent latent channels | [Early/fine methods](early-fine-wave-2026-10-07.md) |

There are 17 models. The existing [early/fine annual evaluation](early-fine-annual-2026-10-08.md)
already covers U-global and the four early/fine arms. Reuse those five outputs;
run only the twelve missing presentation comparisons. The five cooldown evaluations
in that separate job remain a supplemental comparison, outside this 17-model table.

## Fixed evaluation protocol

- Three established January 1 starts: **2015, 2018 and 2021**. Each forecast
  initializes once from 19 five-day surface/forcing history bins, then evolves
  for 73 five-day steps. Future surface observations are targets only.
- Use the same prescribed ERA5 forcing and global 1° grid for all models.
  This tests ocean evolution with known forcing, not operational atmospheric forecasting.
- Compare the **same three forecasts at days 30 and 365**. Surface quantities
  describe five-day bins ending at those leads. OHC uses full January and
  December forecast means, respectively, matching monthly IAP targets.
- Headline RMSE-only score: equal-weight mean of four errors divided by common
  seasonal-climatology errors: SST, SSH-derived geostrophic velocity, OHC
  0–700 m, and OHC 700–2000 m. The control uses training-only monthly surface
  and interior climatology. Both leads use these same four quantities.
- Pool spatial MSE equally across origins before taking its square root;
  show individual origins and physical-unit components as well as the score.
  Per-origin normalized rows use the same pooled climatology denominator.
- Include each checkpoint's own initialized persistence. Report ADT RMSE,
  intermediate leads, and annual EKE separately. Annual EKE uses anomalies
  around a sequence's own annual mean; it is not the monthly fixed-lead EKE
  term and is excluded from both headline scores.
- Preserve the original integrated-plus-spectral validation selection. No
  checkpoint reselection or hyperparameter tuning uses these annual outcomes.
- Include matched day-30/day-365 maps with common support and scales at two
  pixels per grid cell.

Three January starts provide a limited diagnostic. Differences are descriptive;
they do not establish general climate skill, seasonal robustness, or seed uncertainty.
The 30-day values here also differ in cohort and aggregation from the previous
96-origin monthly table. All new comparisons use the same Torch evaluator/runtime.

## Execution

Evaluation producer: `25cc663fd4163ccfc0f056edc343732a484aa6e7`. The model and numerical annual evaluator are byte-identical to the earlier `5f61785a` producer. Ten targeted tests pass, covering annual inference, MSE pooling, calendar-month pairing, nonfinite errors, and complete result-file hashes.

Root: `torch:/scratch/jr7309/runs/2026-10-08-presentation-annual`.
Selected checkpoints are copied from retained Beta sources via Torch's DTN and
checked by full SHA256 read-back. The CPU audit checks selected checkpoint lineage,
training normalization/data hashes, contiguous annual inputs, finite forcings,
and exact agreement of the first 30 days and monthly OHC with prepared samples.
GPU evaluation requires that audit to pass. Original checkpoints are mounted read-only.

[Exact model/checkpoint paths and scoring protocol](artifacts/presentation-annual-2026-10-08/paths.json).
The [collector](../../../scripts/collect_presentation_annual.py) verifies result
hashes and matching references before combining the two evaluation roots.

The earlier annual verifier mistakenly compared file suffixes against `json`/`npz` without leading dots, producing empty output-hash lists. This producer requires every named result file and hashes it explicitly. A separate CPU read-back audit validates the existing ten early/fine annual outputs and writes `VERIFIED-OUTPUT-HASHES.json`, preserving their original weights, outputs, and audit records. This repairs provenance verification; it changes no forecast or metric.

Ruff, mypy, schema validation, secret detection, and ten targeted tests pass. The repository-wide REUSE hook has 47 pre-existing missing-license findings in unrelated report artifacts; new files include their licenses.

**October 8 recovery:** audit **19447968 passed**, rehashing all 48 transferred
source files, repeating the full data audit, and validating the ten existing
annual outputs. Evaluation array **19447970** has begun real inference; its
remaining tasks are waiting for RTX capacity (twelve tasks, at most four concurrent
GPUs). The existing early/fine array **19446317** completed all ten tasks; five
belong in this presentation comparison.

The first audit **19447443** passed its data checks but failed at an unnecessary
module-load command. Its dependent array **19447444** never started and used
zero GPU-hours. The replacement changes only the launcher guard; numerical code,
checkpoints, and evaluation protocol are unchanged. Original records are retained.

[Original submission](artifacts/presentation-annual-2026-10-08/submission.json) ·
[Recovery and completion callbacks](artifacts/presentation-annual-2026-10-08/recovery-submission.json).
Callbacks **19448008/19448129** and the supervised SSH relay are installed. The
final report will combine the twelve new models with the five original early/fine
controls after checking complete outputs and hashes for every model.
