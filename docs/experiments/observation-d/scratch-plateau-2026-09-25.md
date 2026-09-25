<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Scratch versus D: completed 8k extension

**25 September 2026, 00:25 ET.** The observation-only run completed 8,000 joint updates and its original-policy held-out evaluation. Both inference policies stopped improving after 6k. The conditional 16k extension was **not launched**: continued improvement at 8k was the user's condition, and it was not observed. This is an operational plateau decision from one seed and sparse checkpoints, not proof that longer training could never help.

## Main comparison

![Scratch versus D, validation through 8k](artifacts/2026-09-24-budget/batch-stat-curves/validation-scratch-versus-d.png)

| Model | Observation joint updates | BatchNorm inference policy | Integrated + spectral validation score ↓ |
| --- | ---: | --- | ---: |
| D → observations | 3,000, selected | Frozen training statistics | 0.52729 |
| D → observations | 4,000, terminal | Frozen training statistics | 0.54283 |
| Observation-only random | 4,000 | Per-sample statistics | 0.57264 |
| Observation-only random | 5,000 | Per-sample statistics | 0.56456 |
| Observation-only random | 6,000 | Per-sample statistics | **0.55730** |
| Observation-only random | 8,000 | Per-sample statistics | 0.56278 |

**D → observations** uses the original OM4-pretrained initializer and evolution network, then 1,000 observation reconstruction updates and 4,000 observation joint updates, without additional OM4 continuation. **Observation-only random** starts both networks randomly and uses only observations: the same 1,000-update reconstruction phase, followed by 8,000 joint updates. Both have approximately 153.3M parameters and effective batch eight; full recipes and the original D training are in the [main methods](compute-allocation-2026-09-24.md#model-and-training-definitions) and [D history](d-training-history.md#original-d-training-recipe).

Scratch improves 2.7% from 4k to 6k, then worsens 1.0% at 8k. Its best measured per-sample score is **5.7% above D's selected score**. This is substantially closer than the original stored-statistics comparison suggested. It does not isolate the value of pretraining: learning rates, normalization and BatchNorm training recipes differ, and D inherits a substantial earlier OM4 budget. Nor does the gap establish significance across training seeds.

The graph follows each model's training normalization: D was trained with frozen statistics, scratch with sample statistics. Validation uses the same nine origins and integrated-plus-spectral objective at 5/15/30-day leads. Only retained scratch weights can be rescored; the 7k checkpoint was not retained and is not interpolated as a measurement. The [all-model batch-statistics diagnostic](batch-stat-curves.md) includes the sensitivity of D to changing its inference policy.

## Original selection and held-out results retained

The original stored-statistics scratch validation sequence at 6k/7k/8k is **0.84104 / 0.84252 / 0.86768**. Its official best checkpoint remains 6k. The diagnostic also has its best measured score at 6k, but no original selection or held-out output has been silently replaced.

| Model / selection scope | Selected update | Original-policy held-out composite ↓ | Spectral error (dex) ↓ | SST RMSE (°C) ↓ |
| --- | ---: | ---: | ---: | ---: |
| D → observations, through 4k | 3,000 | 0.58605 | 0.30521 | 0.49664 |
| Observation-only random, frozen through 4k | 4,000 | 0.93434 | 0.58911 | 1.05021 |
| Observation-only random, through 8k | 6,000 | 0.91254 | 0.51078 | 1.15008 |

These are 96 independent monthly initializations across 2015–2022, each forecasting about one month, not an eight-year continuous rollout. The scratch rows use the original problematic running-statistics inference policy. **They must not be read as corrected-policy held-out evidence for a large pretraining advantage.** Full 96-origin held-out rescoring under per-sample statistics has not been performed; the corrected comparison above is validation-only and posthoc. Under the original policy, the composite improves 2.3% from the frozen 4k selection, while SST RMSE worsens—another reason to report components rather than RMSE alone.

## Completion and reproducibility

Training job **18409920** completed the full 8k budget and the 96-origin evaluation. Read-only 8k validation job **18488061** completed in 34 GPU-seconds; final CPU anomaly/year audit **18488644** completed in 117 seconds. Hash/cohort checks passed for all six evaluation directories, including the separately frozen random-4k comparison. Total allocated use, including calibration, retries, preemptions and diagnostics, is **42.9886 GPU-hours**, below the 100-hour ceiling. The queue is empty and the monitoring timer remains disabled. Original 4k/6k/8k checkpoints and evaluations remain intact; no checkpoint was deleted or overwritten by rescoring.

[Final original-policy component table](artifacts/2026-09-24-budget/plateau-8k/heldout-table.md), [validation records](artifacts/2026-09-24-budget/plateau-8k/validation-curves.csv), and [batch-statistics records](artifacts/2026-09-24-budget/batch-stat-curves/rescored-points.csv) accompany the figures. The plateau artifact directory also contains compressed selection/accounting summaries, annual diagnostics and full input evidence. Reconstruct the latter by concatenating `evidence.json.gz.part000` then `part001`, then decompressing; run `scripts/summarize_observation_budget.py` on that JSON snapshot to reproduce the original-policy tables. Batch-statistics evidence includes its separate scoring and plotting scripts.

The central result is the much narrower scratch-versus-D gap after correcting scratch inference normalization, followed by no further measured gain from 6k to 8k. Additional-OM4 arms remain supporting evidence that continuation after D was not useful under the tested allocation; they are not the focus or a proof of the globally optimal pretraining duration.
