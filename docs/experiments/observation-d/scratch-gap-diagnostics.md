<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# What explains the observation-only gap?

**24 September 2026, 18:44 ET.** The observed gap is not clean evidence that learning
from observations is intrinsically much harder. A fixed-checkpoint diagnostic
identifies a large BatchNorm inference effect before any extra training or search.

## Same weights, different BatchNorm statistics

Every row uses the original validation-selected checkpoint from the matched-budget
comparison. These are nine fixed observation validation origins; the training probe
uses nine deterministically spaced months from the 243 training examples. Both
modes use the same data, normalization, model weights and integrated-plus-spectral
scorer. No held-out cohort or checkpoint selection was changed.

| Model and inference mode | Training-probe score | Validation score | Validation training-objective value |
|---|---:|---:|---:|
| D → observations — stored running statistics | 0.4736 | 0.5271 | 0.00575 |
| D → observations — per-sample statistics diagnostic | 0.9155 | 0.9101 | 0.03270 |
| Observation-only random (4k) — stored running statistics | 0.8473 | 0.8801 | 0.01641 |
| Observation-only random (4k) — per-sample statistics diagnostic | 0.4394 | 0.5726 | 0.00547 |

**Stored running statistics** is the official inference mode. **Per-sample
statistics diagnostic** computes BatchNorm means/variances from each forward's
activations, over its batch and spatial dimensions, instead of using accumulated
running buffers. Forecasting processes one monthly example at a time. This uses
no future observed target values; it changes only inference normalization. Buffer
changes exist only in memory, and the checkpoint files remain unchanged.

For scratch, this reduces validation score by 34.9%, removing **87.1% of the
original 0.3530 validation gap** to D → observations. The remaining gap is 0.0456,
or roughly 8% of the diagnostic scratch score. This is a sensitivity result on one
seed and nine validation origins, not a revised held-out estimate of pretraining
benefit. The diagnostic mode was investigated after seeing the gap and has not
been used to select checkpoints.

The direction of the control is informative: per-sample statistics greatly worsen
D → observations. Transfer was trained with frozen running statistics, whereas
scratch updates BatchNorm and optimizes with per-forward statistics. Each mode
therefore favors the conditions under which its corresponding weights were trained.
A poorly representative running-statistics estimate, or sensitivity to the
train/evaluation normalization change, is a more concrete explanation than simply
saying scratch needs much more data. This test does not by itself distinguish the
causes of that sensitivity or establish a software defect.

## Loss behavior and remaining differences

The training objective is 0.8 × monthly interior T/S error + 0.1 × SST error +
0.1 × SSH error + 0.1 × auxiliary reconstruction error, all on their prescribed
normalized scales. It does not directly optimize the reported spatial spectra or
velocity/EKE diagnostics. Low training loss is therefore not enough to establish
a good reported forecast score.

Under its training-compatible per-sample statistics, scratch's training-probe
objective is **0.00209**, versus **0.01443** with running statistics. Its validation
objective likewise falls from **0.01641 to 0.00547**. Thus the model has learned
weights that can fit the intended objective substantially better than ordinary
inference suggests. Scratch's training-probe score becomes better than transfer's,
while its validation score remains worse; some generalization gap remains, but a
nine-month probe is not a full training-set generalization analysis.

Plausible remaining advantages of OM4 pretraining include dense full-state
supervision, a learned evolution operator and a short-to-long rollout curriculum.
The original fast D run chiefly trained the initializer; it inherited dynamics
selected after about 1.26 million OM4 updates. Scratch learns a 77-channel physical
state interface and absolute next-state mapping from sparse temporal supervision,
rather than training a completely free latent representation. These facts help
explain why the recipes differ, but the present diagnostic prevents attributing
the entire original score gap to those benefits.

## Scope and follow-through

This is diagnosis, not a search for the strongest observation-only model. The
original selections, normalization policy and active 8k training remain intact.
The authorized conditional extension to 16k remains governed by the original
validation objective. The final report will distinguish the original comparison
from normalization sensitivity, and will not present the large original gap as
an isolated estimate of the value of OM4 data.

Diagnostic job **18466058** completed successfully on RTX in 63 seconds. Recomputed
standard-mode validation scores agreed with their recorded values within 0.04%.
Full per-origin loss components, metrics, checkpoint hashes and the diagnostic
script are retained in [the evidence](artifacts/2026-09-24-budget/scratch-diagnostics/evidence.json.gz).
