<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Latent observation-training continuation

Authorized September 29 after the [completed short-budget report](latent-wave.md).
Both seeds continue the same observation-training objective for approximately
48 additional fitting hours. Architecture, OM4 pretraining, datasets, splits,
normalization, learning rate, replay and validation selection are unchanged.

## Why continue

Lower validation composite is better; this is the same frozen eight-member-mean
criterion used for selection, not training denoising loss or held-out test CRPS.

| Seed | Update 1,500 | Update 2,000 | Final short-budget update |
| --- | ---: | ---: | ---: |
| 1729 | 1.17556 | 1.03902 | 1.01125 at 2,026 |
| 1730 | 1.03414 | 0.95980 | 1.01074 at 2,059 |

Both show late improvement over the last full validation interval. Seed 1730's
final check worsened, so improvement is not monotonic. The incumbent best score
is retained; further training need not produce a better selected model. The
short-budget report and its checkpoint files remain unchanged.

## Continuation contract

New run group: `latent-d192-extended-v1`, under
`/orcd/pool/008/jrusak/diffusion-interior-engaging/runs` on Engaging.
Each seed resumes its **last** observation checkpoint, including AdamW state,
CPU/CUDA random state, update count and elapsed fitting time. The existing best
weights remain the incumbent for selection. No optimizer or data-order reset.

The cumulative observation ceiling changes from 20 to **68 fitting hours**, and
from 6,000 to **10,000 updates** so the previous update ceiling does not prematurely
end the extra time. Approximately 48 hours remain after the actual prior fitting
time. Two single-L40S jobs use four CPUs and 96 GiB host memory each. A 46-hour
invocation cap leaves margin inside the 48-hour Slurm allocation; a dependent
second allocation finishes the remaining global budget. Preemption/resumption
does not reset that budget. Queueing, cache startup and evaluation add wall time.

`samudra.experiments.diffusion_extend` writes copies into a new destination and
records parent checkpoint hashes, old/new protocols and prepared hashes in
`CONTINUATION.json`. Only budget metadata and the terminal-state flag change;
the source checkpoints are untouched. The immutable training implementation is
`8291df4dc7600401da4dcd76172eb18a5c7cc6b3`; the migration/evaluation implementation
is `b218131efe839b313b0e67d26e9d8215732d1609`.
A regression test verifies exact model and optimizer equality against an
uninterrupted stochastic training run after a completed-stage extension.
All four extension/resume tests pass.

## Jobs and evaluation

- First training array: **24284649**, seeds 1729/1730, running on L40S at launch.
- Remainder array: **24284650**, dependent on successful first-array completion.
- Native controls: **24284713**, dependent on training completion.
- Monthly and annual report: **24284652**, dependent on training completion.
- Superseded native submission **24284651** was cancelled while pending to correct
  its checkpoint-phase environment; it consumed no allocation.

Both native controls and the same 96-origin monthly/three-origin annual report
are scheduled against the validation-selected continuation checkpoints. Their
outputs go to the new run group, preserving the original comparison. Report
individual-member structure/spectra and temporal consistency alongside mean
scores and calibration. No half-degree targets or sampled-latent arm is added.

The previous campaign total was **121.5853 GPU-hours**. This extension should
add approximately 96 training GPU-hours plus setup, preemption overhead and
about six evaluation GPU-hours, remaining well within the **576 GPU-hour** cap.
Actual allocation accounting takes precedence over this estimate.

[Continuation receipts and job manifest](latent-assets/extension-provenance.json.gz).

At 11:42 UTC September 29, resumed seeds had reached updates **2,029** and
**2,062**, respectively, with finite losses and approximately 36-second updates.
Both native-cache equivalence checks passed, and observed host memory was about
62 GiB per job, below the 96 GiB allocation. This verifies real optimizer
progress after restart, not just scheduler acceptance.

## Monitoring and provisional plateau rule

Check the existing job IDs and validation exports hourly, with additional checks
only around actionable transitions. Assess each seed separately. Flag a
provisional plateau when the best of the latest three consecutive scheduled
500-update validations improves by less than 1% over the best score preceding
that window. Missing checks or an off-schedule final validation do not fill the
three-check window. The reproducible helper is
`python -m samudra.experiments.diffusion_plateau --parent PATH --continuation PATH`.

A flag triggers an interim report, not a claim of convergence or automatic
cancellation of the authorized training. Keep final reporting distinct from
interim validation: collect completed native and held-out monthly/annual outputs,
verify them, and compare with the frozen short-budget baselines. Notify on a
plateau report, completion, or a persistent operational blocker. A failed SSH
check alone does not establish that Slurm jobs stopped and does not justify
restarting them.

For the final comparison, `diffusion_latent_summary` accepts
`--previous-latent-runs` for the original run directories and `--latent-runs` for
the extension. It checks both completed evaluation contracts and the continuation
receipts, rejecting changes beyond the training budgets or a different parent
checkpoint. Write its output into a separate extension-assets directory.
`diffusion_latent_figures` then plots the baseline, physical diffusion, original
latent and extended latent results together with distinct colors and markers.
The original report remains unchanged.
