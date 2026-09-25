<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Execution status

25 September 2026: preparation in progress; no diffusion training jobs submitted.

- Created `codex/diffusion-interior-beta` from the upstream observation branch.
- Confirmed beta access, B200 partition and a four-GPU minimum for standard/test QoS.
- Located existing half-degree OM4 and observation/annual bundles on beta.
- Started exact-release one-degree data staging and existing-data audits.
- Upstream fresh InstanceNorm OM4 pretraining completed at 14,261 updates, but
  scratch observation production and transfer calibration are still running.
  The final chosen baseline model/checkpoint is not available.
- H0/H1 has been inserted before E; half-degree targets cannot alter forcings,
  surface inputs, latent grid, state scales or observation targets.

## Baseline handoff requirements

Record the upstream chosen model's branch and immutable producer, model class and
normalization, complete run manifest, selected checkpoint and SHA256, final
training completion/selection evidence, observation statistics/grid/score
reference hashes, and corresponding OM4 pretraining checkpoint where applicable.
Preserve architecture/adapter contracts and distinguish the final observation
baseline from its OM4 pretraining source. Use completed validation selection;
do not choose a live `best.pt` or select with held-out scores.

Qualification on beta must strictly reload those weights and reproduce the
upstream validation metrics within a documented precision tolerance. Record any
source-branch changes needed before production on this new branch. Do not mutate
or push the upstream branch.


## Prepared launch gate

`scripts/watch_diffusion_beta.py` checks hourly for completion of both final
InstanceNorm runs (`scratch-main`, `transfer-main`) and their matching monthly
and annual held-out reports. It then takes the lower **final upstream validation
score**, never the held-out score, with identical data/statistics/selection
references required. Live best checkpoints cannot qualify. This implements the
upstream declared selection criterion; a changed model/selection contract must
be reconciled before submission.

The selected full observation model and its manifest, completion, score reference
and checksum are transferred through Torch's DTN to beta and verified again.
The first submitted job is bounded at **4 GPUs x 2 hours = 8 GPU-hours**:

- reproduce the selected baseline's nine-origin validation score on beta;
- ten-update real OM4 full-state deterministic initializer fitting;
- matched ten-update real OM4 full-state diffusion fitting, requiring gradients
  in the surface initializer, improving fixed-noise training loss and strict reload;
- fixed-latent-grid / larger-target-grid synthetic gradient and memory smoke.

The last probe is not an H scientific result; registered target-coordinate
geometry remains required for H production. The new module is qualification
scaffolding, not a completed C/D recurrent processor or observational diffusion
training implementation. A/B production caps and sampling/observation-operator
qualification follow the fitting report. No production runs auto-submit from
this watcher. A durable submission intent prevents duplicate allocation on an
ambiguous SSH response; failures require inspection before resubmission.

Existing-data audits passed: all 350 observation checksums, all 416 annual
checksums, 379,625 half-degree chunks and representative field decoding. Six CPU
regressions cover end-to-end gradient reach at both target sizes, channel/area
weighting and readiness rejection of unfinished reports or mismatched weights,
normalization and selection evidence. One-degree staging remains in progress.


## Deployment update — 25 September, 22:42 UTC

Draft [PR #895](https://github.com/m2lines/Samudra/pull/895) is open and assigned
to jder, with a results table of contents. Qualification producer `9ee94d411`
is pushed and its source archive passed read-back verification on beta.
The hourly dependency watcher is active. Its first live check is waiting for
unfinished upstream training and the exact-release one-degree data verification.
No GPU job has been submitted by this campaign.

Existing-data verification covers 393,574,212,039 bytes of half-degree chunks,
21,888,642,966 bytes of monthly observation payloads and 1,025,294,370 bytes of
annual payloads. The new one-degree copy is progressing. Once its full source
read-back succeeds, a final audit checks cross-resolution timestamps and data
readiness automatically.

The watcher will submit one bounded qualification and monitor it hourly,
collecting all four result records and checking matched A/B input hashes. It
stops for a report/production-sizing decision after qualification; it does not
implement unattended execution of the entire 576-GPU-hour campaign.
