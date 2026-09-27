<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# From-scratch diffusion wave

Authorized after reviewing A/B. This **supersedes the proposed four-arm width
ablation and residual-correction experiments**. The task is to learn conditional
interior generation directly, not reconstruct an inherited deterministic prediction.

- Initialize the surface-history encoder and width-192 diffusion decoder randomly.
  Train them jointly on OM4 interior denoising, including velocities. The encoder
  output is unconstrained conditioning, with no separate physical-state target.
- Reuse only the pre-observation OM4 dynamics, frozen for downstream forecasts.
  Fine-tune the new initializer on observations with native OM4 replay.
- Run seeds 1729 and 1730. Compare against the unchanged selected deterministic
  baseline, not the modified A decoder from the previous wave. Report differences
  in training effort and frozen/trainable dynamics when interpreting that reference.
- Preserve existing temporal splits, normalization, forcings and observation
  operators/cohorts. No half-degree targets in this wave.
- Qualify real-grid gradients, memory, strict reload and data integrity first.
  Aim initially for 24,000 OM4 and 6,000 observation updates per seed; use measured
  throughput and validation curves to fit training and analysis within three wall
  days. Record any budget truncation explicitly, rather than declaring convergence.
- Report mean accuracy, fair CRPS, finite-ensemble calibration, spatial/depth
  structure, native velocity retention/plausibility and downstream forecasts.
  Preserve native true-interior/persistence controls, fixed-date member maps and
  baseline point-mass CRPS on exactly the same support.

This document authorizes no later wave. The 576 allocated GPU-hour campaign ceiling
continues to include the 21.4211 GPU-hours already spent; all attempts count. W&B
remains offline. Final results, figures and machine-readable evidence will be
published here and linked from PR #895, followed by the requested Slack notification.

## Execution status

Scratch loading is explicit in qualification/training contracts and never imports
source initializer parameters into the trained encoder. CPU tests check that only
evolution weights are inherited. Real-grid L40S qualification passed: finite observation forecasts, nonzero gradients
to the encoder/adapter/decoder, frozen dynamics, and identical loss after strict
checkpoint reload. The observation forward/backward probe took 5.87 seconds and
used 8.08 GiB. Production peaks so far are 12.5 GiB GPU and about 5.3 GiB host.

Both seeds are pretraining. The first 354-second allocations were stopped because
measured end-to-end throughput would have truncated the original ten-hour fitting
cap before 24,000 updates. Their outputs are preserved and costs counted. Identical
seeds/settings restarted with a 26-hour fitting allowance, reserving the remainder
of the three-day window for observation fitting and reporting. No training result
is inferred from qualification loss or these early updates.

The unchanged baseline's matched monthly evaluation completed with composite
0.591343, versus its previously published 0.591421 (0.013% relative difference).
Its predictions pass directly through the reporting adapter without another
learned decoder. Per-origin point-mass CRPS, spatial diagnostics and fixed-date
arrays are exported for comparison. All three annual origins also completed. The copied report contains 207 files
(942 MB), fully checked against source hashes. Baseline monthly interior
point-mass CRPS/MAE is 0.0512984 in the fixed observation-standardized units.

Two 6,000-update observation jobs are queued with a success dependency on both
pretraining runs and a 20-hour fitting allowance each. The runner verifies the
completed pretraining marker and selected checkpoint hash before loading its own
scratch weights. This is the authorized second phase of this wave.

The new summary path checks identical cohorts and observation manifests before
comparing both scratch seeds with this one unchanged reference. Paired year-block
intervals will remain conditional on the fitted seeds and fixed evaluation draws.
Native controls additionally report an unevolved zero-m/s velocity reference;
non-velocity channels are excluded from that reference.


## OM4 pretraining complete

Both seeds completed all 24,000 updates. Their completion markers and selected
checkpoint hashes were verified, and both observation-fitting jobs started from
those selected scratch checkpoints. Best within-seed denoising validation losses
were 0.0913 and 0.1955; these use different fixed noise draws and are not a common
cross-seed forecast-skill score.

The 24-origin native OM4 controls completed for both selected pretrained models.
Their CSV hashes and checkpoint provenance were verified. At initialization,
within ±60° and excluding known SST/SSH, equal-depth-channel ensemble-mean RMSE is:

| Seed | Temperature, °C | Salinity, dataset units | u, m/s | v, m/s |
| --- | ---: | ---: | ---: | ---: |
| 1729 | 0.6391 | 0.1528 | 0.0538 | 0.0462 |
| 1730 | 0.6409 | 0.1677 | 0.0559 | 0.0577 |
| Zero-velocity reference | — | — | 0.0813 | 0.0566 |

Both mean predictions improve u over zero; v improvement is not consistent across
seeds. These are model-world initialization errors, not observed velocity skill or
proof of realistic stochastic samples. The unchanged baseline observation report,
post-fine-tuning controls, calibration and maps remain necessary for the final
comparison. No lift over the selected deterministic baseline is claimed yet.

Completed allocations through pretraining and these controls total **38.7992
GPU-hours**; observation fitting is additional and remains in progress. The initial
short attempts and qualification preemption are included.
