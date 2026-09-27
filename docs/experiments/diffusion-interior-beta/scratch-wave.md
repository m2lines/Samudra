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
evolution weights are inherited. GPU qualification and production results are pending.
