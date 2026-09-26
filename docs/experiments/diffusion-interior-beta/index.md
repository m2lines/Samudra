<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Joint interior diffusion and latent evolution

Authorized 25 September 2026; total ceiling **576 allocated GPU-hours**, including
qualification, evaluation, unsuccessful attempts and repeats. Results are pending.

This campaign depends on the final model/checkpoint decision from thread
`01a0d027-0851-7213-a64e-f1d40d40e64d`, tracked in [PR #892](https://github.com/m2lines/Samudra/pull/892).
The new branch is `codex/diffusion-interior-beta`, branched from
`codex/d-observation-pilot` at `67ce359ef`. The upstream campaign remains separate.
The upstream observation comparison is complete and its transfer checkpoint is
the selected reference. Execution moved to Engaging on September 26 with the
user-approved local OM4 releases; the original branch and report links are retained.

## Contents

- [Experiment plan and fixed contracts](plan.md)
- [Data staging, provenance and readiness](data.md)
- [Baseline handoff and execution status](progress.md)
- [A/B: full-state initialization results](#ab-full-state-initialization)
- [C/D: latent evolution results](#cd-latent-evolution)
- [H: half-degree diffusion-target results](#h-half-degree-target-supervision)
- [E: sampled latent initialization results](#e-sampled-latent-initialization)

## A/B: full-state initialization

Upstream baseline and observations verified on Engaging. Baseline, observation-
gradient, native-reader parity and end-to-end runner probes passed. Production
OM4 pretraining for A/B completed 12,000 updates at each of two paired seeds.
All four observation runs also completed 6,000 updates. Held-out reporting is
underway after a readiness-gate correction. [Pretraining sampler sensitivity
records](trained-sampling.json).
No held-out comparative results are available yet.

## C/D: latent evolution

Pending the A/B report and implementation qualification. No comparative results.

## H: half-degree target supervision

Runs **before E**. Only the diffusion targets change resolution; inputs,
forcings, the latent grid and observation targets remain fixed. No results yet.

## E: sampled latent initialization

Conditional extension after H and latent reconstruction/evolution diagnostics.
No results yet.
