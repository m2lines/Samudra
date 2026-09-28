<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Joint interior diffusion and latent evolution

Authorized 25 September 2026; total ceiling **576 allocated GPU-hours**, including
qualification, evaluation, unsuccessful attempts and repeats. Both A/B and the
from-scratch follow-up are complete. Total allocated compute is **61.86 GPU-hours**;
subsequent waves require approval.

This campaign depends on the final model/checkpoint decision from thread
`01a0d027-0851-7213-a64e-f1d40d40e64d`, tracked in [PR #892](https://github.com/m2lines/Samudra/pull/892).
The new branch is `codex/diffusion-interior-beta`, branched from
`codex/d-observation-pilot` at `67ce359ef`. The upstream campaign remains separate.
The upstream observation comparison is complete and its transfer checkpoint is
the selected reference. Execution moved to Engaging on September 26 with the
user-approved local OM4 releases; the original branch and report links are retained.

The latest completed wave is [direct diffusion pretraining from scratch](scratch-wave.md).
It replaces the width ablation/residual proposal following discussion of the A/B
design. Both the encoder and width-192 decoder start randomly; only dynamics are
inherited. Against the unchanged deterministic reference, both seeds improve interior
fair CRPS by 23–24% but worsen mean accuracy and forecast scores. Fine-tuning loses
native velocity skill, and annual SSH error is dominated by a mean bias. The report
includes calibration, fixed-date maps, native controls and all machine-readable summaries.

## Contents

- [Completed from-scratch results](scratch-wave.md)
- [Experiment plan and fixed contracts](plan.md)
- [Data staging, provenance and readiness](data.md)
- [Baseline handoff and execution status](progress.md)
- [A/B: full-state initialization results](#ab-full-state-initialization)
- [C/D: latent evolution results](#cd-latent-evolution)
- [H: half-degree diffusion-target results](#h-half-degree-target-supervision)
- [E: sampled latent initialization results](#e-sampled-latent-initialization)

## A/B: full-state initialization

[Read the A/B report, controls, maps and next-wave proposal](ab-results.md).

B improves monthly interior CRPS but worsens mean-field accuracy. Native controls
show poor B initialization and the trained decoder has a demonstrated synthetic
noise bottleneck. Neither A nor B beats the selected upstream composite reference.
The originally proposed decoder-repair comparison was superseded by the authorized
from-scratch wave. These historical A/B allocations total **21.4211 GPU-hours**,
already included in the campaign total above.


## C/D: latent evolution

[The authorized latent diffusion wave is running](latent-wave.md). Host-resident
caching passed real-grid qualification on L40S after initial H200 runs were
superseded by preemption and loading overhead. Both seeds completed 24,000 scratch pretraining updates; observation adaptation
and pretrained diagnostics are running. Comparative results
remain pending. The deterministic C arm is not part of this wave.

## H: half-degree target supervision

Runs **before E**. Only the diffusion targets change resolution; inputs,
forcings, the latent grid and observation targets remain fixed. No results yet.

## E: sampled latent initialization

Conditional extension after H and latent reconstruction/evolution diagnostics.
No results yet.
