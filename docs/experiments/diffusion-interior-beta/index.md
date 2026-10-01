<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Joint interior diffusion and latent evolution

Authorized 25 September 2026; total ceiling **576 allocated GPU-hours**, including
qualification, evaluation, unsuccessful attempts and repeats. The A/B,
physical-state scratch, artifact-diagnostic and persistent-latent waves are complete.
Allocated compute through the stopped-run report and performance investigation is
**223.5028 GPU-hours** of the 576-hour cap, including the endpoint-map follow-up.
[The October 1 stopped continuation report is complete](latent-extension.md).
Monthly composites improve to 0.743/0.757, but annual forecasts become severely
unstable and monthly ensembles become underdispersed. Observation training used
±60° support and is not matched to newer global deterministic controls.
The full observation-training comparison awaits review.
[A small matched OM4-only noise pilot](correlated-noise-pilot.md) is authorized
and running: one seed, 2,000 updates each with white versus correlated Gaussian noise.
[Completed performance tests](training-performance.md) measured 1.96× faster L40S
updates or 4.1× versus the original L40S setup using an H200 and opt-in code changes.

This campaign depends on the final model/checkpoint decision from thread
`01a0d027-0851-7213-a64e-f1d40d40e64d`, tracked in [PR #892](https://github.com/m2lines/Samudra/pull/892).
The new branch is `codex/diffusion-interior-beta`, branched from
`codex/d-observation-pilot` at `67ce359ef`. The upstream campaign remains separate.
The upstream observation comparison is complete and its transfer checkpoint is
the selected reference. Execution moved to Engaging on September 26 with the
user-approved local OM4 releases; the original branch and report links are retained.

The preceding short-budget wave is [persistent latent evolution with diffusion readouts](latent-wave.md).
Both seeds finished 24,000 OM4 updates and approximately 2,000 observation updates
under the time cap. The saved salinity examples lose the conspicuous diagonal
artifact, but test composites (1.019 and 0.961) remain worse than the deterministic
reference (0.591). Interior fair CRPS improves 8–11%, while native velocity skill
and annual stability deteriorate. The report includes individual-member spectra,
calibration, before/after native controls, annual forecasts and provenance.

The [previous physical-state diffusion results](scratch-wave.md) remain available:
23–24% interior CRPS improvement, with worse point scores and velocity retention.
## Contents

- [Small matched spatially correlated-noise pilot](correlated-noise-pilot.md)
- [Day-30/day-365 members and means: observation and OM4 inputs](endpoint-maps.md)
- [Completed stopped-run report: monthly lift, annual instability](latent-extension.md)
- [Completed training performance investigation](training-performance.md)

- [Completed latent-state results](latent-wave.md)
- [Completed physical-state from-scratch results](scratch-wave.md)
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

[Read the completed persistent-latent report](latent-wave.md). Both seeds and all
monthly, native and annual evaluations completed. These runs jointly trained the
latent processor and diffusion decoder from scratch; readout noise is independent
at each lead, not a coherent sampled latent trajectory. The deterministic C arm
was not part of this wave.

## H: half-degree target supervision

Runs **before E**. Only the diffusion targets change resolution; inputs,
forcings, the latent grid and observation targets remain fixed. No results yet.

## E: sampled latent initialization

Conditional extension after H and latent reconstruction/evolution diagnostics.
No results yet.
