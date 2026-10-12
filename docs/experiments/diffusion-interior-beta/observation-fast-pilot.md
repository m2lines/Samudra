<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# One-step observation-training pilot

**Status:** qualification **212300 passed** all three arms, gradient checks and
paired full-sampler readouts. Production **212306 is running** on four Beta B200s,
started October 12 at 02:34 UTC. No production quality results yet. Implementation
passed 37 targeted tests and all repository checks, including typing, secret
scanning and licensing.

Observation updates currently backpropagate through the 32-step diffusion sampler
for two members, forecast leads, and the separate monthly reconstruction task.
The previous residual continuation measured about 265 seconds per observation
update versus 17 seconds per OM4 update on one B200, at effective batch eight.
This pilot asks whether a cheaper observation objective can preserve the quality
of samples produced by the original sampler. The larger scratch pair remains
pending a revised compute estimate and approval.

## Matched comparison

All arms start from the replay-bound parent at **8,128 observation and 8,128 OM4
updates**, the same parent used in the [direct-processor pilot](direct-processor-pilot.md).
They retain residual latent evolution, OM4 replay and the soft latent bound.

| Arm | Observation objective |
|---|---|
| Control | Existing two-member CRPS through the full sampler |
| Fast | One denoiser call per decoded state; observed-only squared-error surrogate |
| Hybrid | Three fast observation updates followed by one original CRPS update |

Each arm receives **64 observation + 64 OM4 updates**, effective batch eight,
identical sample order and paired seeds, fresh AdamW, and cosine cooldown from
1e-5 to 1e-6. The final checkpoint has **8,192 updates of each task**. Save at
32, 64 and 128 mixed updates. No model selection uses annual test scores.
Data, full-latitude masks, forcings, architecture, correlated noise and OM4
denoising remain fixed. No new denoising-step count is introduced at inference.

## What the cheap objective means

Observations do not provide a complete clean ocean state to corrupt with noise.
Use frozen-parent samples on training examples to supply missing input values.
These cached fields are **imputations, not supervision**. For each example:

1. Replace valid observed SSH/SST entries in the imputed forecast with their
   observed values.
2. Adjust the imputed interior sequence so its calendar-weighted monthly mean
   matches each available interior measurement. With weights `w_t`, apply
   `x_t += w_t / sum(w_s**2) * (observed_mean - sum(w_s*x_s))` only where the
   monthly target is valid. This preserves temporal detail in the null space of
   the averaging operator; it does not treat a monthly observation as daily truth.
3. Add fresh correlated noise, then make one differentiable denoiser call per
   decoded state, conditioned on the current model's latent. Observed targets
   never become encoder inputs or recurrent latent states.
4. Score observed surfaces and monthly averages only. Retain the forecast task
   weights (0.8 interior, 0.1 SST, 0.1 SSH), hidden-surface completion (0.1),
   monthly reconstruction (0.1), and existing wet-neighbor spatial term (0.5).
   Squared errors use the existing physical scales; missing interiors and
   velocities receive no observation supervision.

The objective uses the OM4 lognormal noise-level distribution and EDM weighting.
It changes both the training objective and the number of differentiable
denoiser calls. It is **not an equivalent estimator of the original CRPS**, nor
an exact joint denoising likelihood for these incomplete monthly observations.
The frozen imputation distribution can bias learning. The hybrid tests whether
occasional original updates help maintain full-sampler behavior. Initial cache
generation, storage, and compute must be reported separately, including
amortization over this short pilot.

## Qualification and evaluation

Before production, test projection identities and masks; verify finite gradients
reach initializer, processor and decoder; exercise both objectives in the
hybrid schedule; check checkpoint reload and provenance. Cache construction must
reject validation/test examples and bind parent/data fingerprints.

Evaluate with the unchanged **32-step sampler and eight members**. Use the
existing validation cohort, native OM4 probes and three annual observation
origins (January 2015, 2018, 2021). Report day-30/day-365 blended error and its
components, member errors and spectra, ensemble spread/CRPS, latent growth and
representative maps. Do not compare surrogate training loss numerically with
CRPS training loss. The purpose is a cheap early signal about speed and retained
forecast quality, including long-rollout degradation; 64 observation updates
cannot establish from-scratch equivalence or calibrated joint trajectories.

## Execution budget

Beta production: one node with four B200s. Run the control while preparing the
shared frozen-parent cache on a separate GPU, then start fast and hybrid arms.
Use the short-test queue only for qualification. Production wall-time limit is
eight hours; total ceiling including qualification and recovery is **40 allocated
GPU-hours**, enforced using Slurm accounting. All allocated GPUs count, including
idle capacity. Production uses online W&B. Register durable completion follow-up
and hourly fallback checks.

Qualification runs eight mixed updates per arm, verifies finite nonzero
initializer/processor/decoder gradients, reloads the resulting checkpoints, and
compares finite two-member 32-step readouts against the parent on one validation
month. These checks do not select a checkpoint or establish a quality result.
The short-test allocation has a one-hour limit (at most four allocated GPU-hours).
Runtime archive: `obs-fast-v1-04fdba652e9e`; implementation commit: `b5d5e3c11`.
Qualification completed successfully in 34m32s, using **2.3022 allocated GPU-hours**.
Its callback is retired. An independently supervised scheduler watcher and hourly
fallback callback are registered for production. The production allocation's
eight-hour limit gives a combined maximum of **34.3022 allocated GPU-hours**.

### Qualification observations

All arms completed four OM4 and four observation updates, with the hybrid using
three fast observation updates and one full update. Initializer, processor and
decoder gradients were finite and nonzero. Every saved checkpoint reloaded and
produced finite fields under the original 32-step sampler.

| Fixed November 2013 validation example, two draws | Forecast CRPS |
|---|---:|
| Parent | 0.053834 |
| Control | 0.053976 |
| Fast | 0.054956 |
| Hybrid | 0.054832 |

Same input fingerprint and sampling seed in all cases. The cheap arm is about
1.8% worse than the control on this one check. Four observation updates and one
validation month do not establish retained quality or a meaningful ranking.

After excluding the first observation update's compilation cost, median update
time was **298.63 seconds control versus 13.23 seconds fast**, approximately
22.6 times faster. These are only three timing samples per arm; they exclude
OM4 work, preparation, startup and evaluation. Preparing the 32-example smoke
cache plus its parent readout took **369 seconds** on a separate GPU. Complete
smoke training subprocess times were 1,937 seconds control, 641 seconds fast,
and 1,112 seconds hybrid. Production will provide the amortized comparison with
all 243 training months cached and substantially more updates.

Campaign root:
`/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-observation-fast-v1`.

Sources: [objective](../../../src/samudra/experiments/diffusion_observation_fast.py),
[trainer](../../../src/samudra/experiments/diffusion_interventions.py),
[runner](../../../scripts/run_diffusion_observation_fast.py),
[Slurm launcher](../../../scripts/slurm_diffusion_observation_fast_beta.sbatch).
