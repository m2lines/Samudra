<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# One-step observation-training pilot

**Status:** implementation and qualification in progress. No scientific results yet.

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

Campaign root:
`/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-observation-fast-v1`.

Sources: [objective](../../../src/samudra/experiments/diffusion_observation_fast.py),
[trainer](../../../src/samudra/experiments/diffusion_interventions.py),
[runner](../../../scripts/run_diffusion_observation_fast.py),
[Slurm launcher](../../../scripts/slurm_diffusion_observation_fast_beta.sbatch).
