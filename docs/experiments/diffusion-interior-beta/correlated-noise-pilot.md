<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small matched spatial-noise experiment

Authorized October 1: **one seed, two fresh OM4-only runs, 2,000 updates each**.
The comparison isolates spatial noise covariance under a small total training
budget. It does not continue the 23,500-update selected model and does not use
observation fine-tuning. Status: implementation and qualification checks in progress;
no scientific result yet.

Both arms start from the identical saved scratch step-0 weights in
`latent-d192-v2/D-1729/om4/best.pt`. The checkpoint's actual state is verified to
be step 0 before any fitting. The retained large-run checkpoints were at
23,500/24,000 updates; no usable small-budget trained snapshot was found.

| Choice | White control | Correlated Gaussian |
|---|---|---|
| Marginal noise variance at each wet cell | 1 | 1 |
| White variance fraction | 1 | 0.5 |
| Independent Gaussian-filtered variance fraction | 0 | 0.5 |
| Gaussian filter | None | 5×5, standard deviation 1 grid cell |
| Training | 2,000 OM4-only updates | 2,000 OM4-only updates |

The filtered component uses periodic longitude and zero-padded latitude;
normalization uses the squared kernel and wet mask so coastal and polar wet
cells also have unit variance. Channels and the two decoded temporal slots have
independent noise. The retained white component gives a full-rank covariance
on the wet cells. This is a grid-scale diagnostic, not a fixed-distance or
basin-connected physical covariance model. Model targets and outputs are not
spatially smoothed.

Both training corruption and initial sampling noise use the specified covariance.
For additive Gaussian corruption with constant covariance C, the exact denoiser
D gives the score C^-1(D-x)/sigma^2; multiplication by C in the probability-flow
ODE cancels it, leaving the existing derivative (x-D)/sigma. Thus the same Heun
update is appropriate. The scalar EDM preconditioning remains a valid network
parameterization, and noise retains unit marginal variance. This experiment
keeps the explicit input/output bypass, internal skips, noise-level schedule,
loss weighting, and 16-step sampler fixed.

Both arms consume two Gaussian arrays per noise draw, discarding the second in
the white arm. This aligns all subsequent random noise levels and base draws
across arms. The default white path outside this pilot preserves its original
single-draw behavior. Data order, initial parameters, AdamW settings, precision,
mask, forcing and architecture match. Initializer, processor and decoder train
jointly; the unused ERA5 adapter stays frozen. Existing observation normalization
is reused, as in the original OM4 training; no observation targets enter the loss.

## Evaluation

Use the fixed final 2,000-update weights in both arms, not a checkpoint selected
on the test maps. Retain eight day-30 members on three held-out native January
origins (2015, 2018, 2021), showing SSH, SST and T/S at 550 m. Compare individual
members and means, neighboring-cell roughness, member spectra, physical error,
CRPS and calibration. This pilot addresses grain at a trained rollout horizon,
not annual stability or observation-transfer skill.

The native validation denoising losses are logged for optimization diagnostics;
the two corruption distributions make their raw values unsuitable as a direct
cross-arm skill ranking. Save intermediate weights at 1,000 and 2,000 updates.
The fixed three-date cohort is a diagnostic sample, not a broad significance test.

Two single-L40S jobs have a two-hour allocation cap each, including preparation
and evaluation (at most four newly allocated GPU-hours before any recovery).
Training checkpoints permit resume after preemption without resetting updates.
The existing campaign total before this pilot is 223.5028 / 576 GPU-hours.

Implementation: [noise generator](../../../src/samudra/experiments/diffusion_noise.py),
[training and member export](../../../src/samudra/experiments/diffusion_correlated_pilot.py),
[Slurm launcher](../../../scripts/slurm_diffusion_correlated.sbatch).
