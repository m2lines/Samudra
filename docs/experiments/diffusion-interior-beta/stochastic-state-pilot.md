<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Observation-initialized stochastic state pilot

Status: production job **215378** completed. See [results](stochastic-state-results.md). This page records the new pilot,
not results from the completed independent-diffusion-readout campaign.

The question is whether uncertainty carried by evolving member-specific latent
states produces plausible ocean trajectories, useful ensemble spread, and gradual
error growth through a year. Every member has its own two-frame latent history.
The physical decoder is deterministic and no decoded ensemble mean is fed back.

| Arm | Initial latent uncertainty | Transition uncertainty |
| --- | --- | --- |
| deterministic | None | None |
| initial | Learned conditional perturbation | None |
| fresh | Same initial mechanism | Fresh learned conditional perturbation every five days |
| ar1 | Same initial mechanism | Temporally correlated perturbations, rho 0.9 per five days |

All arms instantiate identical parameter tensors from seed 1729. Active noise
modules learn state-dependent amplitudes and project eight global and eight
coarse spatial noise channels into the latent state. Coarse noise uses a 6x12
grid with periodic longitude interpolation. Initial noise projection scale is
0.05. The AR1 innovations preserve marginal variance; their e-folding time is
approximately 47.5 days. This is a first, limited conditional noise model, not a
claim that Gaussian latent perturbations capture every relevant mode.

## Matched inputs and training

- Existing observation cohorts: 243 training months, 9 validation months, and
  96 test months; existing normalization and zero-filled missing inputs.
- Same 19-frame surface/forcing initialization, observed SSH/SST, monthly
  interior T/S targets, ERA5 forcing adapter, OM4 one-degree targets/forcing,
  cosine-area weights, and full-latitude wet masks as the existing global setup.
- Fresh weights; width-128 latent on 45x90, four residual processor blocks,
  width-64 deterministic decoder on 180x360. The previous diffusion decoder was
  width 192. Historical scores provide context, not a matched architecture control.
- Effective batch eight per update, two training members, BF16 computation with
  FP32 parameters, AdamW, learning rate 1e-4, clipping 1.0, weight decay 0.01.
- Strict alternating OM4/observation updates. Both use ensemble CRPS plus the
  existing wet-neighbor spatial-increment score (coefficient 0.5), rather than
  an OM4 denoising objective. Deterministic members coincide, reducing their CRPS
  to absolute error. All arms use the same soft latent-growth penalty (0.1).
- Six five-day OM4 forecast steps, plus initial interior and hidden-surface
  completion supervision. No replay in this first matched pilot.
- Observation forecast weights remain 0.8 interior / 0.1 SSH / 0.1 SST, with
  accessory completion and reconstruction at weight 0.1 each.
- The accessory monthly reconstruction now initializes at the first target-month
  surface bin and evolves the rest of the month, using exact calendar weights.
  It no longer independently reinitializes from later observed surface bins.
  This deliberate task change is shared by all four arms; future interior labels
  never enter an initializer.
- Final 20% of updates cool to 1e-6. Qualification runs never initialize production.

The first wave is capped at roughly 40 allocated GPU-hours, including qualification
and evaluation. The selected budget is **500 observation + 500 OM4 updates per arm**.
Qualification measured approximately 8.2 seconds per observation update and
25 seconds per OM4 update: about 4.7 hours of training, plus evaluations and
startup, within an eight-hour four-GPU allocation. Four independent one-GPU arms share one production
Beta host; this is not four replicas of one DDP job. Online W&B is required outside
qualification. No automatic larger training wave is authorized by these scripts.

## Evaluation

Inspect early checkpoints at 125 and 250 observation updates when those precede
the final budget. The cheap screen uses January 2018 and the final evaluation uses
the existing January 2015/2018/2021 annual cohort plus the nine validation months.
The 2018 screen overlaps the final cohort and is not an independent holdout.

Use eight member trajectories and the existing 30/365-day surface and monthly OHC
metrics. Retain mean and single-member maps/errors, regional spatial spectra,
five-day temporal increments, lag-one correlations, and monthly calibration.
Year-long temporal diagnostics include the seasonal cycle unless explicitly
removed; they should not be described as deseasonalized variability.

The comparison asks whether random initialization alone suffices, whether learned
transition innovations help, and whether correlated innovations improve temporal
structure. A short fresh run that is undertrained cannot establish that an approach
fails. Randomness does not itself guarantee stable dynamics or calibrated spread.
Forcing is shared across members, so uncertainty is conditional on prescribed
atmospheric forcing.

## Reproduction

- Model: [stochastic_state.py](../../../src/samudra/experiments/stochastic_state.py).
- Trainer: [stochastic_state_train.py](../../../src/samudra/experiments/stochastic_state_train.py).
- Evaluator: [stochastic_state_evaluate.py](../../../src/samudra/experiments/stochastic_state_evaluate.py).
- Launcher: [slurm_stochastic_state_beta.sbatch](../../../scripts/slurm_stochastic_state_beta.sbatch).
- Campaign root: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/stochastic-state-obs-v1`.
- Data uses links to the completed scratch campaign's verified observation,
  annual-observation, OM4, native-loader, and reference inputs; no data copies or
  historical checkpoint modifications are needed.

Checkpoints retain model, Adam, exposure counts, RNGs, and data/source contracts.
Evaluation receipts identify the checkpoint hash, exact origins, and output hashes.

## Launch provenance

GPU qualification job `215374` passed all four arms, checkpoint resume, gradient
checks, shared initial weights, and eight-member annual evaluation. Qualification
used `stochastic-state-7ddfb25187ca`; production uses
`stochastic-state-727f254bdcb1`, with only typing assertions/variable naming fixes
after qualification. All 229 source files were verified after remote staging.
The 24 focused CPU tests and all pre-commit checks passed. Production receives
fresh weights, not qualification weights. Quiet monitoring checks scheduler,
training events, online W&B receipts, and evaluation completion artifacts.
