<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Experimental plan

## Objective and baseline

Test whether joint probabilistic interior reconstruction and latent evolution
improve observationally grounded ocean prediction. Freeze the model/checkpoint,
normalization, forcing adapter and scoring contracts chosen in the upstream
observation campaign before production. Reproduce its validation metrics on beta
before attributing any differences to diffusion. The prior one-channel diffusion
pilot is motivation, not the baseline: its ensemble was substantially underdispersed.

## Main comparisons

| Arm | State passed to dynamics | Decoder/supervision | Purpose |
| --- | --- | --- | --- |
| Reference | Selected upstream physical state | Unchanged selected model | Reproduce upstream baseline on beta |
| A | Two jointly reconstructed physical states | Deterministic decoder, interior MSE | Matched multivariate initialization control |
| B | Two jointly sampled physical states | Conditional diffusion; denoising gradients reach surface initializer | Probabilistic initialization with the same frozen physical dynamics as A |
| C | Persistent latent state | Deterministic decoder; initial/future interior MSE reaches initializer and processor | Test latent evolution |
| D | Persistent latent state | Conditional diffusion decoder; initial/future denoising loss reaches initializer and processor | Test diffusion supervision of the recurrent representation |
| H0 | Same as D | Matched extra one-degree diffusion targets | Control for H1's target schedule and training exposure |
| H1 | Same as D | Add half-degree OM4 diffusion targets during pretraining only | Test higher-resolution target supervision without higher-resolution inputs |
| E | Sampled initial latent per member | Evolve each latent member separately | Conditional follow-up testing propagated initialization uncertainty |

A/B share capacity/conditioning and frozen physical dynamics; sample the two
initial states jointly, not independently. C/D share initializer and processor
architecture and train end-to-end. No full-interior teacher encoder or fixed
latent label is required. A teacher-encoder experiment is a diagnostic only.
Condition the diffusion decoder on the latent throughout its blocks/resolutions.
Ordinary denoising training uses a sampled noise level, not backpropagation
through the full sampling chain; future targets do require gradients through
intervening ocean timesteps.

## H: targets only, before E

H0/H1 must share **identical** surface-history tensors, past and future forcing
tensors, forcing normalization/adapter, latent spatial grid and width, observation
files/operators, state scaling, train/validation/test dates, optimizer settings,
seeds and target-event schedule. Both start from the same declared initialization
and use the same resolution-capable decoder and parameter count. Do not load
half-degree forcings or half-degree surface histories into the initializer or
processor, including via a shortcut conditioning path.

At auxiliary target events, H1 uses native half-degree T/S/U/V fields at exactly
the matched OM4 timestamp; H0 uses one-degree targets at that same timestamp.
One-degree target events remain in both arms. Start with a predeclared 50:50
base/auxiliary event mix. Match updates and examples, log FLOPs/walltime/GPU-hours,
and report the extra half-degree decoding cost. A compute-matched continuation
of H0 can follow if that difference materially affects the conclusion.

Only the decoder's target/noisy-state spatial extent and target-coordinate/mask
metadata change; the recurrent latent does not. Use a resolution-capable shared
head with target coordinates and registered spherical grid mapping. Do not assume
Gaussian-latitude half-degree cells are exact 2x2 subdivisions of the one-degree
grid. Losses are wet-area normalized per channel then balanced across variable
groups; four times as many target pixels must not silently quadruple the loss.
Use the same physical state scales, not half-degree-specific statistics.

Demonstrate that paired H0/H1 input and forcing hashes are identical and latent
shapes/parameter counts match. The observation fine-tuning data, forcing adapter,
resolution and scoring procedure remain unchanged in both arms. Score primarily
on that unchanged observation task and one-degree model-world task; half-degree
reconstruction is an auxiliary diagnostic. This is a test of additional spatial
supervision from a finer **regridding of OM4**, not an independent high-resolution
simulation or new independent time samples.

## Data, supervision and evaluation

Use the upstream one-degree grid, five-day intervals, 19-frame surface/forcing
history and temporal splits. Pretraining targets include all 19 levels of T/S/U/V;
SSH is included where required by the physical-state interface. Anchor known
initial SST/SSH. OM4 pretraining retains original OM4 flux forcings; observation
training retains the upstream ERA5 adapter. No future observed surface forcing.

Retain upstream monthly interior aggregation, missingness masks, observation
normalization and versioned scoring. Do not turn monthly IAP values into daily
labels or zero-fill unsupervised interior channels as ground truth. Before the
observation diffusion qualification, explicitly verify the observation-operator
loss: a masked monthly mean is not a fully observed instantaneous diffusion target.
Sampling/aggregation and gradient estimators must be documented and paired with
the deterministic control; no complete interior data are assumed at inference.

The prepared observation-training objective samples independent complete trajectories,
applies the existing duration-weighted monthly operator to each member, then
scores only observed T/S and the existing five-day SST/SSH bins. Use fair CRPS
with at least two independent members: mean absolute error minus the unbiased
pairwise spread correction. This avoids the small-ensemble shrinkage bias of
empirical CRPS; an individual fair estimate may be negative. Observation masks,
physical scales and 0.8/0.1/0.1 interior/SST/SSH weights match upstream. Retain
OM4 denoising replay for velocity and deep channels without observation labels.
The prepared A/B fine-tuning runner defaults to an additional matched OM4 replay
example every fourth observation update, weighted 0.1; both arms use the same
replay dates and schedule. These defaults must be frozen with the production
caps after qualification. The ERA5 adapter trains while the OM4 physical stepper
stays frozen. Final selection uses the unchanged upstream validation criterion
on separately evolved ensemble means, with calibration reported alongside it.
This changes the stochastic arm's objective from squared error to a proper
probabilistic score; report that distinction and the ensemble-mean squared-error
metrics rather than attributing any gain solely to architecture. Qualification
must still establish real-grid sampling cost, stable gradient flow through frozen
physical dynamics, and sampling-step convergence before production.

Report initialization and days 5/15/30, next-month interiors, and the upstream
one-year autoregressive diagnostic (no year-long backpropagation). Compare point
RMSE/anomaly correlation, CRPS/coverage/ranks, depth/cross-variable covariance,
stratification, velocity structure, spatial and temporal variability, native-cell
maps of fixed members, and inference cost. Preserve validation-only selection;
use held-out data for reporting. Existing repeatedly inspected cohorts are a
development benchmark rather than a pristine new confirmation set.

Controls: inferred persistence, true-interior physical dynamics, removed/shuffled
latent conditioning across noise levels, validation-only spread calibration;
optional full-interior encoder reconstruction/rollout to diagnose representation
versus initialization. D's independently decoded marginals do not establish
coherent sample trajectories; that is E's question.

## Execution stages and budget

1. Stage and verify data, wait for final upstream baseline, transfer/checksum its
   weights/config/scales and reproduce its validation behavior on beta.
2. Qualify A/B: gradient reach, small-set fitting, strict save/reload/resume,
   correct masked multivariate losses, memory/throughput and inference sampling.
3. Run A/B, collect a report; use evidence to size C/D.
4. Run C/D and seed repeats/targeted diagnostics, collect a report.
5. Run H0/H1 with the targets-only contract; report before conditional E.

Reserve a planning maximum of 24 GPU-hours for qualification, 120 for A/B,
144 for C/D, 144 for H, 72 for evaluation/repeats/recovery and 72 for conditional E:
**576 total**. These are ceilings, not requested jobs. Freeze per-run caps using
measured throughput. Count Slurm allocated GPUs, including unused GPUs in beta's
four-GPU minimum allocation. Prefer independent useful jobs on all four GPUs of a
single node before introducing multi-node communication. Reports separate code
readiness, allocations, observed progress, data provenance and scientific results.
