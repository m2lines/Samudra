<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Joint noise and conditional GAN initializer screen

This screen tests whether the broad velocity perturbations in the learned-noise
initializer improve when randomness is processed jointly with surface inputs,
when independent fine-grid randomness is available, or when OM4 initialization
is supervised with a conditional adversarial objective.

| Arm | Initializer noise | OM4 initializer objective |
| --- | --- | --- |
| B-learned-control | Existing four coarse channels, three width-32 noise-only ConvNeXt blocks, middle/decoder FiLM | Fair CRPS |
| B-joint-coarse | Coarse noise modulates encoder, middle and decoder features | Fair CRPS |
| B-joint-multires | Independent random maps at native, half and quarter resolution; encoder/middle/decoder modulation | Fair CRPS |
| B-conditional-gan | Four full-grid Gaussian maps projected into conditioning before the Samudra U-Net | Paired conditional WGAN-GP generator loss |

All generators retain Samudra ConvNeXt U-Nets with widths 64/128/256. All arms
use a stochastic initializer and a deterministic processor. Independent
initializer members are evolved as separate trajectories. The GAN has no
additional late FiLM injection. The physical grid is 180×360; random feature grids are 45×90 (coarse),
90×180 and 180×360. These are not additional ocean datasets. The coarse and multiresolution arms reuse
noise across encoder and decoder stages at the same spatial scale.

## Training contract

Each fresh run has 2,000 generator optimizer updates: 1,000 observation and
1,000 OM4, alternating, effective batch eight, microbatch one, four ensemble
members in training. This is 8,000 generator-training examples per source.
Seed 1729, learning rate 1e-4, EMA half-life 100 updates, no replay and no
cooldown match the preceding noise screen. Checkpoints are retained at
0, 500, 1,000, 1,500 and 2,000 updates. Evaluation uses eight members, raw and
EMA weights separately, and the existing held-out dates.

The generator update counts and source sampling are matched. Different noise
shapes change the mapping from a random seed to a member, so members are not
identical random perturbations across architectures. New branches preserve
common initial physical-backbone weights. The existing 19-bin surface history,
two physical state slots, 77 fields, shared ERA5 adapter and full-latitude masks
are unchanged; this is not a new data or forcing comparison.

The GAN changes only the OM4 initializer term, with its existing coefficient
0.1. OM4 forecast/monthly and all observation losses remain unchanged.
It sees the latest 77-channel initializer target, matching existing direct
supervision. There is no observed velocity target or fabricated observed
interior target for the critic.

## Conditional GAN adaptation

The objective follows [Perezhogin, Zanna and Fernandez-Granda (2023), Appendix
C.1](https://arxiv.org/html/2302.07984#A3.SS1). A critic compares a generated pair
against pairs containing one generated and one real state under the same
conditioning. The mixed pairs are (generated1, real) and (real, generated2);
their critic scores are averaged against (generated1, generated2). Gradient
penalty interpolates between the generated pair and a randomly chosen mixed
pair, differentiating only with respect to the two physical fields. It uses
gradient penalty 10 and critic drift penalty 0.001.
The generator minimizes minus the critic score of its generated pair.

Five critic optimizer updates use separate training-only OM4 batches before
each OM4 generator update. With effective batch eight, this adds 5,000 critic
updates and 40,000 OM4 example exposures. Those exposures and critic time are
reported separately; this arm is not matched in total computation or data
exposure. Critic optimization does not backpropagate into the generator.
Generator optimization differentiates through a frozen critic. Critic
conditioning is detached from the generator's learned ERA5 adapter so the
adversarial gradient cannot change the condition directly.

Adaptations are the Samudra generator, surface/forcing/geography conditioning,
77-channel joint output, coastline masks and nonperiodic latitude boundaries.
The fully convolutional critic uses widths 32/64/128 and spatial pooling.
This is a method adaptation, not reproduction of the idealized two-layer
subgrid-forcing experiment. GAN versus the other arms changes both the noise
architecture and objective; a gain cannot be attributed solely to adversarial
supervision. Float32 critic and gradient-penalty computation
avoid reduced-precision second-derivative qualification issues.

## Qualification and reporting

The first 24 production updates form a qualification segment and count toward
the 2,000-update budget. Qualification requires finite raw/EMA model state,
finite losses and gradients, correct source counts, saved GAN optimizer state,
and measured warm GPU utilization over 50% across updates 5–24, the threshold authorized for these
short runs. A projected training duration (excluding evaluation) above 14 GPU-hours per arm stops
before continuation for review. The segment is then resumed with Adam, EMA,
critic state and random-generator states retained.

Training and evaluation will use immutable source snapshots and verify node-local
OM4/forcing caches and record container/source/data provenance. Torch preflight found the existing
container and source caches accessible; each job must verify its local copy. Four single-GPU Torch jobs may run in
parallel; no priority QoS is requested.

The first report will include shared held-out initializer CRPS/RMSE curves at
500/1,000/2,000; training critic scores are reported separately. Initializer
evaluation uses January, April and May 2014, and rollout evaluation uses the
existing nine validation origins pinned in the observation manifest. It includes
individual members and ensemble means, CRPS and its components, coverage,
mean/member RMSE, velocity power spectra and 30-day forecast metrics. Follow-up
analysis will inspect vorticity/divergence, cross-variable/depth structure and
fixed-noise changes under different conditioning. A CRPS or power improvement
alone will not establish realistic currents. The prior multiscale-CRPS loss
trial is a separate completed experiment, not one of these new arms.

## Submission receipt

Submitted 2026-11-01 to Torch `rtx6000_lzanna`, array `21616800`
(tasks 0–3 follow the table order), one GPU per task and at most four tasks
concurrent. Initial scheduler state was pending. The separate CPU callback
job `21616806` depends on termination of the whole array. Quiet monitoring is
registered with a first check after 15 minutes; an ETA awaits GPU qualification.

The frozen source manifest contains 305 verified files with SHA-256
`5e06e9dbec2dfeb94f7b5eb01cca5775e2bdea79e66b224728a95b6d457d1377`.
Local validation passed 78 CPU tests and Ruff. GPU qualification and scientific
results are pending. Outputs are under
`/scratch/jr7309/stochastic-joint-noise-20261101/production/`.
