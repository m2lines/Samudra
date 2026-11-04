<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Shared-input stochastic initializer capacity sweep

This sweep tests whether jointly processing conditioning and noise, with more
initializer capacity, improves the spatial structure of sampled ocean interiors.
The reference is the completed learned-noise control in
[joint-noise-results.md](joint-noise-results.md). Its coarse-noise processor is
separate from its conditioning U-Net. Its marginal scores do not establish that
individual samples have realistic currents.

All four new initializers concatenate the same inputs before the first Samudra
ConvNeXt block: surface history and ERA5 conditioning, four coarse Gaussian
channels upsampled from 45×90, four independent 180×360 Gaussian channels, and a
broadcast 64-element Gaussian vector. The coarse and fine fields pass through
the same learned network. The U-Net retains its full-resolution skip path.
There is no separate noise processor or FiLM modulation in these four arms.

All three noise sources are mutually independent and resampled for each member
at initialization. Each member then follows its own deterministic forecast
trajectory; no new noise is drawn at later steps. The initializer predicts two
physical state slots with 77 channels each: temperature, salinity and u/v at
19 depths, plus SSH. Direct initializer supervision targets the latest slot.
The processor consumes the state pair, ERA5 and context, predicts the next
five-day state, and shifts the pair. Forecasts use realized future ERA5.

| Run | Initializer widths | Blocks at each encoder, middle, and decoder level |
| --- | --- | --- |
| B-shared | 64 / 128 / 256 | 1 |
| B-shared-deep | 64 / 128 / 256 | 3 |
| B-shared-wide | 96 / 192 / 384 | 1 |
| B-shared-deep-wide | 96 / 192 / 384 | 3 |

The shallow initializers have seven complete ConvNeXt blocks; the deep variants
have 21. The deterministic processor remains a seven-block 64/128/256 Samudra
U-Net. Its initial weights are identical across the four new arms. Each processor
is then trained jointly with its initializer. The completed reference is reused;
its processor initialization is not forced to match the new arms. That comparison
changes noise inputs and architecture together. The four new arms isolate
capacity differences within the shared-input design.

Each fresh run has 2,000 optimizer updates: 1,000 OM4 and 1,000 observation
updates, effective batch eight, four training members, seed 1729, learning rate
0.0001, and EMA half-life 100 updates. Data, forcing, masks, objectives and source
schedule match the previous screen. Training uses existing normalization,
shared ERA5 through the existing eight-to-three adapter, 19 five-day surface
history bins, two physical state slots, and 77 physical channels. There is no
replay or cooldown in this screen. The initializer CRPS coefficient remains 0.1;
forecast and monthly objectives retain their existing 0.5 coefficients.

The first 24 updates of each run qualify speed, finite optimization, source
counts, memory and measured GPU utilization above 50%. These updates count
toward 2,000. Qualification jobs stop before full training. Production resumes
the saved model, Adam, EMA and RNG states only after all four estimates have
been checked against a 40 GPU-hour campaign cap. The cap includes qualification,
training, data staging and evaluation; individual Slurm time limits enforce the
released allocation budget. Failures and retries count toward the same cap.

Initializer evaluations use raw and EMA checkpoints at updates 500, 1,000 and
2,000, eight members, and the same three held-out dates as the reference. Final
30-day forecast evaluations use the same nine observation origins. Compare
mean and member RMSE, CRPS accuracy and spread terms, coverage, and the existing
five-component climatology-normalized forecast blend (SST, SSH-derived velocity,
EKE, and upper/deep ocean heat content; see the reference report). Plot velocity samples,
spectra, vorticity and divergence. Repeat the fixed-noise, changing-conditioning
diagnostic; sensitivity alone is not evidence of realistic uncertainty. This is a one-seed screen with three initialization
cases, so any improvement needs confirmation before a general claim.

Select follow-up work using both probabilistic skill and member structure. Extra
fine-scale power alone is not success. Report parameter counts, measured compute
and training exposure alongside scores. GAN tuning is outside this sweep.

Execution sources are `scripts/run_stochastic_shared_noise.py` and
`scripts/slurm_stochastic_shared_noise.sbatch`. A hashed source snapshot and
qualification/release receipts accompany the remote campaign. Runtime and
scientific results remain unmeasured until those jobs finish.

## Conditional follow-up: OM4-only reconstruction

If the shared-input sweep does not produce useful improvement, isolate the
initializer by training on OM4 reconstruction alone. Remove the processor,
forecast losses and observation batches. Use OM4 SST and SSH history as inputs
and directly supervise the OM4 interior at the final input bin. Initially retain
the current 19 five-day bins, including SST/SSH at the target time. The proposed
first task is narrower: predict only surface u/v from this history; neither target velocity
field is supplied as an input. Keep any ERA5 conditioning identical across
controls. Use temporal holdouts; report ensemble-mean and member RMSE, CRPS,
coverage, spectra and individual-member maps against the OM4 reference.

Compare a deterministic reconstruction model trained with MSE and a stochastic
model trained with CRPS, with matched conditioning, backbone capacity and data
exposure. Include a small training-set fit check before interpreting held-out
failure. This separates reconstruction learning from interactions with forecast
training and observation-only supervision. Surface velocity recovery is a
simpler experimental setup, but coarse SST/SSH may still leave substantial
uncertainty about currents. This is a proposed fallback, not an additional
submitted campaign; choose its compute budget after reviewing the current sweep.
