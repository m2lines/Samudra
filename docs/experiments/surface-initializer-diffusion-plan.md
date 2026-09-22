<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# One-field initializer diffusion pilot

Authorized: one-field pilot, September 22, 2026. Planning envelope: 8–24 allocated GPU-hours including qualification, a matched deterministic control, sampling, and recoveries. Stop this wave and report before proposing additional training.

Question: does conditional residual diffusion reconstruct a useful distribution of current-time 550 m salinity (`so_9`) beyond frozen D and the deterministic salinity specialist, or merely add plausible-looking but uninformative variability?

## Fixed task and data

Use the exact D surface/forcing history: 19 five-day SSH/SST frames and OM4 wind stress/heat flux over 90 days, geographic and seasonal features, existing normalization, and OM4 one-degree data on Torch at `/scratch/jr7309/data/om4_onedeg_v3`. No new forcing or interior observations are inputs. Training uses the original 2,805 eligible windows (1975–2013), validation the original 11 origins, and held-out evaluation the original 99 origins (November 2014–November 2022). These are model-world reconstructions, not observed-ocean results or forecasts.

Freeze original D. Predict its **current-time single-field residual**, unlike the prior diagnostic fine-tuning loss on both reconstructed times. Supply D's field, monthly climatology, surface/forcing history, context and masks as conditions. Cache D predictions without gradients. Normalize the residual by its exact area-weighted RMS over training origins only. D itself was trained on these dates: training-versus-validation residual mismatch is an explicit limitation to measure, not assume away.

## Arms and optimization

- Conditional diffusion: a three-resolution convolutional U-Net, base width 64, skip connections, noise embeddings and affine conditioning in residual blocks. Longitude wraps; latitude does not. EDM preconditioning with sigma_data=1, lognormal noise with log mean -1.2 and standard deviation 1.2, weighted denoising loss, FP32 state/noise arithmetic around BF16 convolutions.
- Deterministic control: exactly the same network and input conditions, zero noisy-field input and fixed sigma=1, trained against the residual with area-weighted MSE. This distinguishes extra field-specific capacity from distributional modeling.

Both use seed 1729, AdamW at 2e-4, weight decay .01, effective batch eight (microbatch two), clipping at one, EMA .999, identical step-indexed sampled examples and 30,000 updates maximum or four training-loop wall hours, whichever comes first. Validation sampling time counts toward that cap. Save resumable checkpoints every three minutes. Model size and measured throughput are recorded at runtime. No full-state or dynamics weights are updated. Use one RTX PRO 6000 Blackwell per job, with the Rust loader and two CPUs; run the two arms concurrently after qualification.

Select EMA checkpoints every 1,000 updates using validation empirical ensemble CRPS (eight members, 16 Heun intervals = 31 denoiser calls per member, fixed seeds). For the deterministic control empirical CRPS equals MAE. This is a probabilistic selection criterion; also report ensemble-mean MSE. Do not tune on held-out fields. Qualification uses three updates and reduced sampling, and does not supply production weights.

## Evaluation and decision

Run final evaluation in fresh processes to release training caches. Use 16 members and 32 Heun intervals (63 denoiser calls per member). Retain all held-out member fields and the exact common targets, D fields, climatology, masks, dates, and normalization. Compute per-origin ensemble-mean MSE/correlation, individual-member MSE, empirical CRPS, spread, central 90% empirical interval coverage, anomaly amplitude, and spatial spectra/covariance. Compare with D, the previous salinity specialist, climatology, and the matched deterministic control. Examine training residuals versus validation/held-out residuals for underdispersion. Finite-ensemble interval coverage and nine held-out calendar years limit calibration claims.

Show the same native-cell map dates and scales as the previous report, with ensemble mean, fixed-index sample members, spread, and error. Never select the visually best member. Quantify uncertainty in paired score differences with calendar-year block bootstrap; one training seed does not quantify training-seed uncertainty. A good-looking sample alone is not success. Report all terminal attempts and allocated GPU-hours, along with code/checkpoint provenance and reproducible assets in draft PR #885.
