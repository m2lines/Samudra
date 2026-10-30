<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Noise structure and velocity learning

This experiment tests whether structured input randomness or learned spatial
processing of that randomness improves the weak currents in stochastic
initializations. The existing [placement diagnostics](placement-diagnostics.md)
show that ordinary CRPS members lack strong currents across the whole field;
displacement alone cannot explain the amplitude deficit.

## Matched training

Each arm starts from scratch and receives 2,000 mixed optimizer updates:
1,000 OM4 and 1,000 observation updates, effective batch eight. The seed is
1729, the training ensemble has four members, and evaluation uses eight.
All arms use model B, a stochastic Samudra initializer and deterministic
Samudra processor, with widths 64/128/256. Training uses the existing fair
CRPS objective and weights, AdamW, LR 1e-4, and EMA half-life 100 updates.
There is no replay, cooldown, or additional mean-MSE loss in this comparison.
The 1-degree data, train/validation splits, shared ERA5, normalization, masks,
19-frame surface history, and forecast task remain fixed.

| Arm | Spatial randomness | Learned noise processing |
|---|---|---|
| B-control | Four IID channels on a 45×90 grid | Existing FiLM heads |
| B-multiscale-noise | Independent Gaussian channels with widths 0/1/2/4 coarse cells | Existing FiLM heads |
| B-noise-processing | Same draws as B-control | Three width-32 ConvNeXt blocks before existing FiLM heads |

All arms retain the 64-dimensional global Gaussian vector. The multiscale
widths are Gaussian standard deviations, one per channel; zero leaves that
channel unfiltered. Filters wrap longitude and truncate at latitude boundaries. Their expected
variance is matched to the IID control at each FiLM injection resolution;
individual draws are not renormalized. The learned branch processes only the
noise field, leaving the physical-feature backbone unchanged. Its final
residual projection starts at zero, preserving the initial control function.
This is a narrower test than adding joint physical-feature/noise blocks.
The processing arm adds 172,804 parameters to the 22,700,874-parameter control.
Its comparison tests the added branch as a package, including its capacity
and compute; it does not isolate those from spatial processing.

Each single-GPU job verifies node-local data, executes 24 real training
updates, checks finite raw/EMA weights and losses, and requires measured warm
GPU utilization above 50% before resuming to 2,000. These 24 updates count
toward the budget. Checkpoints are saved every 500 updates. The process-local
THP opt-out and cuDNN compatibility workaround are retained.

## Evaluation

At 500, 1,000 and 2,000 updates, evaluate raw and EMA initializations on the
same three held-out, exactly aligned OM4 histories used in the earlier report.
Score all 19 depths and export eight-member velocity maps at levels 0/9/18
at the final checkpoint. Evaluate the unchanged nine-origin observation
forecast blend at 2,000 with both raw and EMA weights.

Report velocity mean RMSE, member RMSE, fair CRPS and its terms, spread,
coverage of the eight-member min/max range, and errors in temperature/salinity/SSH.
Compare full-region member speed distributions, mean squared speed u²+v²,
individual-member spectra, and neighborhood exceedance fractions at radii
0/1/2/4/8 cells for speeds 0.03/0.1/0.3/1 m/s. Means and
members remain separate. Neighborhood scores combine event frequency and
placement and cannot isolate a displacement mechanism on their own. Retain
the previous multiscale-loss and MSE runs as context, separately from the
new matched control.
One training seed and three held-out OM4 histories make this an exploratory
comparison; gains need confirmation across more seeds and dates.

## Information diagnostic

A separate four-arm probe crosses the original inputs versus inputs augmented
with contemporaneous low-pass gold OM4 velocities, and fair CRPS versus
per-member MSE. Each starts from the earlier campaign checkpoint
`/scratch/jr7309/stochastic-transitions-20261027/production/B/step-02000.pt`,
using raw weights at 2,000 updates. The probe
retains its Adam history, resets EMA, and trains only initialization for 128
OM4 updates, effective batch eight. All arms receive an identical
zero-initialized adapter; standard-input arms feed it zeros.
The oracle is the latest history-bin gold velocity at all 19 depths, smoothed
with Gaussian standard deviation two output-grid cells and renormalized by
wet-cell area. Longitude wraps; latitude does not.

The added velocities are a target-derived oracle, not an available observation
or a usable forecast. This probe asks whether stronger conditioning changes
the optimization response under the two losses. It cannot establish the best
possible skill from the original observations. Retain the low-pass input
itself as a reference and score the same held-out dates; training uses only
the existing OM4 training windows.

## Evidence and follow-up

Frozen source fingerprints, data receipts, contracts, GPU utilization, W&B
identifiers, checkpoint counts, and scheduler IDs accompany the runs. Live
qualification measured 7.35–7.47 seconds per mixed update on one RTX6000,
or about 4.1 hours for 2,000 updates, excluding startup and evaluation. The short
oracle probe runs on a fourth GPU if available. No larger follow-up is implied
by this plan.

Submitted on October 30, 2026:

| Work | Torch job | Output root |
|---|---|---|
| Three 2k noise arms, array indices 0/1/2 in table order | 21423135 | `/scratch/jr7309/stochastic-noise-20261030` |
| Four 128-update information probes | 21423395 | `/scratch/jr7309/stochastic-information-20261030` |

Each root contains `SOURCE.json` with per-file hashes and `paths.json` with
the frozen data configuration. CPU-only after-any callbacks are jobs 21423139
and 21423435 respectively. At 10:42 EDT all four GPU allocations were running.
All three noise arms passed the 24-update qualification, with 82.9–83.8%
mean GPU utilization, finite losses and gradients, and less than 0.001 seconds
mean data waiting per update. All four information probes subsequently
completed 128 updates in a total of 21m54s; see the
[information-probe results](information-probe-results.md). Their callback is
retired. All three noise experiments subsequently completed training and raw/EMA
evaluation with exit code zero, using 13.5164 GPU-hours in total. See the
[noise-ablation results](noise-ablation-results.md). The source and diagnostic
changes passed 69 CPU tests.
