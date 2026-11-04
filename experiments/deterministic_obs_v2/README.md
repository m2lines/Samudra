<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Deterministic observation campaign

This experiment compares observation-only training with mixed OM4/observation
training for a deterministic ocean forecast initialized from surface history.
Implementation is stacked on the config-based observation harness in
[PR #955](https://github.com/m2lines/Samudra/pull/955).

As of 2026-11-03, all five original GPU qualification arms passed.
**No full scientific training has been submitted.** The new Rust interval reader,
bounded prefetch and DDP path passes CPU tests and is undergoing separate
[scaling qualification](scaling.md) on one and eight RTX GPUs.
Torch array `21786733` completed on RTX PRO 6000 Blackwell Server Edition GPUs.
Peak allocated GPU memory was 5.15–6.61 GiB. Data preparation/loading dominated
update time; the [qualification report](qualification.md) gives measured runtime
estimates for that original serial-loader implementation.
CPU data-preparation job `21784710` completed successfully in 11m54s.
Each GPU arm used one GPU, eight CPU cores, 96 GiB host memory and a two-hour
limit. Scheduler routing permitted preemptible L40S, A100, RTX6000, H100, H200 and
B200 capacity; only RTX PRO 6000 Blackwell was exercised. All jobs used account
`torch_pr_347_lzanna`, preemption-only routing and requeue.

The first GPU attempt, array `21785317`, reached one optimizer update in arms A,
B and C, then failed because the qualification wrapper omitted the shared W&B
logger's required step argument. Arm D was cancelled after allocation and E before
allocation. That attempt consumed 313 allocated GPU-seconds (0.087 GPU-hours).
The wrapper now passes an explicit logging step and has typed logger/config
interfaces so type checking covers this call. The replacement array uses a new
output directory and preserves the failed attempt's artifacts. See the
[attempt receipt](receipts/attempt-21785317.json).

| Arm | Main data, by optimizer update | Initializer widths | Replay | Total parameters |
| --- | --- | --- | --- | ---: |
| A | observations | 128/192/256/384 | yes | 117,473,624 |
| B | 50% OM4, 50% observations | 128/192/256/384 | yes | 117,473,624 |
| C | observations | 280/380/480/520 | yes | 169,773,168 |
| D | 50% OM4, 50% observations | 280/380/480/520 | yes | 169,773,168 |
| E | 50% OM4, 50% observations | 280/380/480/520 | no | 169,773,168 |

Each initializer maps 19 five-day SST/SSH samples, validity, ERA5 history and
geographic/calendar context into two physical states corresponding to the final
two history intervals. For observations, the history covers the 95 days before
the forecast origin; retained surface values come from its final ten days.
Each state has 77 channels:
u/v/temperature/salinity at 19 depths, plus SSH. Measured initial surface values
are retained; missing surface values and the interior are predicted.

All arms use the same full Samudra U-Net evolution backbone, with widths
280/380/480/520, dilations 1/2/4/8, capped GELU, affine InstanceNorm and a 3×3
physical output head. The initializer size comparison holds these choices fixed.
Processor initialization and subsequent RNG state are identical across initializer
sizes for a given seed. The forecast evolves only its own two physical states,
ERA5 and geographic/calendar context; it receives no later ocean observations.

Both sources condition directly on eight ERA5 fields: 10 m u/v wind, 2 m air and
dewpoint temperature, surface pressure, downwelling solar and thermal radiation,
and precipitation. All eight channels use observation-fitted atmospheric scales;
there is no three-field projection. Future ERA5 is prescribed for these experiments.
The [preset](../../src/samudra/configs/observations/deterministic.yaml) contains the
model, loss and optimizer choices.

Observation targets are OISST SST, DUACS absolute dynamic topography (SSH), and
IAP monthly gridded temperature/salinity. Observation training spans May 1993
through July 2013; OM4 uses eligible native windows from 1975 through July 2013.
Depth numbers below are indices in the [OM4 level definition](../../src/samudra/constants.py):
indices 0–13 cover 2.5–1850 m; the complete state extends to 6000 m.

An update accumulates eight monthly forecast windows, normally four fresh and four
replayed. A window has six or seven five-day transitions; overlap weights form
calendar-month means. Replay continuation recomputes a boundary-straddling bin
when needed. Buckets 1–11 follow the fresh month (bucket 0). Arm E uses eight fresh windows. Replay pools are separate for OM4 and
observations, with two detached two-frame states per lead-month bucket and a
maximum age of 250 optimizer updates measured from original initialization.
Forecast and replay transitions cannot cross OM4 calendar gaps. OM4 history uses
19 native samples without inserting a missing-day placeholder; it can span more
than 95 calendar days when an earlier interval gap is present.

The observation loss is `0.5 monthly T/S + 0.25 SST + 0.25 SSH`, plus separate
`0.1 reconstruction + 0.1 missing-surface completion` auxiliaries. Reconstruction
trains the initializer against monthly observed T/S using contemporaneous surface
history in a separate branch; that branch does not feed the forecast. Observation
completion scores deliberately hidden, originally valid SST/SSH values in the two
initial states; naturally missing observations supply no completion labels.
Losses use finite wet support and area weighting. Temperature and salinity each
receive half of the monthly T/S loss after averaging their observed depths.
OM4 uses a variable-balanced dense forecast loss; its reconstruction scores the
two initialized physical states excluding SST/SSH, and its completion uses dense
OM4 surfaces wherever inputs were hidden. Both auxiliary weights remain 0.1. All
physical state normalization comes from global observation training statistics;
unobserved velocity uses mean zero and standard deviation one, and unobserved deep
T/S repeats the deepest observed scale.

Each screen arm gets 4,000 main updates plus 1,000 observation-only cooldown
updates. Main learning rate is 1e-4; cooldown decreases it to 1e-6, retaining Adam
moments and resetting EMA to the raw model once. Selection uses the best EMA
checkpoint across both main training and cooldown, scored on a 365-day forecast from
2013-11-01. The score equally averages 29 observed channels' area-weighted RMSE
(SST, SSH, temperature depths 1–13 and salinity depths 0–13),
normalized by their observation training standard deviations. This is a single
validation origin, not an untouched test of forecast skill.

Only the selected configuration continues from its **uncooled** 4k checkpoint.
Its main trajectory reaches 16k updates, with independent 1k observation-only
cooldowns at 4k, 8k, 12k and 16k. Total winning-configuration work is 20k updates;
15k remain after its 5k screen. Forks preserve optimizer, EMA, RNG and replay state,
and permit only duration and snapshot-schedule changes. The intermediate cooldowns
cannot alter the continuing main trajectory. The
[campaign generator](../../src/samudra/observations/campaign.py) writes all five
screen configs and the winner continuation/branch configs without submitting jobs.

Qualification uses the production update loop with four main and two cooldown
updates at effective batch eight. It checks exact next-update resume through the
cooldown boundary, uncooled checkpoint forks, fresh/replay exposure, a complete
73-step/29-channel annual validation and peak GPU memory. Arm D also checks global
normalization moments, exact agreement between cached monthly samples and the
same months rebuilt from daily shards, and twelve-month replay chains for both
sources. These checks establish engineering behavior; six updates do not
establish scientific performance.

Timing receipts separate source-specific optimizer-update time, data loading,
annual validation (including an improved-best checkpoint), separate checkpoint
writing and restore. The runtime calculation counts that best-checkpoint save once. Full replay-capacity checkpoint
timing includes populated Adam moments. The
[runtime estimator](../../src/samudra/observations/runtime_estimate.py) accounts for
validation every 250 updates and saves every 25 updates. It reports observed timing
ranges and a separately labeled 30% planning allowance. Queue delay and unknown
preemption frequency are excluded. The five 4k-plus-1k screens are estimated at
479–524 aggregate GPU-hours with that allowance using the current loader;
see the [per-arm estimates](qualification.md).

The replacement qualification code is frozen at `024177bc7b612042275381f3470d37e795e5d396`.
The matching PhysicsNeMo container is built from
`4f86be45a09cb9eba1b51b6273e0189941b02495`; CPU pull job `21784191` completed in
9m24s. The source-overlay builder requires byte-identical dependency lockfiles.
[Preparation](prepare_torch.sbatch), [GPU launch](qualify_torch.sbatch) and
[data paths](torch-data-paths.json) preserve the Torch setup. Checkpoints and full
receipts live under `/scratch/jr7309/deterministic-obs-v2/`.
The replacement run wrote to `qualification-v2/`; its completion callback,
CPU job `21786737`, also completed. All qualification callbacks are retired.

Daily data were copied from the existing Beta observation preparation and historical
shared-ERA5 preparation. Download/read-back checks found zero differences across
24,546 observation files and 6,611 historical ERA5 files; the
[transfer receipts](receipts/) record those checks. Forty ERA5 year manifests
(1975–2014) agree on field order and grid hash. The completed cache contains all 439 eligible native OM4 windows and 2,812 five-day
intervals, with input hashes and exact read-back verification in its proof. Global
normalization and all ten screen/qualification configs are prepared; see the
[preparation receipt](receipts/PREPARED.json). No upstream
raw data were reprocessed or transferred in this task.

Before a full campaign, use the qualification measurements to settle checkpoint
cadence and GPU capacity. The current harness uses one GPU per arm; parallel arms
can use several GPUs, but distributed training within one arm is not implemented.
Exact numerical resume is checked on the assigned GPU type. Changing GPU types
across preemptions can change floating-point results even though the forecast
model has no stochastic latent variables. The number of confirmation seeds and
whether to add validation origins also remain campaign choices; the implemented
screen uses seed 1729 and the original annual selection protocol.
