<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longer training and observation replay

Question: does additional training or exposure to the model's own longer latent
rollouts improve the monthly forecasts and the severe annual drift documented
in the [original pilots](monthly-latent-pilots-2026-10-16.md) and
[365-day evaluation](monthly-latent-annual-2026-10-18.md)?

Status: implementation and qualification preparation; no new results yet.

## Four runs

| Literal run name | Initialization | Main training | Cooldown | Observation replay |
|---|---|---|---|---|
| A-extend | Original A raw 2k checkpoint, including optimizer/EMA/RNG | Another 2k updates, reaching 4k main; 50% OM4 / 50% obs | 1k obs only; ends at total 5k | None |
| E-extend | Original obs-only E raw 2k checkpoint, including optimizer/EMA/RNG | Another 2k updates, reaching 4k main; 100% obs | 1k obs only; ends at total 5k | None |
| A-replay | Fresh random initialization, original seed 1729 | 2k updates; 50% OM4 / 50% obs | 1k obs only; ends at total 3k | Half of observation examples |
| E-replay | Fresh random initialization, original seed 1729 | 2k updates; 100% obs | 1k obs only; ends at total 3k | Half of observation examples |

Here “updates” means optimizer updates, with eight accumulated examples each,
matching the existing pilots. The new computation is 3k updates / 24k examples
per run. The replay pilots compare with original A/E at the same 2k+1k budget;
the extensions compare with A/E before their original cooldown. None resumes
cooled 3k weights, and qualification weights never initialize production.

All runs retain the same 48,702,016-parameter latent processor and physical
diffusion decoder, 1° fields (180×360), 19 five-day history bins, OM4
normalization, AdamW, clipping at 1, and EMA half-life 100 updates. The main LR
is fixed at 1e-4. Cooldown retains Adam moments, resets EMA to raw at its start,
and linearly decays LR to 1e-6 over 1k updates. There is **no latent magnitude
penalty, soft bound, or new architecture**.

The observation objective remains 0.8 monthly interior temperature/salinity
denoising + 0.2 native five-day SST/SSH denoising + 0.1 initial surface
reconstruction. There is no direct OHC-integral training loss. A retains the
original OM4 full-field loss and native forcing interface. E has no OM4
optimization examples but retains shared OM4 normalization.

## Replay definition

Each observation update accumulates four fresh forecasts and four replay
continuations, except when no eligible replay state exists during startup.
This also applies during the observation-only cooldown. Model updates in A
are unchanged: its main mix is 50% OM4, 25% fresh obs and 25% replay obs.

Fresh forecasts initialize from the original observed 95-day surface history.
They save a detached latent state at the native five-day boundary needed to
cover the next calendar month. Replay continues that state with subsequent
ERA5 forcing and geographical/calendar context, then scores the later month's
surface and interior observations. No later observed ocean field resets,
corrects, or enters the continued forecast state. Decoder lead conditioning
uses the true elapsed lead from the original initialization, rather than
resetting it each month.

Backpropagation covers one later month per replay example and stops at the
cached state: this is truncated backpropagation, not differentiation through
an entire stored year. Its reconstruction term is a separate fresh initializer
branch at the target month, preserving the original reconstruction weight and
initializer learning without feeding that branch into the continued forecast.

The pool holds up to eight entries for each of the next eleven calendar-month
leads. We sample uniformly among occupied lead buckets, then uniformly within
the selected bucket; this encourages long leads rather than a geometric
distribution dominated by the second month. The total trajectory is capped at
73 five-day steps. Months requiring more steps, and targets outside the
original May 1993–July 2013 training split, are ineligible. Cached states expire
250 optimizer updates after their **original initialization**, including OM4
updates; extending a chain does not refresh that age. States, timestamps, age,
pool contents and the separate replay RNG are serialized for exact restarts.

Training samples have five-day bins anchored to each month's origin; adjacent
monthly files cannot simply be concatenated. A read-only interval cache derives
all required phases from the already prepared daily SST/ADT/ERA5 archive using
the original strict five-day mean. It verifies exact equality with all 243
original training months' future surface and forcing payloads. Its latest raw
date matches the last original training sample's final bin; it includes no
validation/test targets. It retains source-file hashes and full cache read-back
hashes. Expected cache size is about 19 GB, with no new upstream transfer.

## Comparisons and qualification

Selection and the main learning curves retain the existing nine-origin monthly
RMSE blend (SST, geostrophic velocity, EKE, upper/deep OHC), the same fixed
climatology denominators, eight diffusion members and 32 sampling steps.
Validation runs every 250 updates. Report the surface-only blend and each OHC
component alongside the full blend. Save raw and EMA endpoint validation so
averaging cannot hide a difference.

At the fixed cooled endpoints, repeat the same three continuous annual test
rollouts (January 2015, 2018 and 2021), with 73 five-day steps and no latent reset.
Compare 30/90/180/365-day metrics, monthly OHC and latent RMS against the original
A/E endpoints. Annual tests are reporting only, not a checkpoint-selection
criterion. Report actual replay lead exposure and age distributions: a maximum
allowed horizon does not prove that training reached it. One seed and matched
update counts do not establish significance or exact FLOP matching.

Before production: verify pinned source/data hashes, finite updates in all four
arms, exact serialization/restart of model/EMA/optimizer/RNG/replay state,
diffusion evaluation, and a finite replay example ending at day 365. Run the
four production processes on one four-GPU Beta node, with online W&B and durable
completion follow-up. Prepared data use Grace CPU compute. Keep prior source
checkpoints read-only and retain failed attempts and GPU accounting.

Implementation: [replay state and loss](../../../src/samudra/experiments/monthly_replay.py),
[runner](../../../scripts/run_monthly_replay.py),
[interval preparation](../../../scripts/prepare_monthly_replay.py).
