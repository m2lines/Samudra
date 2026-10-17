<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small latent/diffusion pilots

Status: **All five pilots completed** in Beta job **214108** (10h19m46s).
Each reached 2,000 main updates and 1,000 observation-only cooldown updates,
with online W&B, twelve validation checkpoints, and saved main/cooled endpoints.
Producer `1c9817d44`; full qualification **214096** passed. Allocated production
GPU time was 41.3178 hours; including qualification attempts, **41.5228 GPU-hours**.

## Pilot results

These are **validation results**, not held-out test or annual rollout results.
The shared blend is the mean of five raw-RMSE ratios to fixed seasonal climatology:
SST, geostrophic velocity, EKE, and OHC at 0–700 m and 700–2,000 m. No spectra enter
selection. A score of one equals the climatology blend. Nine monthly origins,
eight ensemble members, and 32 sampler steps are fixed across arms.

| Arm and meaning | Main endpoint (2k) | Cooled endpoint (3k) | Selected step / score |
|---|---:|---:|---:|
| A: OM4 + obs / surface | 1.3037 | 1.1182 | 3000 / 1.1182 |
| B: OM4 + obs / multitask | 1.3677 | 1.1257 | 3000 / 1.1257 |
| C: OM4 + CM4 + obs / surface | 1.3639 | 1.1330 | 3000 / 1.1330 |
| D: OM4 + CM4 + obs / multitask | 1.4121 | 1.1393 | 3000 / 1.1393 |
| E: Obs only / surface | 1.1821 | 1.1443 | 2750 / 1.1421 |

![Validation curves](assets/monthly-latent-pilots-2026-10-16/validation-curves.png)

The simple OM4+obs arm A finishes best, 2.27% below obs-only E at the matched
3k endpoint. Multitask initialization and replacing half of OM4 with CM4 show
no improvement in this pilot. Before cooldown, obs-only is best. The mix arms
receive 1,000 observation updates during main training versus 2,000 for E;
after cooldown those counts are 2,000 versus 3,000. Thus this is a matched total
update budget, not a matched observation exposure comparison. Cooldown changes
both the learning rate and the mix for A–D, so its improvement cannot be attributed
to learning rate alone. One seed and small final differences do not establish a
reliable ranking. All final blends remain above seasonal climatology overall;
component performance should guide objective and training-setup follow-ups.
Obs-only E has the best surface SST, velocity and EKE errors, while A has the
best OHC errors in both depth ranges. The composite winner therefore reflects
better subsurface heat content, not uniformly better forecasts.

Cooled endpoint components (SST/velocity/EKE averaged over leads 5, 15 and 30 days):

| Arm | SST (°C) | Velocity (m/s) | EKE (m²/s²) | OHC 0–700 m (GJ/m²) | OHC 700–2,000 m (GJ/m²) |
|---|---:|---:|---:|---:|---:|
| A | 0.7879 | 0.13712 | 0.02144 | 0.7585 | 0.5908 |
| B | 0.7845 | 0.13708 | 0.02151 | 0.7779 | 0.5950 |
| C | 0.7418 | 0.13736 | 0.02145 | 0.7734 | 0.6298 |
| D | 0.7565 | 0.13714 | 0.02149 | 0.8099 | 0.6155 |
| E | 0.6998 | 0.13534 | 0.02121 | 0.8142 | 0.6570 |

[Machine-readable curves and component metrics](assets/monthly-latent-pilots-2026-10-16/results.json).
[Checkpoint byte hashes](assets/monthly-latent-pilots-2026-10-16/checkpoint-sha256.txt)
are relative to the run root. These verify retained files; this report does not
claim a separate post-training checkpoint reload evaluation.

- [A: OM4 + obs / surface](https://wandb.ai/ocean_emulators/observational-transfer/runs/5488bc38064c)
- [B: OM4 + obs / multitask](https://wandb.ai/ocean_emulators/observational-transfer/runs/66188f65768e)
- [C: OM4 + CM4 + obs / surface](https://wandb.ai/ocean_emulators/observational-transfer/runs/ba6af0dc0924)
- [D: OM4 + CM4 + obs / multitask](https://wandb.ai/ocean_emulators/observational-transfer/runs/c8ed764155f9)
- [E: Obs only / surface](https://wandb.ai/ocean_emulators/observational-transfer/runs/b0723b64e7ab)

## Training diagnostics (initial log audit)

The logged gradient norm is the **pre-clipping** global norm of the accumulated
batch-eight gradient. Every update clips it to at most 1 before AdamW. During
updates 1,001–2,000, all five arms clipped every update. For A:

| Source | Mean loss ± SD | Median gradient norm | 90th percentile | Clipped updates |
|---|---:|---:|---:|---:|
| Observation | 0.2630 ± 0.0708 | 4.839 | 7.015 | 100% |
| OM4 | 0.6412 ± 0.0655 | 3.113 | 4.571 | 100% |

A's source-dependent mean accounts for 88.5% of loss variance in that interval
(87.7–89.5% across A–D). This explains most of the alternating trace; it does
not establish harmful gradient conflict. The denoising objective averages
area-weighted error within each supported channel and then averages channels,
so extra output channels do not mechanically multiply the loss. Observation
and model updates nevertheless differ in target distributions, support, native
versus monthly weights, and diffusion noise. Compare each source's trajectory
and physical validation, not the pooled scalar alone.

Heavy clipping is an optimization question, not evidence by itself of exploding
updates: norms are finite and validation improves, and the logged norm is not
Adam's actual parameter-step magnitude. Observation gradients are larger even
though their scalar loss is smaller. In A's final 500 cooldown updates, clipping
still occurs 91.8% of the time (median norm 2.452). Clipping therefore substantially
changes the gradient scale supplied to Adam; it should be measured before
changing the threshold or interpreting nominal task weights.

Other questions suggested by this audit:

- The composite favors subsurface OHC while surface errors favor obs-only; inspect
  gradient alignment and per-component validation before changing data ratios.
- OM4 and observations use different forcing paths: the learned ERA5 adapter is
  observation-only, while OM4 supplies its native three forcings. Transfer could
  depend on this interface as well as ocean fields.
- The diffusion decoder is stochastic but latent evolution is deterministic;
  decoder spread alone does not establish calibrated trajectory uncertainty.
- Direct monthly and native-window readouts need not agree. Native day-5/day-30
  OHC is especially diagnostic for E, which has no native interior supervision.

Read-only diagnostic job **214194** is queued on production Beta. Script commit
`c7160b76e` imports the unchanged training producer `1c9817d44`; no optimizer
steps or checkpoint writes are performed. It measures A's raw 2k/3k gradients
using four batch-eight replicates per source, with within-source cosine controls,
and exports all five cooled EMA models at two validation origins and leads 5/30.
The initializer probe covers the U-Net; its input projection is not included.
OM4 has no adapter gradients, so adapter cross-source cosine is not informative.
Maps use five-day-bin means; monthly OHC observations are not matching daily
truth. Completion and map plotting remain pending.

## Question and arms

Does model-data supervision improve observation forecasts with a shared latent
processor and physical diffusion decoder? Does varying the initializer's available
inputs improve that transfer? All fields are global 1° (180 × 360).

| Arm | Main optimizer-update mix | Initializer inputs |
|---|---|---|
| A | 50% observations, 50% OM4 | Surface only |
| B | 50% observations, 50% OM4 | Multitask |
| C | 50% observations, 25% OM4, 25% CM4 | Surface only |
| D | 50% observations, 25% OM4, 25% CM4 | Multitask |
| E | 100% observations | Surface only |

One seed (1729), 2,000 main updates, effective batch eight. Constant main LR
1e-4. Then 1,000 observation-only updates with LR linearly declining to 1e-6,
retaining optimizer state and resetting EMA to raw weights at the fork; EMA
half-life 100 updates. This is a deliberately small comparison, not a reliable
ranking of small differences. No later search wave is included in this launch.

Multitask draws: 50% surface-only, 25% missing-variable family, 25% spatial gap.
Model data can supply the available interior fields. Observation examples supply
only observed surface history: the variable task removes SST or SSH, and gaps
remove a quarter-height by quarter-width rectangle. They never acquire monthly
interior targets as initializer inputs. Task changes do not remove forecast
losses or shorten forecast horizons.

## Architecture and losses

Separate initializer and processor. One initialization from nineteen five-day
historical frames, with explicit per-field availability. Its output is one latent
state containing two memory slots, each 128 channels on a 45 × 90 grid. The
processor autoregresses that state using forcings and geographic/seasonal context;
decoded physical samples never feed back. A shared denoiser decodes 77 physical
channels, conditioned on latent state, target support, window duration and lead.
All arms use this same **48,702,016-parameter** architecture. Initial real-observation
qualification measured 0.736 seconds per microbatch and 4.32 GB peak allocated
CUDA memory. This is a short timing including three training examples; it excludes
model-data I/O and validation and is not an end-to-end production estimate.

The proposed monthly speedup is **direct monthly-target denoising**, conditioned
on a calendar-weighted average of evolved latent states. Native five-day targets
use individual latent states. Monthly outputs are not guaranteed to equal the
average of separately sampled five-day outputs; report their consistency as a
diagnostic. Training uses single denoiser calls, not differentiation through a
complete diffusion sampler.

Observation loss: 0.8 monthly interior denoising + 0.2 five-day surface denoising
+ 0.1 initial-surface reconstruction. Interior supervision uses the existing
fourteen temperature and fourteen salinity layers, excluding supplied SST from
interior supervision. Surface loss samples a uniformly chosen forecast lead;
its expectation is the mean over all forecast leads. Missing targets are excluded
from both corrupted inputs and scoring. This is masked marginal denoising, not a
complete-field likelihood or invented interior labels. Reconstruction uses the
last historical surface, not the future month's interior average. Model-data
loss is 0.5 full-field monthly denoising + 0.5 full-field native forecast denoising
+ 0.1 full-field initial reconstruction. Model-data objectives still need full
qualification before production.

## Data and evaluation contracts

CM4 source: `s3://m2lines-pubs/Samudra/v2026-09/cm4_onedeg/CM4.zarr`.
Destination: `/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg/CM4.zarr`.
Source inventory: 131,466 objects, 140,827,879,002 bytes. Grace transfer job
138228 completed in 21m08s, with full read-back verification, zero differences,
and byte-identical source-before/source-after/destination metadata. Its
`COMPLETE.json` records metadata SHA256
`5bfbf9fb994bdbe7247f0d5b88546c8c0a025ada3a632b8d59f556e4272d2e43`.
Callback job 138229 also completed; transfer monitoring is retired.

Live static audit: CM4, the existing OM4 store, and the observation grid have
exactly equal latitude/longitude coordinates; CM4 has the expected 19 layer
depths and thicknesses. No horizontal regridding is required. This verifies
geometry, not completeness of dynamic fields.

CM4 uses its own tauuo/tauvo/hfds and actual noleap interval bounds. Retain the
fixed OM4 normalization for model fields/forcings, including CM4; do not relabel
CM4 dates as OM4 dates. Observation physical outputs use the same fixed reference
scales; ERA5 has its existing training-only normalization and learned forcing
adapter. E has no model-data training examples but shares reference normalization.

OM4 uses the existing recent period (1975 onward, through the training cutoff),
not early OM4 in addition to CM4. The draft model-data adapter reserves the last
20 CM4 model years and uses its first 180 years. The OM4 timestamp axis has sixteen
six-day jumps at leap-year boundaries. It must not silently become contiguous
five-day calendar coverage: the adapter records and excludes monthly windows
whose history or targets cross these gaps. Full data qualification will report
the resulting sample inventory before production.

Use existing observation train/validation/test splits and raw RMSE-blend reporting,
not spectra for checkpoint selection. Validation runs every 250 optimizer updates
over all nine existing validation origins: eight members, 32 Heun steps, fixed
sampler seeds, EMA weights, and the existing five-component normalized RMSE score
with a fixed seasonal-climatology denominator. Report both main and cooled
endpoints and component errors. Full qualification measures this sampling cost.

Surface metrics use the ensemble-mean five-day surface predictions. OHC uses
directly sampled monthly interior temperature/salinity, with its top temperature
layer supplied by the calendar average of native SST predictions (SST is excluded
from IAP interior supervision). This readout is explicit; it is not a claim that
the two decoder windows are physically identical. A frozen-initial-latent readout
is a separate diagnostic, not a physically constant persistence forecast.

## Launch gates

One Beta production node, with up to four independent single-GPU arms at a time
and the fifth subsequently; qualification may change that packing if memory or
throughput requires it. Waiting for the production queue is acceptable.

Initial qualification checks real-observation loading, all three initializer
input tasks, finite nonzero gradients into initializer/processor/decoder/forcing
adapter, memory/throughput, and exact optimizer/checkpoint replay. This is not
sufficient to qualify CM4 or the complete production protocol. Production also
requires verified CM4 transfer, channel/grid/calendar/forcing audits, model-data
loss and gradient checks, a frozen evaluation contract, and online W&B bring-up.
Diagnostic weights are never promoted to production.

Implementation: [model](../../../src/samudra/experiments/monthly_latent.py),
[observation objective](../../../src/samudra/experiments/monthly_pilot_data.py),
[model-data adapter](../../../src/samudra/experiments/monthly_model_data.py), and
[initial qualification](../../../scripts/qualify_monthly_latent.py).
The persistent-latent processor and diffusion core were imported from the
diffusion worktree at `2b98db3deb4370c3b641cf0d1055a8b394917589`; unused auxiliary
loss helpers were omitted. Fixed 4× pooling replaces adaptive pooling where
the grids have an exact integer ratio, enabling deterministic CUDA replay.
Eleven CPU tests pass, including pooling value/gradient equivalence, masking,
native-calendar weighting, exact source allocation, and rejection of missing
wet CM4 values. Repository pre-commit checks pass for the pinned producer.

Preserved qualification attempts: 214083 failed before Python because Slurm
rounded CPU allocation inconsistently; 214086 was held and canceled without
running; 214087 reached backward but deterministic CUDA rejected adaptive pooling.
Including successful 214088 (94 seconds), completed qualification GPU time is
200 seconds (0.05556 GPU-hours); 214094 is excluded until its accounting is final.
214088 verified finite nonzero gradients in all four trainable components for
all three tasks, and bitwise-identical checkpoint replay. Its diagnostic weights
will not be used. The production runner and validation path are implemented;
full model-data/evaluation qualification subsequently passed in 214096.

214094 subsequently verified the retained OM4 normalization and all three fitting
tasks on both model sources. It found 409 eligible recent-OM4 months and 2,156
CM4 months. OM4 loading/fitting took 7.5–17.3 seconds per example versus
1.9–3.1 seconds for CM4, so production will use a packed OM4 copy whose every
frame is read back and byte-hash matched to the existing store. No values, masks,
normalization, or sampling are changed by packing. Twelve CPU tests now pass,
including exact packed/native equality and corruption rejection. Completed GPU
time including failed 214094 is 366 seconds (0.10167 GPU-hours).

Grace preparation jobs 138249 and 138251 failed before Python because the shared
Enroot image cache denied access; the second attempt's environment override was
ignored by Pyxis. Both attempts and logs are retained (zero GPU-hours). Recovery
138253 imported the same pinned image directly into a private project cache before
starting Pyxis and completed in 18m06s; CPU callback 138254 succeeded. The packed
store contains 2,702 × 77 × 180 × 360 float32 values (53,927,596,928 bytes including
the NPY header). Every frame passed source-byte-hash read-back; the complete digest
inventory, file header/size, and unchanged source/grid metadata were also checked.
All three CPU attempts together used 20m36s elapsed allocation time and zero
GPU-hours. Replacement full Beta qualification 214096 uses `FULL_QUALIFICATION=1`, `PRODUCER=1c9817d44`, and
`PACKED_OM4=RUN_ROOT/packed-om4`. It passed, and production 214108 completed all five arms.

Run root: `/projects/ny/lz1955/multiscale/jrusak/runs/monthly-latent-pilots-20261016`.
Proofs live in `initial-qualification-JOBID/`; producer-matched `QUALIFIED.json`
and unchanged data/normalization hashes are required before production updates.
