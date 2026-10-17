<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small latent/diffusion pilots

Status: CM4 transfer and packed OM4 read-back **verified complete**. Full
qualification **214096 passed** for producer `1c9817d44`: all observation and
OM4/CM4 tasks, exact checkpoint replay, and nine-origin/eight-member/32-step
validation. Five-arm production **214108** is submitted on one four-GPU Beta
node (24-hour allocation); it is pending, not yet training. A–D run concurrently,
then E uses A's GPU. Completion callback and quiet monitoring are installed.
This plan supersedes the broad screen's early-OM4 and processor-completion proposals.

Qualification measured 0.707 seconds per observation microbatch (excluding I/O),
1.32–1.51 seconds for warmed packed OM4 examples and 2.01–2.05 seconds for warmed
CM4 examples including I/O and fitting. The first reads were slower. Full
nine-origin validation took about 250 seconds. These short timings suggest the
wave fits the allocation; actual concurrent training throughput remains to be
verified. Qualification attempts total **0.205 GPU-hours**, including failures.

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
full model-data/evaluation qualification remains the launch gate.

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
`PACKED_OM4=RUN_ROOT/packed-om4`. Then submit the five-arm production request if
its measured runtime fits the allocation. The latest advisory full-node start
estimate was about 13 hours away; waiting in the production queue is intentional.

Run root: `/projects/ny/lz1955/multiscale/jrusak/runs/monthly-latent-pilots-20261016`.
Proofs live in `initial-qualification-JOBID/`; producer-matched `QUALIFIED.json`
and unchanged data/normalization hashes are required before production updates.
