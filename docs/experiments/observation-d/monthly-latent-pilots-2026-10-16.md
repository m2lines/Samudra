<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small latent/diffusion pilots

Status: authorized five-arm pilot wave; CM4 transfer running. Initial observation
qualification **214088** is running on Beta with producer `307b1d2ef`.
No production pilot has started. This plan
supersedes the broad screen's early-OM4 and processor-completion proposals.

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
All arms use this same architecture. Parameter counts and measured throughput
will be reported from qualification.

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
native and monthly objectives still need full qualification before production.

## Data and evaluation contracts

CM4 source: `s3://m2lines-pubs/Samudra/v2026-09/cm4_onedeg/CM4.zarr`.
Destination: `/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg/CM4.zarr`.
Source inventory: 131,466 objects, 140,827,879,002 bytes. Grace transfer job
138228 performs full read-back verification and verifies source metadata did not
change before publishing `COMPLETE.json`. Callback job 138229 follows termination.

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
not spectra for checkpoint selection. Hold evaluation starts, sampler seeds and
ensemble size fixed across arms. Report both main and cooled endpoints, component
errors, initialized persistence, and the gap between native-averaged and directly
decoded monthly predictions. Exact validation cadence and sampling cost will be
frozen after qualification, before any production updates.

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
Their allocated GPU time is 106 seconds (0.02944 GPU-hours). The replacement
214088 is not included until its accounting is final. Completion hook and
two-hour fallback reminder are installed; CM4 transfer has a separate monitor.
Production runner, final evaluation contract, and full model-data qualification
remain to finish after these initial checks. No diagnostic weights will be used.
