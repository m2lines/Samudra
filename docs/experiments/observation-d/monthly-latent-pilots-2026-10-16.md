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

Read-only diagnostic job **214194** completed on production Beta (11m56s). Script commit
`c7160b76e` imports the unchanged training producer `1c9817d44`; no optimizer
steps or checkpoint writes are performed. It measures A's raw 2k/3k gradients
using four batch-eight replicates per source, with within-source cosine controls,
and exports all five cooled EMA models at two validation origins and leads 5/30.
The initializer probe covers the U-Net; its input projection is not included.
OM4 has no adapter gradients, so adapter cross-source cosine is not informative.
Maps use five-day-bin means; monthly OHC observations are not matching daily
truth. All twenty map exports and both gradient probes passed finite-value and archive checks.

Additional read-only job **214195** (diagnostic commit `f022e38b2`) follows the
map job and computes counterfactual AdamW update norms relative to module weight
norms under clipping thresholds 1, 5 and no clipping, using the same gradients
and saved optimizer moments. This is a mathematical single-step calculation,
not a training intervention or evidence of longer-term benefit. The calculation
was checked against an actual float64 AdamW step at three gradient scales to
1e-14 tolerance. Each checkpoint retains its saved LR, so compare clipping
choices within a checkpoint before comparing 2k and 3k absolute updates.
It also records mean/RMS/max of learned ERA5 adapter outputs and normalized
OM4 forcings; these are distribution diagnostics, not paired physical errors.
OM4 batches do not use the adapter, whose absent gradients correctly skip both
Adam moments and weight decay. Original checkpoint files remain read-only.

The eight ERA5 inputs, in order, are 10 m zonal wind, 10 m meridional wind,
2 m temperature, 2 m dewpoint temperature, surface pressure, downward surface
solar radiation, downward surface thermal radiation, and total precipitation.
Their preparation names are defined in `observation_prepare.py::FORCING`.

## Gradient alignment and example maps

Four independent batch-eight probes per source use A's raw weights at each
endpoint, identical batch/noise draws across the two checkpoints, and the actual
training objectives. These are training examples, not validation performance.
Cosines are averaged over all cross-source batch pairs; within-source means
exclude the diagonal. Global clipping preserves each gradient direction.

| Module | Cross-source cosine at 2k | Cross-source at 3k | Obs/obs at 3k | OM4/OM4 at 3k |
|---|---:|---:|---:|---:|
| Initializer U-Net | 0.593 | 0.062 | 0.273 | 0.983 |
| Processor | 0.599 | -0.016 | 0.112 | 0.963 |
| Decoder | 0.695 | -0.040 | 0.392 | 0.997 |

The tasks align positively before cooldown; afterwards they are approximately
orthogonal, with weak negative processor/decoder cosines. Observation gradients
also become substantially less consistent across batches. This does not support
a claim of strong persistent task conflict throughout training.

| Checkpoint/source | Probe loss | Initializer norm | Processor norm | Decoder norm | Adapter norm |
|---|---:|---:|---:|---:|---:|
| 2k / obs | 0.1806 | 0.582 | 0.299 | 6.480 | 0.0136 |
| 2k / OM4 | 0.5487 | 0.182 | 0.105 | 3.029 | 0 |
| 3k / obs | 0.0909 | 0.078 | 0.054 | 0.481 | 0.0050 |
| 3k / OM4 | 0.8746 | 1.051 | 0.819 | 24.506 | 0 |

These are mean pre-clipping module norms, not parameter updates. The decoder
dominates the probed gradients; the ERA5 adapter does not. The same OM4 probes
get worse after observation-only cooldown while observation probes improve:
evidence of lost model-data fit during specialization. That makes decoder/loss
balance and retaining some model supervision worth investigating. It does not
isolate forcing as the cause, nor prove that retaining OM4 improves obs forecasts.
The measured norm breakdown omits the initial frame projection; the additional
Adam diagnostic includes it. Frozen-checkpoint probe gradients differ from
online training gradients and should not be substituted for the training trace.

Latent RMS increases over 30 days by 5.6–6.0% for A, 4.1–4.2% for B, 4.9–5.1% for
C, 2.1–2.3% for D and 13.0–13.8% for E in these two examples. This is not explosive
short-horizon growth; longer-rollout stability remains untested here.
[Full gradient records](assets/monthly-latent-pilots-2026-10-16/gradient-diagnostics.json)
and [latent RMS traces](assets/monthly-latent-pilots-2026-10-16/latent-rms.json).

Maps use the cooled 3k EMA endpoints, eight-member means, and matched origins
November 2013 and April 2014. Each grid cell occupies exactly 2×2 pixels, with
shared per-variable scales across arms and leads within each panel. Color limits
use pooled 1st/99th percentiles; gray marks missing support. Longitude runs 0–360°,
north at top. The separately sampled map ensembles are not the identical random
stream used for the aggregate validation scores. OHC panels are native five-day
readouts, not the directly supervised monthly product; they have no matching
five-day observation target. Model fields are smooth at this resolution; these
absolute maps alone cannot establish dynamical skill or ensemble calibration.

![November surface forecasts](assets/monthly-latent-pilots-2026-10-16/surface-maps-2013-11.png)

![November native OHC](assets/monthly-latent-pilots-2026-10-16/ohc-maps-2013-11.png)

![April surface forecasts](assets/monthly-latent-pilots-2026-10-16/surface-maps-2014-04.png)

![April native OHC](assets/monthly-latent-pilots-2026-10-16/ohc-maps-2014-04.png)

This diagnostic consumed 0.7956 allocated GPU-hours (one GPU active on a required
four-GPU Beta node), bringing completed allocation time to 42.3183 GPU-hours.
Clipping/forcing diagnostic 214195 also completed (3m49s, 0.2544 allocated GPU-hours); total completed campaign allocation is now 42.5728 GPU-hours.

## Clipping and forcing follow-up

Job **214195** completed without updating weights. Results use four batch-eight
replicates per source, raw A checkpoints and their saved AdamW moments. At 2k,
median predicted next-step size as a percentage of each module's weight norm:

| Source/module | Clip 1 | Clip 5 | No clipping |
|---|---:|---:|---:|
| Obs initializer | 0.0396% | 0.1396% | 0.1632% |
| Obs processor | 0.0383% | 0.1351% | 0.1590% |
| Obs decoder | 0.0506% | 0.1662% | 0.2036% |
| OM4 initializer | 0.0459% | 0.0912% | 0.0912% |
| OM4 processor | 0.0387% | 0.0645% | 0.0645% |
| OM4 decoder | 0.0596% | 0.1299% | 0.1299% |

Clipping materially changes the update; Adam does not cancel its effect when
existing moments are held fixed. These small relative update norms do not suggest
an immediate giant-step problem. They also do not prove that increasing the
threshold improves training. Global clipping is driven mainly by decoder
gradients and scales down initializer/processor gradients at the same time.
This supports testing clipping policy before changing physical normalization.

At the cooled endpoint, these four obs probes have total norms 0.41–0.64, so
threshold choice has no effect on their next step. OM4 probes have norms 20–28;
clip 1 to 5 raises the median decoder update about 2.6-fold. This is an endpoint
probe, not a contradiction of frequent clipping during the preceding online
cooldown trajectory. The endpoint LR is 1e-6 rather than 1e-4 at 2k; do not
attribute their smaller absolute updates solely to clipping.

Forcing summaries at 2k (means and RMS averaged across sampled examples and all
spatial cells, including land):

| Representation | Channel means | Channel RMS |
|---|---|---|
| Learned ERA5 adapter | -0.147, 0.598, 0.979 | 0.537, 1.153, 1.880 |
| Normalized OM4 stress-x/stress-y/heat-flux | 0.009, 0.007, -0.011 | 0.845, 0.826, 0.856 |

The learned representation differs from the physical forcing representation;
that alone is not a defect because its channels have no physical calibration.
The comparison also includes zero-filled land for OM4, while the adapter can emit
nonzero values on land, so these are not matched ocean-only statistics. Adapter
gradients remain small; there is no evidence that they directly dominate clipping.
A separate lead is OM4 heat-flux tails: the largest sampled normalized absolute
value is approximately 180, despite global RMS below one. Locate those values,
check raw units and saved normalization, and examine spatial support before
changing scales or adding an OM4 adapter. A global mean/RMS cannot resolve that.

Recommended next experiment, subject to approval: matched short continuations
from 2k with threshold 1 versus 5, same LR/data draws, monitoring module update
sizes and both source losses. If decoder domination remains limiting, compare
module-wise clipping against that control rather than immediately changing data
normalization. Forcing interpretation needs ocean-only, geographically localized
checks; these diagnostics have not established forcing as the cause of lost OM4 fit.

Numerical audit: the original float32 cosine calculation produced self-cosines
up to 1.007 because its vector normalization and dot-product reductions differed.
Published matrices now normalize the saved Gram entries by their own diagonal,
which algebraically cancels the original normalization factors; zero-gradient
adapter rows remain zero and are not evidence of alignment. Source now uses a
float64 Gram calculation. This correction changes the 2k initializer cross-source
mean from 0.597 to 0.593, without changing the interpretation. Raw cluster output
is preserved. [Full clipping and forcing records](assets/monthly-latent-pilots-2026-10-16/clipping-diagnostics.json).

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

## Approved clipping and OM4-adapter continuations

All three branches fork A's raw and EMA **2,000-update main endpoint**, preserving
its Adam moments and RNG/data-draw state. Each trains 1,000 more updates with
constant LR 1e-4, batch eight, seed 1729, and exactly 50% obs / 50% OM4. This is
continued mixed training, not another obs-only cooldown. Retain EMA half-life
100 without reset; evaluate the same nine-origin/eight-member/32-step raw-RMSE
blend every 250 updates. Compare both curves and the common 3k endpoint.

| Name | Global clipping threshold | OM4 forcing interface |
|---|---:|---|
| clip1 | 1 | Existing normalized physical forcings |
| clip5 | 5 | Existing normalized physical forcings |
| adapter | 1 | Identity-started learned affine 3→3 transform |

The adapter has 12 parameters (3×3 weights plus three biases), operates pointwise
on normalized OM4 forcings for initialization and evolution, and restores zero
on model land. ERA5 retains its existing learned adapter. Only the new parameters
receive fresh Adam state; the backbone and its optimizer history are preserved.
The adapter begins exactly at the old interface, avoiding an initial input shift.
This is the minimal learned-affine interface test, not a capacity-matched nonlinear
adapter test. No CM4 or extra seeds are added.

Beta job **214212 completed** all three qualifications and continuations in
2h57m01s on one four-GPU node. Monitoring is closed. Producer `6a76e61ee`; thirteen focused CPU tests and all repository pre-commit
checks pass. Source checkpoint hash is pinned in the launcher. All three branches
must pass real obs/OM4 batch-eight updates, finite gradients, checkpoint reload,
and (adapter branch) identity-output and nonzero-learning checks before any
production branch starts. Qualification weights are discarded: production reloads
the original immutable 2k checkpoint. Source-specific W&B losses/clipping are
logged alongside the mixed trace. New run root:
`/projects/ny/lz1955/multiscale/jrusak/runs/monthly-clip-adapter-20261017`.

Runtime reference: original A used 6h34m including validation, comprising 5h19m
for its first 2k mixed updates and 1h15m for its obs-only cooldown. Approximately
three hours per new 1k mixed continuation is a rough estimate, excluding queue;
the three arms run concurrently. The earlier five-pilot allocation's 10h20m
included E running after A and should not be interpreted as A's runtime.

## Clipping and adapter continuation results

All branches completed 1,000 additional mixed updates (500 obs + 500 OM4),
ending at 3k total with 1,500 updates from each source. No cooldown or EMA reset.
All four validation checkpoints were finite; all three select their 3k endpoints.

| Branch | Final RMSE blend ↓ | Obs updates clipped | OM4 updates clipped |
|---|---:|---:|---:|
| clip1 | 1.165185 | 100.0% | 100.0% |
| clip5 | 1.164020 | 25.2% | 6.0% |
| adapter | 1.161374 | 100.0% | 100.0% |

![Matched continuation curves](assets/monthly-latent-pilots-2026-10-16/continuation-curves.png)

Clip5 initially destabilizes the continuation: its EMA blend rises to about
1.75 at update 2,250, versus 1.25 for clip1, before recovering. Both source
training losses also spike. The threshold change is abrupt while retaining Adam
moments accumulated under clip1; this test does not isolate whether a gradual
change or training with clip5 from the beginning would behave differently.

Clip5 improves the final blend by 0.10% versus clip1; the adapter improves it by
0.33%. These are tiny single-seed differences, not convincing evidence of improved
forecast skill. Heavy clipping was real and changing it substantially altered
optimization, but this test does not show that clipping1 was the main obstacle.
Likewise this small affine forcing adapter does not show a meaningful benefit;
it does not rule out more expressive or physically constrained forcing alignment.

| Branch | SST °C | Velocity m/s | EKE m²/s² | OHC 0–700 GJ/m² | OHC 700–2,000 GJ/m² |
|---|---:|---:|---:|---:|---:|
| clip1 | 0.75890 | 0.139835 | 0.021602 | 0.85163 | 0.63128 |
| clip5 | 0.72716 | 0.140836 | 0.021636 | 0.83540 | 0.64846 |
| adapter | 0.76309 | 0.139859 | 0.021592 | 0.83195 | 0.63282 |

Clip5 has 4.2% lower SST RMSE but 2.7% worse deep OHC and 0.7% worse velocity
RMSE than clip1; the pooled blend hides these tradeoffs. The adapter lowers
upper-ocean OHC error by about 2.3%, with little change in the other components.
These component differences also need replication before a scientific claim.

The adapter learned rather than remaining frozen: raw diagonal weights end at
0.96085/0.95912/0.96833, largest absolute off-diagonal about 0.01009, and largest
absolute bias about 0.00159. Both new Adam parameter states record exactly 500
OM4 updates. Saved weights are finite. This mainly resembles a modest rescaling
of normalized forcing, not a large change of representation.

For context, the previous obs-only cooldown endpoint scored 1.11824 versus
1.16518 for continued mixed training at clip1. That is not an isolated cooldown
comparison: both the LR schedule and source mixture differ. The continuation
results support prioritizing objective/data-exposure questions over interpreting
the original large pre-clipping norms as a sufficient explanation of performance.

[Full curves and component records](assets/monthly-latent-pilots-2026-10-16/continuation-results.json).
- [clip1 W&B](https://wandb.ai/ocean_emulators/observational-transfer/runs/5a7ce70bf5f6)
- [clip5 W&B](https://wandb.ai/ocean_emulators/observational-transfer/runs/9ad43d840011)
- [adapter W&B](https://wandb.ai/ocean_emulators/observational-transfer/runs/4ff5004a2eac)

This allocation used 11.8011 GPU-hours including qualification (three active
training GPUs on a required four-GPU host). Total completed campaign allocation,
including earlier pilots and diagnostics: **54.3739 GPU-hours**. This report adds
no test-set or annual-rollout claims and launches no further experiments.
