<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native quarter-degree observation experiment: proposed follow-on

Run after the [4k/6k auxiliary reproductions](early-rmse-6k-2026-10-10.md#auxiliary-loss-reproductions)
finish. On October 11 the user authorized **U-global, observation-only, and the
best one or two variants from the 6k comparison**, judged at both 30 and 365 days.
All quarter-degree runs use 6,000 total updates. Quarter-degree production has not yet been
qualified or submitted; short preparation tests are underway. Reporting both native-resolution and matched 1°
metrics remains agreed.

## Runs

| Name | Processor and predicted state | OM4 updates | Observation updates |
|---|---|---|---|
| U-global-quarter | Global 720×1440; same U-Net widths/depth and approximately 63M parameters | 3,000 recent-period examples at ¼° | 3,000 at ¼° |
| U-obs-only-quarter | Same global processor and initializer, both randomly initialized | None | 6,000 at ¼° |
| Promoted variant(s), names fixed after the 6k comparison | Same native global grid; retain the selected data, latent-state or auxiliary-supervision change where applicable | 3,000; for an earlier-data variant, 1,500 recent + 1,500 from 1958–1974 | 3,000 at ¼° |

Train fresh with seed 1729, fixed LR 1e-4, the existing physical losses, effective
batch of eight, five-day steps and unchanged history/rollout lengths. OM4-plus-obs
arms retain the stretched progressive mixture. Observation-only uses observation
examples throughout, with the same initializer reconstruction and forecast losses;
it receives no OM4 training examples. This matches total optimizer updates, not
observation exposure or exact FLOPs. Preserve chronological splits.

## Choosing the variants

First finish the auxiliary reproductions and evaluate the **fixed validation-selected
checkpoints from the 6k runs** for U-global (both seeds), the four earlier-data
variants, U-aux01 and U-aux01-anomaly. Do not substitute the older 4k annual results.
Do not choose a different checkpoint using annual test outcomes.

Reuse the [matched annual comparison](presentation-annual-2026-10-08.md#fixed-evaluation-protocol):
three January starts (2015, 2018, 2021), continuous 73-step forecasts, and the same
four-component climatology-normalized RMSE blend at days 30 and 365 (SST,
geostrophic velocity, OHC 0–700 m, OHC 700–2000 m). Pool equal-origin MSE before
taking square roots. Each lead has its own common climatology denominators.
Report EKE separately: annual temporal-anomaly EKE is not the fixed-lead monthly
cohort EKE used in the five-component training-validation blend.

Promote the best non-baseline variant at each horizon, taking their union (one
or two variants). Deduplicate recipes that become identical at native quarter
resolution: early-coarse and early-fine without latent channels both map to
U-early-quarter; their latent counterparts both map to U-early-latent-quarter. Inspect component errors and both U-global seeds alongside
that ranking; describe weak gains as uncertain. This avoids inventing a weight
between short and long horizons. The three starts are a small, previously viewed
test cohort: using them to choose the next wave is exploratory model development,
not an untouched test of the winning hypothesis. Show all candidate scores and
per-origin results. Quarter-degree results require their own evaluation.

This is full global ¼° evolution, not a ¼° encoder around a 1° processor.
The same convolutional architecture has a smaller receptive field in geographic
units at ¼°; hold architecture fixed in this first comparison and document that
limitation. The current early-fine model's 4× encoder/decoder cannot carry over
unchanged: earlier and recent data now share a resolution. Promoting its result
therefore tests earlier-data/latent benefit, not the original resolution-adapter
mechanism. It does not imply introducing an unrequested 1/16° source.

## Confirmed inputs and remaining preparation

Read-only Beta checks on October 10 confirmed the five-day-averaged store:

```
/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg_compact_zstd
```

This is the averaged dataset subsequently prepared in thread
`01a0e8d3-f694-7830-aad3-6d9f00717c95`, not its separate snapshot store.
It contains OM4, mean and standard-deviation stores; packed depth arrays use
Blosc/Zstd compression. Saved conversion receipts verify lossless payload
read-back. Compressed payload is approximately 635 GB. The 4,745 timestamps span
1958–2022, on a Gaussian 720×1440 grid. A direct coordinate audit found 4,728
five-day intervals and 16 six-day intervals at leap-year boundaries (December 28
to January 3, Julian calendar). Keep the published timestamps and existing OM4
step convention; do not reject or silently rewrite those boundary intervals. This is a prepared regridded product,
not the original tripolar simulation grid. Its release differs from the current
Torch wave's quarter-degree v2 source, so comparisons include that provenance
difference. All quarter-degree arms will use the same release.

Observation sources already exist under:

```
/projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/prepared
```

| Product | Confirmed source resolution | Proposed treatment |
|---|---|---|
| OISST | Regular ¼° daily | Remap to Gaussian ¼°; preserve the existing five-day averaging convention |
| DUACS ADT/velocity | Regular ⅛° daily | Aggregate/remap with valid ocean support; retain the established velocity diagnostic definition |
| ERA5 surface forcing | Regular ¼° hourly | Align spatially and temporally to the same observation examples |
| IAP temperature/salinity | ½° monthly, 41 depths | Retain its native observational support for supervision/evaluation through a model-to-observation operator |

The checked pilot caches are still 180×360. A new quarter-degree cache is needed;
no new primary-product transfer is required. Equal OISST/OM4 array dimensions do
not mean identical coordinates: regular versus Gaussian latitude differs.
Do not manufacture fine-scale subsurface truth by upsampling IAP and counting
the interpolated cells as independent observations. Preserve monthly OHC depth
integration, masks, coast handling, calendar alignment, and split isolation.

## Selection and reporting

Agreed selection: retain the same five-component **RMSE-only** blend and equal
weights, evaluated at the new observation supports, with a frozen training-only
seasonal-climatology control and validation denominators established on that
grid. Full-field errors remain neither detrended nor deseasonalized. Spectra do
not enter checkpoint selection.

Also evaluate conservatively coarsened predictions through the unchanged 1°
observation operators and frozen reference for a direct comparison with the
current runs. Calculate velocity/EKE after the chosen remapping where required
by each operator; do not interchange derivative, squaring and averaging.
Before calling the 1° comparison numerically identical to the existing one,
verify that quarter-to-one-degree wet coverage supports every required legacy
cell. If it does not, freeze a common intersection and rescore both the saved
1° comparators and their climatology there; label those scores as common-support
rescoring rather than the original published numbers. Native-resolution and
1° blend values are different protocols and must be labelled separately.

Report validation curves/components, fixed endpoints and selected checkpoints,
own initialized persistence, matched rollout maps and temperature profiles.
Use the same held-out origins. Retain annual rollouts as an evaluation follow-up
to the short-lead comparison; they must not choose the training checkpoint.

## Execution status

The prerequisite annual comparison is queued on Torch after the auxiliary wave's
verified result export: audit **19600907** → eight-model evaluation **19600908**
(two GPUs concurrently) → collection **19600909**. It uses evaluation producer
`56aac9375a59a7204cd54d85a0c5a3b8e9383d59` and the existing annual data/runtime at
`/scratch/jr7309/runs/2026-10-11-rmse-6k-annual`. The audit freezes completed 6k
selected checkpoint hashes and requires each run's 96-origin short-horizon
completion record. These are queued evaluations, not new training or completed
annual results. Quarter-degree production remains gated on these results and
real-data qualification.

Beta diagnostic **212260** completed all three synthetic 720×1440 full-objective
backward tests on one GB200 (`beta_test`, 0.1281 allocated GPU-hours). The exact
62,967,680-parameter U-global model used bfloat16 and activation checkpointing:

| Task | Forecast leads | Peak allocated / reserved GPU memory | Mean microbatch time after first update |
|---|---:|---:|---:|
| OM4 | 6 | 27.1 / 29.2 GiB | 1.75 s |
| Observation | 6 | 43.7 / 49.9 GiB | 3.19 s |
| Observation | 7 | 47.2 / 53.4 GiB | 3.67 s |

All three updates per case had finite gradients reaching the expected components.
[Raw records](artifacts/quarter-bringup-2026-10-11/) preserve runtime, source and
configuration hashes. These are synthetic microbatch-one measurements, excluding
data preparation; they do not establish effective-batch-eight throughput,
native-IAP operator behavior, or four-GPU correctness. No weights were saved.
Production remains unsubmitted.

Local groundwork now includes a bounded compact-OM4 window reader and explicit
native-IAP projection in losses and metrics. Legacy scalar-statistics outputs
were checked for exact equality on a synthetic 243-month fixture; real compact
reads, real-data operators and four-GPU training are still to be qualified.

## Beta bring-up and decision points

Use one Beta node (four B200s) per run, initially sequentially. Qualify the actual
U-global initializer plus autoregressive training, not just a single processor
forward pass. Preserve the effective batch when distributing accumulation.
The existing pilot preloads whole OM4 periods on GPU and prepared observations
on CPU; this cannot simply be scaled by changing grid dimensions. Integrate the
verified compact reader with bounded window loading/caching, then check exact
channel/time/mask equivalence, fitting, gradients, resume and numerical replay.
Rebuild the current branch's ARM64 native extension: its `FlatOm4ReadPool` API
differs from the earlier Beta producer's `ZarrReadPool`. The current
`CompactOm4Reader` provides bounded explicit time/channel reads, including packed
depth arrays; normalization and masking remain the experiment's responsibility.
Use activation checkpointing as needed and measure memory, throughput and
validation overhead before giving a production ETA. Data parallelism alone
does not make an oversized single sample fit. Quarter-degree fields have 16×
as many cells; neither 1° runtime nor the other thread's different model provides
a reliable runtime estimate here. Wait for capacity rather than moving jobs
repeatedly; Beta had no idle nodes at the inventory snapshot.

Use **native-support RMSE selection**, with the unchanged 1° score reported for
comparison. The agreed scope is two baseline arms plus one or two promoted
variants, all trained fresh for 6k updates. Final variant identities depend on
the completed matched 30/365-day comparison.

If an auxiliary variant wins, preserve what its target means: the current q
labels measure variance inside a **1° cell using ¼° velocities**. They are not
unresolved variance inside a ¼° cell. A native-quarter model can retain that
supervision through a head aggregated to the original 1° support; it cannot
silently relabel q as finer-scale truth. Document and qualify the head/aggregation
before submitting that arm. This transfer changes the relation of the auxiliary
target to the resolved state, so it is not an exact resolution-only reproduction.
