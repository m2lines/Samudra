<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native quarter-degree observation experiment

The [4k/6k auxiliary reproductions](early-rmse-6k-2026-10-10.md#auxiliary-loss-reproductions)
and matched eight-model annual comparison are complete. On October 11 the user authorized **U-global, observation-only, and the
best one or two variants from the 6k comparison**, judged at both 30 and 365 days.
All quarter-degree runs use 6,000 total updates. The training dataset is verified;
four-arm GPU qualification **212376** is queued after the real-data CPU preflight
passed. Quarter-degree production
has not yet been qualified or submitted. Reporting both native-resolution and
matched 1° metrics remains agreed.

## Runs

| Name | Processor and predicted state | OM4 updates | Observation updates |
|---|---|---|---|
| U-global-quarter | Global 720×1440; same U-Net widths/depth and approximately 63M parameters | 3,000 recent-period examples at ¼° | 3,000 at ¼° |
| U-obs-only-quarter | Same global processor and initializer, both randomly initialized | None | 6,000 at ¼° |
| U-early-quarter | Global 720×1440; same 77 physical channels; native analogue of the day-365 early-fine winner | 1,500 recent + 1,500 from 1958–1974, all at ¼° | 3,000 at ¼° |
| U-early-latent-quarter | Same global processor plus 10 recurrent latent channels on every task; native analogue of the day-30 early-latent winner | 1,500 recent + 1,500 from 1958–1974, all at ¼° | 3,000 at ¼° |

Train fresh with seed 1729, fixed LR 1e-4, the existing physical losses, effective
batch of eight, five-day steps and unchanged history/rollout lengths. OM4-plus-obs
arms retain the stretched progressive mixture. Observation-only uses observation
examples throughout, with the same initializer reconstruction and forecast losses;
it receives no OM4 training examples. This matches total optimizer updates, not
observation exposure or exact FLOPs. Preserve chronological splits.

## Choosing the variants

The completed [annual comparison](early-rmse-6k-2026-10-10.md#matched-annual-comparison)
selects **U-multitask-early-latent at day 30 (0.8714)** and
**U-multitask-early-fine at day 365 (0.9561)**. Their union gives the two early-data
arms above. Seed-1729 U-global scores 0.8924/1.0286. This is the result of the
prespecified rule below, not a new ranking based on quarter-degree outcomes.

The rule was to finish the auxiliary reproductions and evaluate the **fixed validation-selected
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
Recent and earlier OM4 examples will share the native-quarter preprocessing and
forcing statistics. The historical early-data loader normalized physical fields
directly, whereas the matched quarter reader preserves the baseline's normalization
round trip; this is a matched native-resolution recipe, not bitwise reproduction
of that historical loader.

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
no new primary-product transfer is required. The verified calendar requires
10,882 daily dates for 348 monthly examples; the three annual sequences add no
extra daily coverage. Daily plus materialized monthly/annual arrays total roughly
1.6 TB before compression, against approximately 35 TB of available project storage.
Equal OISST/OM4 array dimensions do
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

The [static geometry audit](artifacts/quarter-bringup-2026-10-11/common-grid-audit.json)
now confirms that common-support rescoring is required. With at least 90% native
wet-area coverage, the intersection retains 41,519 of 44,892 legacy surface cells,
35,803 of 38,933 upper-OHC cells, and 33,850 of 37,190 deeper-OHC cells. Removed
cosine-weighted areas are 6.62%, 7.15%, and 8.26%, respectively. These are static
model-mask differences; actual observation availability and velocity stencils
further restrict each origin. Use the same archived 1° truth and training-only
climatology for every candidate on the intersection.

Report validation curves/components, fixed endpoints and selected checkpoints,
own initialized persistence, matched rollout maps and temperature profiles.
Use the same held-out origins. Retain annual rollouts as an evaluation follow-up
to the short-lead comparison; they must not choose the training checkpoint.

### Frozen 1° comparison controls

The saved 6k forecasts have now been rescored on that common support, before any
quarter-degree training. These are **existing 1° models**, with their original
validation-selected checkpoints. Model definitions and original full-support
scores remain in the [6k report](early-rmse-6k-2026-10-10.md). Climatology is
recomputed from the same training months on the intersection; its score is 1.

| Existing model | Monthly blend | Annual cohort: day 30 | Annual cohort: day 365 |
|---|---:|---:|---:|
| U-global | 0.8418 | 0.8920 | 1.0281 |
| U-global-seed1730 | 0.8500 | 0.8954 | 1.0344 |
| U-multitask-early | 0.8402 | 0.8953 | 1.0344 |
| U-multitask-early-latent | **0.8295** | **0.8709** | 0.9731 |
| U-multitask-early-fine | 0.8345 | 0.8809 | **0.9551** |
| U-multitask-early-fine-latent | 0.8531 | 0.9036 | 1.0185 |
| U-aux01-6k | 0.8535 | 0.8870 | 1.0396 |
| U-aux01-anomaly-6k | 0.8345 | 0.8821 | 1.0013 |

Monthly scores use the 96-origin, five-component blend with surface errors
averaged across days 5/15/30. Annual scores use the separate three-origin,
four-component protocol above. The same variants lead at days 30 and 365 on
the intersection. This check does not change the selected quarter-degree recipes.

[Monthly components](artifacts/quarter-common-controls-2026-10-12/monthly-components.csv),
[annual components and individual origins](artifacts/quarter-common-controls-2026-10-12/annual-components.csv),
and [frozen inputs and verification](artifacts/quarter-common-controls-2026-10-12/)
retain the complete scores, denominators, masks and hashes. CPU job **19615441**
completed in 5m27s, using no GPUs; it checked the saved prediction payloads and
canonical truth rather than running new forecasts. Source is `81eaa1a29`;
cluster artifacts are under
`/scratch/jr7309/runs/2026-10-12-quarter-common-controls/reports`.

## Execution status

All prerequisite 1° reproductions and annual evaluations are complete. Annual
CPU audit **19607393**, eight-model GPU array **19607394**, and collector
**19607691** passed; no GPU evaluations were repeated. Selected checkpoint and
output hashes are linked in the annual report. Quarter-degree production remains
gated on real four-GPU qualification of each of the four recipes.

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

The static-grid audit and a real-data five-day preparation test passed. The latter
checked surface fields, all eight finite ERA5 inputs, native IAP profiles/OHC,
timestamps and output hashes. Grace daily preprocessing **131367**, monthly
assembly **131375**, and final statistics/verification **131376** have completed.
All 348 monthly examples and three static/statistics NPZs passed full readback;
normalization uses exactly 243 training months. The
[readiness receipt](artifacts/quarter-bringup-2026-10-11/observations-ready.json.gz)
pins the complete inventory and statistics inputs. Daily preparation used
`dd79f4630`, and materialization used `ec0e0f0a9`. Annual assembly **131377** and
verification **131378** also completed: all 312 annual NPZs and three origin
manifests passed, including exact overlap with the monthly examples
([annual receipt](artifacts/quarter-bringup-2026-10-11/annual-ready.json.gz)).
The ARM compact-reader build passed (**131371**): one full 80-channel frame
and four indexed windows exactly matched independent Xarray decoding, including
missing values. Beta projection test **212269** passed forward and gradient checks
on the real grids with synthetic physical fields; repeated results were bitwise
identical under the pinned runtime. This checks the observation operator, not yet
end-to-end training on observed values.

Local groundwork now includes a bounded compact-OM4 window reader and explicit
native-IAP projection in losses and metrics. Legacy scalar-statistics outputs
were checked for exact equality on a synthetic 243-month fixture; the integrated OM4 training windows, observed-value losses and four-GPU training
are still to be qualified.

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
comparison on the common support described above. The four recipes are fixed
in the run table; no auxiliary-loss arm won the promotion rule.

The staged four-arm qualification producer is
`f190debb180d4ecfd9120dd46070c54b39c148ae`. The corrected final gate requires native
observation validation and, for OM4-trained arms, 11-origin OM4 retention,
plus exact next-update replay in a separate four-rank process for every trained
path. Earlier-data arms separately qualify recent, early and observation paths,
verify first/last/leap-boundary windows against xarray, and require gradients in
every recurrent latent output row when present. Both periods retain the original
OM4 update counter and seed stream; earlier data replaces alternate OM4 updates.
Qualification weights are diagnostic-only and rejected by production resume.
The legacy `--om4-validation-origins=12` was an upper bound: all eight completed
6k controls actually evaluated 11 origins. The unchanged August 1, 2013–October 5,
2014 context contains 87 frames and 63 complete 25-frame windows. The corrected
runner pins stride-six IDs `0,6,…,60` and the historical origin timestamps
October 30, 2013–August 26, 2014 (noon); forecast targets end September 25, 2014.
It rejects changed cohorts rather than extending validation into the test period.
CPU preflight caught the previous exact-12 guard in `81eaa1a29` before any GPU
allocation. The replacement preserves the original validation boundaries.
The runner, data, schedule, latent-state and early-fine tests passed **109 CPU
tests**. Each arm has its own two-hour test allocation, at most two concurrently.
Replacement array **212376** is queued; superseded **212374** was canceled before
allocation. Corrected CPU preflight **131990** passed actual recent and
first/last/leap early windows against xarray, including normalization and finite
six-tensor outputs (53.8 seconds in the audit; 2m31s allocation). Its
[receipt](artifacts/quarter-bringup-2026-10-11/cpu-windows-131990.json.gz) preserves
source/runtime/data hashes. Failed CPU attempt **131988** remains saved; neither
CPU attempt used GPUs. The correction passes 90 focused tests. There are no
production jobs or quarter-degree scientific results yet.
