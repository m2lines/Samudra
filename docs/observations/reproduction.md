<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global observation / OM4 task reproduction

This is the **Global physical-only** model from the September 30 comparison in
[PR #892](https://github.com/m2lines/Samudra/pull/892): a surface-history initializer
and shared autoregressive processor, task-specific input adapters, affine
InstanceNorm, and 77 physical-labelled state channels. There are no additional
memory channels. The reference model has **62,966,546 parameters**.

The harness is deliberately a fixed Python training plan. It uses Samudra's typed
`UNetBackboneConfig`, `BlockConfig`, `DataConfig`, batch loader, observation metric
kernels and W&B logger. Network and OM4 data components have YAML presets; the
optimizer, losses, task schedule and validation policy are ordinary code in
`samudra.tasks.observation`. This does not change `samudra train` or introduce a
multitask framework. Historical arms, cluster launchers and report machinery are
not required.

## Data preparation

Start from the upstream **daily** OISST SST, DUACS ADT/ugos/vgos, monthly Argo-IAP
temperature/absolute salinity and **hourly** ERA5 stores, named `oisst.zarr`,
`duacs.zarr`, `argo-iap.zarr`, `era5-surface.zarr` in `SOURCE`. Do not substitute the
centered-five-day observation stores used by the older native-grid evaluation:
this task constructs its own exact five-day bins around each monthly origin.
The sampled source metadata hashes are in the verification artifact below. The
original stores are under `/mnt/home/jrusak/data/obs_full_range/prepared` on Empire
AI; the prepared training set is `/scratch/jr7309/data/obs-d-pilot` on Torch.
Provider downloads and inter-cluster transfers are outside this harness.

OM4 input is the one-degree unblurred `om4_onedeg_v3`, with `OM4.zarr`,
`OM4_means.zarr` and `OM4_stds.zarr`. Export its target grid using the normal
Samudra environment:

```bash
python -m samudra.observations.grid --om4 "$OM4" --output "$DAILY/grid.npz"
```

For preprocessing, use a separate environment with numpy, scipy, pandas, xarray,
zarr<3 and **gsw==3.6.23**, as in the original preprocessing. GSW's newer numpy
requirements should not alter the training environment. `prepare.py` is standalone
and does not import torch or Samudra training code. It can run on a CPU host:

```bash
python src/samudra/observations/prepare.py \
  --source "$SOURCE" --grid "$DAILY/grid.npz" --output "$DAILY" \
  --product sst --year 1993
```

Run each product (`sst`, `adt`, `era5`, `iap`) for 1993–2022. Also prepare the first
five days of 2023 for sst/adt/era5 (`--year 2023 --limit 5`); December 2022's last
forecast bin crosses the year boundary. ERA5 needs the next midnight accumulation.
Then assemble monthly samples and fit training-only statistics:

```bash
python -m samudra.observations.archive --daily "$DAILY" --output "$OBS"
```

The preparation preserves conservative area remapping, 90% finite source-area
coverage, strict complete five-day temporal support, interpolated depth brackets,
absolute-to-practical salinity conversion, native-depth OHC integration and ERA5
accumulation timing/units. Inputs carry explicit validity; missing normalized
surfaces are zero placeholders and only observed surfaces are copied into the
initializer output. There is no climatology fallback in model inputs.

Splits are 243 training months (May 1993–July 2013), nine validation months
(November 2013–July 2014), and 96 test months (2015–2022). Statistics retain the
reference experiment's ±60° fitting domain; **inputs, losses and scoring are
global** on finite wet support. These are different contracts. Climatology is
training-only and is used for scoring controls, not model input filling.

## Training and evaluation

```bash
python -m samudra.tasks.observation train \
  --observations "$OBS" --om4 "$OM4" --output "$RUN" \
  --entity YOUR_ENTITY --project observational-transfer --name global-physical-only
```

The fixed plan has 16,000 optimizer updates, exactly 8,000 per task, seed 1729,
effective batch eight, AdamW at 1e-4, weight decay 0.01 and gradient clipping one.
The cumulative observation count after update `s` is
`floor(8000 * s * (16000 + 3*s) / (4*16000²))`; observation frequency increases
through training. There is no separate reconstruction phase or exclusive-observation
finish. The model receives 19 historical five-day surface/forcing frames plus
geographic and seasonal context, and initializes two states. Forecast training
covers six OM4 steps or six/seven steps for an observation calendar month.

OM4 retains its native forcing normalization and dense full-state objective,
with reconstruction and artificial-gap completion. Observations use a learned
8→32→3 ERA5 adapter, 0.8 monthly T/S + 0.1 SST + 0.1 SSH forecast loss, plus 0.1
interior reconstruction and 0.1 surface completion. Later surfaces are used only
in the separate reconstruction target path, never to correct the forecast.
Coefficients do not represent percentages of gradient contribution. Internal
velocity/deep channels are not claimed to be observationally validated ocean states.

This is a single-GPU harness. Default OM4 frame caching uses host memory; allow
roughly 80–100 GiB host RAM for the frame/sample caches and runtime. On a 96-GiB
GPU, `--cache-device cuda` reproduces the earlier resident GPU cache strategy.
The cache is populated by Samudra's existing batch loader and checked against
uncached endpoint windows before use. The experiment branch's unmerged Rust
loader is not a dependency of this PR.

Restart the identical command to resume `last.pt`, including optimizer, exact
per-task exposure, sample stream, selection reference and RNG state. `--stop-after N`
ends an operational qualification run after N total updates without changing the
16k schedule; omit it to continue. SIGTERM/USR1 request a checkpoint after the
current update. Completed endpoints are separate from the validation-selected
`best.pt`; partial runs do not get completion markers.

```bash
python -m samudra.tasks.observation evaluate \
  --observations "$OBS" --om4 "$OM4" --output "$RUN" --checkpoint best.pt
```

Selection remains global integrated-plus-spectral validation every 100 updates.
The five integrated components are SST, geostrophic velocity, EKE, and OHC in
0–700/700–2000 m; regional SST/ADT/EKE spectra cover days 5, 15 and 30. Velocity
metrics derive from SSH and retain the ±5° equatorial validity exclusion. Required
spectral keys are frozen at qualification. Test scoring uses the common held-out
climatology denominator, as in the report; test data never selects checkpoints.
Evaluation also reports initialized-state persistence and climatology.

W&B receives total/per-task updates, observation and OM4 loss components,
learning rate, gradient norm, validation composite, **all nested scalar integrated
and spectral components**, and OM4 retention. Test/annual namespaces are separate.
Full spectral arrays and channel profiles remain in JSON outputs. The W&B entity,
project and mode are configurable; requested initialization failure is fatal.

For annual evaluation, first prepare each of the three origins (2015-01-01,
2018-01-01, 2021-01-01) from the same daily archive:

```bash
python -m samudra.observations.annual --daily "$DAILY" \
  --origin 2015-01-01 --output "$ANNUAL/2015-01-01"
python -m samudra.tasks.observation annual \
  --observations "$OBS" --om4 "$OM4" --output "$RUN" \
  --annual-data "$ANNUAL" --checkpoint endpoint.pt
```

Annual evaluation performs 73 autoregressive steps from one initialization,
using prescribed future ERA5 and no future surface inputs. It reports errors at
5/15/30/90/180/365 days. The monthly test covers eight years of independently
initialized dates, not an eight-year continuous rollout.

## Reproduction checks

[Machine-readable evidence](artifacts/reproduction-checks.json.gz) records:

- Four product shards regenerated from upstream data: SST, ADT/velocities, ERA5
  for January 1, 1993; IAP T/S and OHC for January 1993. All arrays, finite masks,
  coverage fractions, depths and timestamps match exactly.
- Four monthly examples regenerated from the original daily archive: May 1993,
  February 2000 (leap year), November 2013 and December 2022 (cross-year forecast).
  All eight array members match exactly. Their file hashes and target-grid hash
  also match the dataset actually used on Torch.
- The extracted model's seeded state, parameter count, initialized fields and
  one-step outputs match the experimental implementation exactly on a 64×128
  mixed-validity fixture.

The included `scripts/verify_observation_preprocessing.py` reruns the data checks.
These are sampled comparisons, not a claim that every upstream chunk was rebuilt.
Targeted tests cover global scoring, derivative-stencil support, missingness,
calendar weights, conservative remapping, future-input isolation, W&B scalar
flattening, deterministic sampling and optimizer/RNG resume.
