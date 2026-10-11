<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native quarter-degree observation experiment: proposed follow-on

Run after the [current seven-run comparison](early-rmse-6k-2026-10-10.md)
finishes. The user approved this plan, including reporting both resolutions,
on October 10. No follow-on jobs have been submitted. Choose the budget and optional
second arm using validation results, before looking at the new held-out results.

**October 11 decision:** the current wave has completed. Its 6k U-global seeds
improve selected validation by 5–7% over the 4k reproduction, so use **6,000
updates** for the quarter-degree baseline. The best early-fine validation gain
is smaller than the observed two-seed U-global spread; no optional second arm
is promoted yet. This remains provisional while the user-requested
[matched 4k reproductions](early-rmse-4k-reproduction-2026-10-11.md) test the
earlier advantage; late-window means favor early fine more than minima do. The [completed report](early-rmse-6k-2026-10-10.md)
contains the component and held-out comparisons.

## Runs

| Proposed name | Processor and predicted state | OM4 updates | Observation updates |
|---|---|---|---|
| U-global-quarter | Global 720×1440; same U-Net widths/depth and approximately 63M parameters | 3,000 recent-period examples at ¼° | 3,000 at ¼° |
| U-early-quarter, optionally U-early-latent-quarter | Same global ¼° processor; add 10 recurrent channels only if that variant merits promotion | 1,500 recent + 1,500 from 1958–1974, all at ¼° | 3,000 at ¼° |

Train fresh with seed 1729, fixed LR 1e-4, the existing physical losses, effective
batch of eight, five-day steps, history/rollout lengths, and the stretched
progressive mixture. Preserve chronological train/validation/test splits.
The completed long-run versus original-schedule reproduction supports 6,000
updates. A single
optional arm should have a convincing validation improvement over U-global,
considering component tradeoffs and the observed two-seed U-global variation.
This small seed sample does not establish significance.

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
1958–2022, on a Gaussian 720×1440 grid. This is a prepared regridded product,
not the original tripolar simulation grid. Its release differs from the current
Torch wave's quarter-degree v2 source, so comparisons include that provenance
difference. Both proposed quarter-degree arms will use the same release.

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
Native-resolution blend values and the historical 1° blend are different
protocols and must be labelled separately.

Report validation curves/components, fixed endpoints and selected checkpoints,
own initialized persistence, matched rollout maps and temperature profiles.
Use the same held-out origins. Retain annual rollouts as an evaluation follow-up
to the short-lead comparison; they must not choose the training checkpoint.

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
comparison. The agreed defaults are fresh training, unchanged U-Net architecture,
and at most one promoted earlier-data variant;
the completed current wave determines its identity and the 4k/6k budget.
