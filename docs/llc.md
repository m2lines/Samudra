<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Coarsened LLC on the Samudra grid

The `llc` source reads `LLC4320_samudra_1deg.zarr`, coarsened and regridded onto
Samudra's Gaussian latitude/longitude grid. Native 13-face datasets and tile
caches are unsupported; the former `face` and `i/j` crop options have been removed.

The store stays compact: `U`, `V`, `Theta`, and `Salt` have dimensions
`(time, k, y, x)`, with 51 depth levels. `Eta`, `oceTAUX`, `oceTAUY`, `oceQnet`,
and `oceFWflx` have dimensions `(time, y, x)`. The canonical reader exposes
logical channels such as `U_0` as views of these arrays; it does not rewrite the
store or generate per-level Zarr arrays.

The preprocessing used to create this store centered and rotated velocities and wind stresses
into east/north components. The loader preserves those values. It uses
`lat(y)` and `lon(x)` for geographic axes, preserves `rA` as `areacello`, and
builds Boolean masks from `wet_area(k, y, x) > 0`. Surface fields use the first
level's mask. This includes partially wet coastal cells. Existing loss and
metric weighting remain unchanged.

Normalization comes from the embedded `*_mean(k)` and `*_std(k)` arrays, or
`*_mean(one)` and `*_std(one)` for surface fields. The preset sets
`data_location`, `data_means_location`, and `data_stds_location` to the same store. Statistics must be finite, with positive
standard deviations. The supplied statistics were computed from 96 sampled
hours in the training interval, using wet-area weighting.

The default `all` selection predicts U/V/Theta/Salt at all 51 levels plus Eta
(205 channels). Its four forcing channels are wind stress, heat flux, and
freshwater flux (`oceTAUX`, `oceTAUY`, `oceQnet`, `oceFWflx`). Eta is a prognostic
state and is no longer supplied as observed forcing. Old LLC checkpoints with
the former channel conventions are incompatible with this preset.

## Training and evaluation

Set `experiment.data_root` to the directory containing the store:

```bash
samudra train samudra_llc/train.yaml --experiment.data_root /path/to/coarse_LLC4320
samudra eval samudra_llc/eval.yaml --experiment.data_root /path/to/coarse_LLC4320 \
  --ckpt_path /path/to/checkpoint.pt
```

The training interval is 2011-09-13 00:00 through 2012-09-13 23:00 UTC; validation
is 2012-09-14 00:00 through 2012-11-15 14:00 UTC. Endpoints are inclusive and
cannot overlap. Timestamps remain Gregorian and hourly. `data_stride: [1]`
uses hourly spacing within training windows; a larger stride selects spaced
snapshots without temporal averaging. The presets explicitly use two input
and two output states, with the existing Samudra model on the Gaussian grid and dynamic
MSE loss. Standalone evaluation covers ten days within the validation interval.

CPU/Xarray loading is supported. Rust loading still rejects LLC; native
acceleration, new loss functions, and tile/domain-parallel machinery are outside
this support path.

## Bounded qualification

From a checkout, run on an allocated CPU node or GPU:

```bash
python scripts/smoke_llc.py --store /path/to/LLC4320_samudra_1deg.zarr \
  --output /path/to/new-smoke-output --commit COMMIT_SHA --backend cpu
```

This uses all channels and the full spatial grid with a U-Net with channel widths of 8 and 16,
two input states, one output state, four optimizer updates, ordinary validation,
and two-step rollout validation. It reloads the saved checkpoint for a four-hour
autoregressive evaluation and writes predictions. Checks require changed,
finite weights, finite validation/evaluation metrics, finite wet-cell predictions,
and preserved geographic coordinates. `receipt.json` records the source commit,
metadata hash, model size, progress, checkpoint, and metrics; the resolved
training and evaluation configurations are saved alongside it. W&B is disabled
for this engineering check. The result does not measure forecast skill.
