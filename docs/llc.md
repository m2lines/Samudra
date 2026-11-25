<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Coarsened LLC on the Samudra grid

The `llc` source reads `LLC4320_samudra_1deg.zarr`, coarsened and regridded onto
Samudra's Gaussian latitude/longitude grid.

The store stays compact: `U`, `V`, `Theta`, and `Salt` have dimensions
`(time, k, y, x)`, with 51 depth levels. `Eta`, `oceTAUX`, `oceTAUY`, `oceQnet`,
and `oceFWflx` have dimensions `(time, y, x)`. The canonical reader exposes
logical channels such as `U_0` as views of these arrays.

The preprocessing used to create this store centered and rotated velocities and wind stresses
into east/north components. The loader preserves those values. It uses
`lat(y)` and `lon(x)` for geographic axes, preserves `rA` as `areacello`, and
builds Boolean masks from `wet_area(k, y, x) > 0`. Surface fields use the first
level's mask. This includes partially wet coastal cells.

Normalization comes from the embedded `*_mean(k)` and `*_std(k)` arrays, or
`*_mean(one)` and `*_std(one)` for surface fields. An LLC source reads
its statistics from the data store unless `data_means_location` or `data_stds_location` says otherwise. Statistics must be finite, with positive
standard deviations. The supplied statistics were computed from 96 sampled
hours in the training interval, using wet-area weighting.

The source defaults to `prognostic_vars_key: all` and `boundary_vars_key: all`.
It predicts U/V/Theta/Salt at all 51 levels plus Eta (205 channels).
Its four forcing channels are wind stress, heat flux, and
freshwater flux (`oceTAUX`, `oceTAUY`, `oceQnet`, `oceFWflx`). Eta is a prognostic
state.

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
and two output states, with the Samudra model on the Gaussian grid and
[dynamic MSE loss](utils/loss.md). Standalone evaluation covers ten days within
the validation interval.

The presets use CPU/Xarray data loading.
