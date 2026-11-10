# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Use Samudra's OM4 adapter; reduce compact levels together to share reads."""

import sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

from samudra.config import Om4DataSourceConfig, TrainConfig
from samudra.utils.data import stack_levels


def open_canonical(cfg):
    source = cfg.data.sources[0]
    assert isinstance(source, Om4DataSourceConfig)
    # Boundary anomalies are irrelevant to prognostic tendency statistics.
    # Omitting that derived boundary channel avoids computing the full hfds
    # climatology just to read physical ocean fields and their state stds.
    source = source.model_copy(update={"boundary_vars_key": "tau_hfds"})
    root = cfg.experiment.resolved_data_root
    return source.canonicalize_datasets(
        root.resolve(source.data_location).open(chunks={}),
        root.resolve(source.data_means_location).open(chunks={}),
        root.resolve(source.data_stds_location).open(chunks={}),
    )


def main():
    cfg = TrainConfig.from_yaml(sys.argv[1])
    output = Path(sys.argv[2]) / "stats/persistence_rms.zarr"
    if output.exists():
        raise FileExistsError(output)
    data, _, _, layout = open_canonical(cfg)
    data = stack_levels(data, layout)
    time = cfg.data.sources[0].train_time
    calendar = data.time.dt.calendar
    start = cftime.datetime.strptime(str(time.start), "%Y-%m-%d", calendar=calendar)
    end = cftime.datetime.strptime(str(time.end), "%Y-%m-%d", calendar=calendar)
    data = data.sel(time=slice(start, end))
    if data.sizes["time"] < 2:
        raise ValueError("Need at least two training samples")
    stats = {}
    for variable in ("uo", "vo", "thetao", "so", "zos"):
        field = data[variable]
        wet = (
            data.wetmask if "lev" in field.dims else data.wetmask.isel(lev=0, drop=True)
        )
        change = field.diff("time").where(wet.astype(bool), 0).astype("float64")
        rms = np.sqrt((change**2).mean(("time", "lat", "lon"), skipna=False)).compute()
        if not bool(np.isfinite(rms).all() & (rms > 0).all()):
            raise ValueError(f"Invalid training tendency statistics for {variable}")
        if "lev" in rms.dims:
            for level in range(len(layout.depth_levels)):
                stats[f"{variable}_{level}"] = xr.DataArray(float(rms.isel(lev=level)))
        else:
            stats[variable] = xr.DataArray(float(rms))
        print(variable, rms.values, flush=True)
    result = xr.Dataset(stats)
    result.attrs.update(
        description="RMS native one-step change; physical units; land counted as zero",
        time_range=f"{str(time.start)} to {str(time.end)}",
        time_stride=1,
        n_training_states=data.sizes["time"],
        calendar=calendar,
        adapter="Om4DataSourceConfig.canonicalize_datasets; shared compact-level reductions",
    )
    result.to_zarr(output, mode="w")


if __name__ == "__main__":
    main()
