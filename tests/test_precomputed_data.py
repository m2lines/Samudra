# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stored derived fields are ordinary channels; visualization reads reductions."""

import dataclasses

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from samudra.config import RustDataLoadingConfig
from samudra.constants import build_om4_layout
from samudra.utils.data import CanonicalSource
from samudra.utils.location import LocalLocation
from samudra.viz.core import Viz, VizRun, prepare_viz_groundtruth
from tests.test_config import om4_source_config


@pytest.fixture(params=[False, True], ids=["stacked", "flat"])
def stored_source(request):
    config = om4_source_config(
        prognostic_vars_key="thetao_1", boundary_vars_key="tau_hfds_hfds_anom"
    )
    values = np.arange(24, dtype="float32").reshape(4, 2, 3)
    data = xr.Dataset(
        {
            name: (("time", "y", "x"), values + offset)
            for offset, name in enumerate(("hfds", "hfds_anomalies", "tauuo", "tauvo"))
        },
        coords={
            "time": xr.date_range(
                "2001-01-01", periods=4, freq="5D", calendar="julian", use_cftime=True
            ),
            "y": [-10.0, 10.0],
            "x": [0.0, 1.0, 2.0],
        },
    )
    if request.param:
        data["thetao_0"] = (("time", "y", "x"), values)
        data["mask_0"] = (("y", "x"), np.ones((2, 3), dtype=bool))
    else:
        data = data.assign_coords(lev=[2.5])
        data["thetao"] = (("time", "lev", "y", "x"), values[:, None])
        data["wetmask"] = (("lev", "y", "x"), np.ones((1, 2, 3), dtype=bool))
    fields = data[[name for name in data.data_vars if "time" in data[name].dims]]
    return config, data, fields.mean(("time", "y", "x")), fields.std(("time", "y", "x"))


def canonical_source(config, data, means, stds):
    data, means, stds, layout = config.canonicalize_datasets(data, means, stds)
    return CanonicalSource.from_datasets(
        data,
        means,
        stds,
        data_layout=layout,
        prognostic_var_names=layout.prognostic_var_names,
        boundary_var_names=layout.boundary_var_names,
    )


@pytest.mark.parametrize("missing", ["data", "means", "stds"])
def test_anomalies_and_statistics_must_be_stored(stored_source, missing):
    config, data, means, stds = stored_source
    inputs = {"data": data, "means": means, "stds": stds}
    inputs[missing] = inputs[missing].drop_vars("hfds_anomalies")
    with pytest.raises(ValueError, match="Precompute required fields"):
        canonical_source(config, **inputs)


@pytest.mark.parametrize("compress", [False, True])
def test_rust_reads_stored_anomalies_identically_to_python(
    stored_source, tmp_path, compress
):
    pytest.importorskip("samudra_rust_loader")
    config, data, means, stds = stored_source
    path = tmp_path / "data.zarr"
    data = data.chunk({"time": 1})
    data.to_zarr(
        path,
        consolidated=True,
        encoding=None
        if compress
        else {name: {"compressor": None} for name in data.data_vars},
    )
    source = canonical_source(config, xr.open_zarr(path), means, stds)
    backend = RustDataLoadingConfig().build_source_backend()
    native = dataclasses.replace(
        source,
        _reader=backend.build_reader(
            source.reader,
            data_location=LocalLocation(path=path),
            data_layout=source.data_layout,
        ),
    )
    indices = np.array([3, 1, 1, 0])
    channels = tuple(source.data_layout.boundary_var_names)
    np.testing.assert_array_equal(
        native.read(indices, channels), source.read(indices, channels)
    )
    np.testing.assert_array_equal(
        native.read(indices, ("hfds_anomalies",))[:, 0],
        data.hfds_anomalies.isel(time=indices).values,
    )


@pytest.mark.parametrize("flat", [False, True])
def test_visualization_startup_never_reads_spatial_fields(tmp_path, flat):
    layout = build_om4_layout()

    @delayed
    def forbidden_field_read():
        pytest.fail("Visualization startup read a spatial time series")

    data = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                da.from_delayed(
                    forbidden_field_read(), shape=(4, 19, 2, 3), dtype="float32"
                ),
            ),
            "profile_thetao": (("time", "lev"), np.arange(76).reshape(4, 19)),
            "wetmask": (("lev", "y", "x"), np.ones((19, 2, 3), dtype=bool)),
            "areacello": (("y", "x"), np.ones((2, 3))),
            "areacello_weights": (("y", "x"), np.full((2, 3), 1 / 6)),
        },
        coords={
            "time": xr.date_range("2001-01-01", periods=4, freq="5D"),
            "lev": list(layout.depth_levels),
            "y": [-10.0, 10.0],
            "x": [0.0, 1.0, 2.0],
        },
    )
    expected = data.profile_thetao[1:3].values
    if flat:
        for level in range(19):
            data[f"thetao_{level}"] = data.thetao.isel(lev=level, drop=True)
            data[f"profile_thetao_{level}"] = data.profile_thetao.isel(
                lev=level, drop=True
            )
        data = data.drop_vars(["thetao", "profile_thetao"])
    prepared = prepare_viz_groundtruth(
        "truth", xr.Dataset(), data, slice("2001-01-06", "2001-01-11"), "gaussian"
    )
    prediction = data[
        [name for name, field in data.data_vars.items() if "time" in field.dims]
    ].isel(time=slice(1, 3))
    viz = Viz(
        str(tmp_path), "truth", [VizRun("model", prediction, ["thetao"])], prepared
    )
    np.testing.assert_array_equal(
        viz.profile_groundtruth.thetao.transpose("time", "lev"), expected
    )
    np.testing.assert_array_equal(
        viz.pred_dict["model"]["profile_prediction"].thetao.transpose("time", "lev"),
        expected,
    )
    assert viz.data.thetao.dims == ("time", "lev", "y", "x")
    assert not any(str(name).startswith("profile_") for name in viz.data.data_vars)
    with pytest.raises(ValueError, match="Backfill this Zarr"):
        prepare_viz_groundtruth(
            "truth",
            xr.Dataset(),
            data.drop_vars("profile_thetao_0" if flat else "profile_thetao"),
            slice(None),
            "gaussian",
        )
