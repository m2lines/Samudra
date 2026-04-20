import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from ocean_emulators.utils import apply_mask


@pytest.fixture
def processed_data():
    coords = {
        "wetmask": xr.DataArray(
            np.random.rand(19, 1080, 1440), dims=["lev", "y", "x"]
        ).astype(bool),
        "lon_b": xr.DataArray(np.random.rand(1081, 1441), dims=["y_b", "x_b"]).astype(
            "float64"
        ),
        "lat_b": xr.DataArray(np.random.rand(1081, 1441), dims=["y_b", "x_b"]).astype(
            "float64"
        ),
        "lon": xr.DataArray(np.random.rand(1080, 1440), dims=["y", "x"]).astype(
            "float64"
        ),
        "lat": xr.DataArray(np.random.rand(1080, 1440), dims=["y", "x"]).astype(
            "float64"
        ),
        "angle": xr.DataArray(np.random.rand(1080, 1440), dims=["y", "x"]).astype(
            "float64"
        ),
        "areacello": xr.DataArray(np.random.rand(1080, 1440), dims=["y", "x"]).astype(
            "float64"
        ),
        "dz": xr.DataArray(
            np.random.rand(
                19,
            ),
            dims=["lev"],
        ).astype("float64"),
        "lev": xr.DataArray(
            np.random.rand(
                19,
            ),
            dims=["lev"],
        ).astype("float64"),
        "ilev": xr.DataArray(
            np.random.rand(
                20,
            ),
            dims=["ilev"],
        ).astype("float64"),
        "x": xr.DataArray(np.random.rand(1440), dims=["x"]).astype("float64"),
        "y": xr.DataArray(np.random.rand(1080), dims=["y"]).astype("float64"),
        "time": xr.DataArray(range(10), dims=["time"]),
    }
    ds = xr.Dataset(
        {
            v: xr.DataArray(
                np.ones([10, 1080, 1440], dtype="float32"), dims=["time", "y", "x"]
            )
            for v in ["hfds", "tauuo", "tauvo", "zos"]
        }
        | {
            v: xr.DataArray(
                np.ones([10, 19, 1080, 1440], dtype="float32"),
                dims=["time", "lev", "y", "x"],
            )
            for v in ["so", "thetao", "uo", "vo"]
        },
        coords=coords,
    )
    return ds


@pytest.fixture
def input_data():
    y = xr.DataArray(np.arange(-89, 91, 1), dims=["y"]).astype("float64")
    x = xr.DataArray(np.arange(0, 360, 1), dims=["x"]).astype("float64")
    y_b = xr.DataArray(np.arange(-90, 91, 1), dims=["y_b"]).astype("float64")
    x_b = xr.DataArray(np.arange(0, 361, 1), dims=["x_b"]).astype("float64")
    time = xr.DataArray(np.arange(0, 3, 1), dims=["time"])
    lon = xr.ones_like(x) * y
    lat = y * xr.ones_like(x)
    lon_b = xr.DataArray(np.random.rand(181, 361), dims=["y_b", "x_b"])
    lat_b = xr.DataArray(np.random.rand(181, 361), dims=["y_b", "x_b"])
    # area +wetmask  is fake data for now (might have to change this for range checks later)
    areacello = x * y
    # from https://github.com/m2lines/ocean_emulators/issues/17
    dz = xr.DataArray(
        [
            5,
            10,
            15,
            20,
            30,
            50,
            70,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            800,
            1000,
            1000,
            1000,
        ],
        dims=["lev"],
    ).astype("int64")
    lev = xr.DataArray(
        [
            2.5,
            10,
            22.5,
            40,
            65,
            105,
            165,
            250,
            375,
            550,
            775,
            1050,
            1400,
            1850,
            2400,
            3100,
            4000,
            5000,
            6000,
        ],
        dims="lev",
    )
    wetmask = x * y * lev
    wetmask.data = np.random.random(wetmask.shape) > 0.25
    ocean_fraction = xr.ones_like(wetmask).astype("float64")

    coords = {
        "x": x,
        "y": y,
        "x_b": x_b,
        "y_b": y_b,
        "lev": lev,
        "time": time,
        "dz": dz,
        "areacello": areacello,
        "wetmask": wetmask,
        "ocean_fraction": ocean_fraction,
        "lon": lon,
        "lat": lat,
    }

    ds = xr.Dataset(
        {
            v: xr.DataArray(
                dsa.random.random([360, 180, 19, 3]),
                dims=["x", "y", "lev", "time"],
            )
            if v in ["so", "thetao", "uo", "vo"]
            else xr.DataArray(dsa.random.random([360, 180, 3]), dims=["x", "y", "time"])
            for v in [
                "so",
                "thetao",
                "uo",
                "vo",
                "zos",
                "hfds",
                "tauuo",
                "tauvo",
                "sithick",
                "siconc",
            ]
        },
        coords=coords,
        attrs={"m2lines/ocean-emulators_git_hash": "dummy"},
    )
    # why would they not work when passed at ds creation?

    ds_masked = apply_mask(ds.astype("float32"), wetmask)
    # make xarray-schema happy
    ds_masked = ds_masked.transpose("time", "lev", "y", "x")
    ds_masked = ds_masked.assign_coords({"lon_b": lon_b, "lat_b": lat_b})
    return ds_masked


@pytest.fixture
def raw_prediction_data(input_data):
    coords = {co: input_data.coords[co] for co in ["time", "y", "x"]}
    return xr.DataArray(
        np.random.random([3, 180, 360, 77]),
        dims=["time", "y", "x", "var"],
        coords=coords,
    ).to_dataset(name="__xarray_dataarray_variable__")


@pytest.fixture
def prediction_data(input_data):
    return input_data[["so", "thetao", "uo", "vo", "zos"]].drop_vars("wetmask")
