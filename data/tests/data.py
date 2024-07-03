import dask.array as dsa
import xarray as xr
import numpy as np
import pytest


@pytest.fixture
def input_data():
    y = xr.DataArray(np.arange(-89, 91, 1), dims=["y"])
    x = xr.DataArray(np.arange(0, 360, 1), dims=["x"])
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
    )
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

    coords = {
        "x": x,
        "y": y,
        "lev": lev,
        "dz": dz,
        "areacello": areacello,
        "wetmask": wetmask,
    }

    coords_2d = {k: da for k, da in coords.items() if "lev" not in da.dims}
    ds = xr.Dataset(
        {
            "so": xr.DataArray(
                dsa.random.random([360, 180, 19, 3]),
                dims=["x", "y", "lev", "time"],
                coords=coords,
            ),
            "thetao": xr.DataArray(
                dsa.random.random([360, 180, 19, 3]),
                dims=["x", "y", "lev", "time"],
                coords=coords,
            ),
            "uo": xr.DataArray(
                dsa.random.random([360, 180, 19, 3]),
                dims=["x", "y", "lev", "time"],
                coords=coords,
            ),
            "vo": xr.DataArray(
                dsa.random.random([360, 180, 19, 3]),
                dims=["x", "y", "lev", "time"],
                coords=coords,
            ),
            "zos": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
            "hfds": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
            "tauuo": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
            "tauvo": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
            "sithick": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
            "siconc": xr.DataArray(
                dsa.random.random([360, 180, 3]),
                dims=["x", "y", "time"],
                coords=coords_2d,
            ),
        },
        attrs={"something": "for now"},
    )
    return ds


@pytest.fixture
def raw_prediction():
    return xr.DataArray(
        np.random.random([3, 180, 360, 77]), dims=["time", "y", "x", "var"]
    ).to_dataset(name="__xarray_dataarray_variable__")
