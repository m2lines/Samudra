import xarray as xr
import numpy as np
from ocean_emulators.postprocessing import post_processor


def test_post_processor():
    ds_raw = xr.DataArray(
        np.random.random([3, 180, 360, 77]), dims=["time", "y", "x", "var"]
    ).to_dataset(name="__xarray_dataarray_variable__")
    ds = post_processor(ds_raw)
    assert set(ds.data_vars) == set(["so", "thetao", "zos", "uo", "vo"])
    assert ds.sizes == {"time": 3, "x": 360, "y": 180, "lev": 19}
