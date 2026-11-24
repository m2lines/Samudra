"""Guard the scientific units/masks passed to the standard metric kernels."""

from types import SimpleNamespace

import numpy as np
from run_standard_suite import dataset


def test_physical_units_depth_mapping_and_missing_land():
    prediction = np.full((2, 77, 8, 16), 2, dtype=np.float32)
    mask = np.ones((77, 8, 16), dtype=bool)
    mask[:, 0] = False
    mask[51, 2, 4] = False
    data = SimpleNamespace(
        mask=mask,
        mean=np.arange(77, dtype=np.float32),
        std=np.full(77, 3, dtype=np.float32),
        meta={
            "latitude": np.linspace(-80, 80, 8),
            "longitude": np.linspace(0, 350, 16),
        },
    )
    ds = dataset(prediction, data, ["2015-01-01", "2015-01-06"], "fixture")
    assert ds.thetao.sel(lev=1850).isel(time=0, lat=3, lon=4).compute() == 57
    assert np.isnan(ds.thetao.sel(lev=1850).isel(time=0, lat=2, lon=4).compute())
    assert ds.so.isel(time=0, lev=0, lat=3, lon=4).compute() == 63
    assert ds.zos.isel(time=0, lat=3, lon=4).compute() == 82
    assert np.isnan(ds.zos.isel(time=0, lat=0).compute()).all()
    assert ds.dz.sum() == 6500
    assert ds.areacello.min() > 0
