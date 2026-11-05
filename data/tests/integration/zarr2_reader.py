# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute in a separate Zarr 2 environment, comparing real native NetCDF reads."""

import sys

import fsspec
import xarray as xr
import zarr

if zarr.__version__.split(".")[0] != "2":
    raise RuntimeError(f"Expected an actual Zarr 2 client, got {zarr.__version__}")

expected = []
for path in sys.argv[2:]:
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as native:
        expected.append(native.load())
mapper = fsspec.filesystem(
    "reference", fo=sys.argv[1], remote_protocol="file"
).get_mapper("")
with xr.open_zarr(
    mapper, consolidated=False, chunks=None, decode_times=False
) as actual:
    xr.testing.assert_identical(
        actual.load(),
        xr.concat(expected, dim="time", data_vars="minimal", coords="minimal"),
    )
