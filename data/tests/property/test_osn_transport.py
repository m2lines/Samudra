# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public transport check; bounded byte ranges, not an ocean payload scan."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import fsspec
import numpy as np
import pytest
import xarray as xr
from hypothesis import Phase, given, settings
from hypothesis import strategies as st
from ocean_preprocessing.virtualize_om4 import OSN, SourceArchive, refs_to_dataframe


@pytest.mark.skipif(
    os.environ.get("SAMUDRA_OSN_CHECK") != "1",
    reason="Opt-in public OSN transport integration",
)
@settings(max_examples=4, deadline=None, database=None, phases=[Phase.generate])
@given(st.integers(0, 2**40))
def test_osn_ranges_survive_parquet_transport(position):
    source = SourceArchive.discover(
        "s3://m2lines-pubs/Samudra/raw/OM4-5daily",
        "19580101.ocean_5daily.nc",
        OSN,
        True,
    )
    snapshot = source.snapshot()
    size = snapshot[0]["size"]
    start = position % (size - 64 + 1)
    store, path = source.registry.resolve(source.urls[0])
    expected = bytes(store.get_range(path, start=start, end=start + 64))
    references = {
        "version": 1,
        "refs": {
            ".zgroup": json.dumps({"zarr_format": 2}),
            ".zattrs": "{}",
            "bytes/.zarray": json.dumps(
                {
                    "zarr_format": 2,
                    "shape": [64],
                    "chunks": [64],
                    "dtype": "|u1",
                    "compressor": None,
                    "filters": None,
                    "fill_value": None,
                    "order": "C",
                }
            ),
            "bytes/.zattrs": json.dumps({"_ARRAY_DIMENSIONS": ["byte"]}),
            "bytes/0": [source.urls[0], start, 64],
        },
    }
    with TemporaryDirectory() as directory:
        output = Path(directory) / "references"
        refs_to_dataframe(references, url=str(output), record_size=3)
        mapper = fsspec.filesystem(
            "reference",
            fo=str(output),
            asynchronous=True,
            remote_protocol="s3",
            remote_options={
                "anon": True,
                "asynchronous": True,
                "endpoint_url": OSN,
                "config_kwargs": {"connect_timeout": 10, "read_timeout": 10},
            },
        ).get_mapper("")
        with xr.open_zarr(mapper, consolidated=False, chunks=None) as actual:
            np.testing.assert_array_equal(
                actual["bytes"].values, np.frombuffer(expected, dtype="u1")
            )
    assert source.snapshot() == snapshot
