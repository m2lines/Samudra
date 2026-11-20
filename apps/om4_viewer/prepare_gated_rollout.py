# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and fully verify the native gated EMA rollout for this viewer."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root", type=Path, required=True, help="Directory containing predictions.zarr"
)
parser.add_argument("--truth", type=Path, required=True)
args = parser.parse_args()
root = args.root
if (root / "viewer.zarr").exists():
    raise FileExistsError(root / "viewer.zarr")
s = xr.open_zarr(root / "predictions.zarr", chunks=None)
t = xr.open_zarr(args.truth, chunks=None)
np.testing.assert_array_equal(s.lon.values, np.broadcast_to(t.x.values, s.lon.shape))
np.testing.assert_array_equal(
    s.lat.values, np.broadcast_to(t.y.values[:, None], s.lat.shape)
)
assert all(x in t.time.values for x in s.time.values)
coords = {"time": s.time, "y": t.y, "x": t.x}
d = xr.Dataset(
    coords=coords,
    attrs={
        "viewer_name": "Samudra 1° · native gated · final EMA",
        "title": "Native persistence gate, seed 15, epoch 70 EMA; continuous rollout",
        "source_link": "https://wandb.ai/ocean_emulators/tocean/runs/q5mz68j4",
        "source_commit": "6a19fc3c63d56eaeaf756eb734b2a928b1782e97",
        "prediction_origin": "2014-10-15 12:00:00",
        "preparation": "Physical saved fields; depth slices and masks from matched OM4 grid; no regridding.",
    },
)
for k in t:
    if k.startswith("mask_"):
        d[k] = t[k]
for c in d.variables:
    d[c].encoding = {}
d.time.encoding = {
    "units": s.time.encoding["units"],
    "calendar": s.time.encoding["calendar"],
}
d.to_zarr(root / "viewer.zarr", mode="w", consolidated=False)
for field in ("thetao", "so", "uo", "vo", "zos"):
    field_values = np.asarray(s[field], dtype="f4")
    levels = range(s.sizes["lev"]) if "lev" in s[field].dims else (None,)
    for level in levels:
        key = f"{field}_{level}" if level is not None else field
        values = field_values[:, level] if level is not None else field_values
        assert np.isfinite(values).all(), key
        a = xr.Dataset({key: (("time", "y", "x"), values)}, coords=coords)
        a[key].attrs["units"] = t[key].attrs.get("units") or t.attrs.get(field, {}).get(
            "units", ""
        )
        for c in a.variables:
            a[c].encoding = {}
        a.time.encoding = d.time.encoding.copy()
        a.to_zarr(
            root / "viewer.zarr",
            mode="a",
            consolidated=False,
            encoding={
                key: {
                    "chunks": (1, 180, 360),
                    "compressor": Blosc(
                        cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE
                    ),
                }
            },
        )
        print(key, flush=True)
# Appending xarray datasets replaces root attributes; restore provenance last.
zarr.open_group(str(root / "viewer.zarr"), mode="a").attrs.update(d.attrs)
zarr.consolidate_metadata(str(root / "viewer.zarr"))
r = xr.open_zarr(root / "viewer.zarr", chunks=None)
for field in ("thetao", "so", "uo", "vo", "zos"):
    field_values = np.asarray(s[field], dtype="f4")
    for level in range(s.sizes["lev"]) if "lev" in s[field].dims else (None,):
        key = f"{field}_{level}" if level is not None else field
        np.testing.assert_array_equal(
            r[key].values, field_values[:, level] if level is not None else field_values
        )
print(
    "FULL READBACK MATCH; frames",
    s.sizes["time"],
    str(s.time.values[0]),
    str(s.time.values[-1]),
    flush=True,
)
