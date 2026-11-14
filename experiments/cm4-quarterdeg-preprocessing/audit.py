# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Read every physical chunk; independently audit full-record normalization."""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster
from run import FINAL, PILOT, ROOT, filesystem, record

FIELDS = ("thetao", "so", "uo", "vo", "zos", "hfds", "tauuo", "tauvo", "wfo")


def main() -> None:
    pilot = sys.argv[1] == "pilot"
    out = PILOT if pilot else FINAL
    steps = 10 if pilot else 14600
    fs = filesystem("cm4-publish")
    path = out.removeprefix("s3://")
    ds = xr.open_zarr(out, storage_options={"profile": "cm4-publish"}, chunks={})
    assert dict(ds.sizes) == {
        "time": steps,
        "lev": 19,
        "ilev": 20,
        "y": 720,
        "x": 1440,
        "y_b": 721,
        "x_b": 1441,
        "nv": 2,
    }, dict(ds.sizes)
    assert set(ds.data_vars) == set(FIELDS), list(ds.data_vars)
    times = ds.time.values
    assert times[0].calendar == "noleap"
    assert str(times[0]) == "0151-01-06 00:00:00"
    assert str(times[-1]) == ("0151-02-20 00:00:00" if pilot else "0351-01-01 00:00:00")
    bounds = ds.time_bnds.values
    np.testing.assert_array_equal(bounds[:, 1], times)
    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])
    assert all(v == datetime.timedelta(days=5) for v in bounds[:, 1] - bounds[:, 0])
    for name in ("lev", "ilev", "dz"):
        assert ds[name].attrs["units"] == "meters", (name, ds[name].attrs)
        assert ds[name].attrs["long_name"], (name, ds[name].attrs)
    assert ds.areacello.attrs["units"] == "m2"
    for name, standard in [
        ("uo", "eastward_sea_water_velocity"),
        ("vo", "northward_sea_water_velocity"),
        ("tauuo", "surface_downward_eastward_stress"),
        ("tauvo", "surface_downward_northward_stress"),
    ]:
        assert ds[name].attrs["standard_name"] == standard, (name, ds[name].attrs)
        assert ds[name].attrs["long_name"]

    # Listing checks every expected chunk, so missing objects cannot masquerade
    # as intentionally missing ocean data via Zarr's fill value.
    inventory = fs.find(path, detail=True)
    codecs = {}
    for name in ds.variables:
        meta = json.loads(fs.cat(f"{path}/{name}/.zarray"))
        if meta["shape"]:
            assert meta["compressor"] == {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
            }, (name, meta)
        codecs[name] = meta["compressor"]
        expected = math.prod(
            math.ceil(size / chunk)
            for size, chunk in zip(meta["shape"], meta["chunks"])
        )
        keys = [
            k
            for k in inventory
            if k.startswith(f"{path}/{name}/")
            and not k.rsplit("/", 1)[1].startswith(".")
        ]
        assert len(keys) == expected, (name, len(keys), expected)
    for name in FIELDS:
        expected = (1, 19, 720, 1440) if name in FIELDS[:4] else (1, 720, 1440)
        assert ds[name].encoding["chunks"] == expected
        assert ds[name].dtype == np.float32

    means = xr.open_zarr(
        out[:-5] + "_means.zarr", storage_options={"profile": "cm4-publish"}
    ).load()
    stds = xr.open_zarr(
        out[:-5] + "_stds.zarr", storage_options={"profile": "cm4-publish"}
    ).load()
    for norm in (means, stds):
        assert set(norm.data_vars) == set(FIELDS)
        assert norm.attrs["normalization_steps"] == steps
        assert sum(v.size for v in norm.data_vars.values()) == 81
        for name, field in norm.variables.items():
            if field.ndim:
                codec = field.encoding["compressor"]
                assert (
                    codec.cname == "zstd" and codec.shuffle == 1 and codec.clevel == 5
                )

    stats = {}
    start = time.time()
    with (
        LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="12GB") as cluster,
        cluster.get_client() as client,
    ):
        for name in FIELDS:
            field = ds[name]
            mask = ds.wetmask if "lev" in field.dims else ds.wetmask.isel(lev=0)
            wet = mask.values.astype(bool)
            values = field.data.astype(np.float64)
            finite = da.isfinite(values)
            valid = finite & wet
            safe = da.where(valid, values, 0.0)
            axes = tuple(i for i, dim in enumerate(field.dims) if dim != "lev")
            tasks = [
                valid.sum(axis=axes),
                safe.sum(axis=axes),
                (safe * safe).sum(axis=axes),
                da.where(valid, values, np.inf).min(axis=axes),
                da.where(valid, values, -np.inf).max(axis=axes),
                (wet & ~finite).sum(),
                (~wet & finite).sum(),
            ]
            n, total, squares, lo, hi, invalid_wet, finite_dry = client.compute(
                dask.delayed(tuple)(tasks), retries=5, sync=True
            )
            assert invalid_wet == 0 and finite_dry == 0, (name, invalid_wet, finite_dry)
            mu = total / n
            sd = np.sqrt(np.maximum(0, squares / n - mu * mu))
            assert np.all(np.isfinite(mu)) and np.all(sd > 0), name
            np.testing.assert_allclose(
                means[name].values,
                mu,
                rtol=2e-5,
                atol=float(np.max(sd)) * 2e-6 + 1e-12,
                err_msg=name,
            )
            np.testing.assert_allclose(
                stds[name].values, sd, rtol=2e-5, atol=1e-12, err_msg=name
            )
            stats[name] = {
                k: np.asarray(v).tolist()
                for k, v in [
                    ("count", n),
                    ("mean_float64", mu),
                    ("std_float64", sd),
                    ("minimum", lo),
                    ("maximum", hi),
                ]
            }
            print("AUDITED", name, stats[name], flush=True)

    result = {
        "source": out,
        "steps": steps,
        "shape": dict(ds.sizes),
        "logical_bytes": ds.nbytes,
        "stored_bytes": sum(v["size"] for v in inventory.values()),
        "objects": len(inventory),
        "readback_seconds": time.time() - start,
        "codecs": codecs,
        "statistics": stats,
        "attrs": ds.attrs,
        "job_id": os.environ["SLURM_JOB_ID"],
    }
    record("pilot-audit" if pilot else "audit", result)
    completion = json.dumps(
        {
            "validated": True,
            "source": out,
            "steps": steps,
            "audit_record": str(
                ROOT / "records" / ("pilot-audit.json" if pilot else "audit.json")
            ),
            "job_id": os.environ["SLURM_JOB_ID"],
            "completed_unix": time.time(),
        }
    ).encode()
    fs.pipe(path.rsplit("/", 1)[0] + "/COMPLETE.json", completion)
    print("VALIDATED", out, flush=True)


if __name__ == "__main__":
    main()
