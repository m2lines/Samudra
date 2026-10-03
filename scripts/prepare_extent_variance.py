#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only native surface-velocity subcell variance on actual 1-degree cells."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import cftime
import numcodecs  # type: ignore[import-untyped]
import numpy as np
import zarr  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix

from scripts.prepare_om4_patch_cache import validate_shared_time, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(source_edges, target_edges):
    source_edges, target_edges = np.asarray(source_edges), np.asarray(target_edges)
    if not np.all(np.diff(source_edges) > 0) or not np.all(np.diff(target_edges) > 0):
        raise ValueError("Bounds must be strictly increasing")
    values = np.maximum(
        0,
        np.minimum(target_edges[1:, None], source_edges[None, 1:])
        - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
    )
    if not np.allclose(values.sum(1), np.diff(target_edges), rtol=1e-12, atol=1e-14):
        raise ValueError("Source does not fully cover target bounds")
    return csr_matrix(values)


def rectilinear_bounds(store):
    lat, lon = store["lat_b"][:], store["lon_b"][:]
    if lat.ndim != 2 or lon.shape != lat.shape:
        raise ValueError("Expected full rectilinear corner mesh")
    if not np.allclose(lat, lat[:, :1], rtol=0, atol=1e-10) or not np.allclose(
        lon, lon[:1], rtol=0, atol=1e-10
    ):
        raise ValueError("Curvilinear bounds require polygon aggregation")
    return lat[:, 0], lon[0]


class VarianceAggregator:
    def __init__(self, source_lat, source_lon, target_lat, target_lon, wet):
        self.y = overlap(np.sin(np.deg2rad(source_lat)), np.sin(np.deg2rad(target_lat)))
        self.x = overlap(np.deg2rad(source_lon), np.deg2rad(target_lon))
        self.wet = np.asarray(wet, dtype=bool)
        self.denominator = self.integrate(self.wet.astype("f8"))
        self.area = (
            np.diff(np.sin(np.deg2rad(target_lat)))[:, None]
            * np.diff(np.deg2rad(target_lon))[None, :]
        )
        self.fraction = self.denominator / self.area

    def integrate(self, field):
        return self.x.dot(self.y.dot(field).T).T

    def variance(self, u, v):
        u, v = np.asarray(u, dtype="f8"), np.asarray(v, dtype="f8")
        if not np.isfinite(u[self.wet]).all() or not np.isfinite(v[self.wet]).all():
            raise ValueError("Nonfinite velocity on declared native wet support")
        u, v = np.where(self.wet, u, 0), np.where(self.wet, v, 0)
        denominator = np.maximum(self.denominator, 1e-30)
        mean_u, mean_v = (
            self.integrate(u) / denominator,
            self.integrate(v) / denominator,
        )
        second = self.integrate(u * u + v * v) / denominator
        variance = 0.5 * (second - mean_u * mean_u - mean_v * mean_v)
        if variance.min() < -1e-12:
            raise ValueError("Negative variance beyond floating-point roundoff")
        return np.maximum(variance, 0).astype("f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--global-om4", required=True)
    parser.add_argument("--transfer-success", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    numcodecs.blosc.set_nthreads(1)
    source = Path(args.source)
    baseline = Path(args.global_om4)
    out = Path(args.output)
    receipt = json.loads(Path(args.transfer_success).read_text())
    if receipt.get("destination") != str(
        source
    ) or "Full rclone check --download" not in receipt.get("validation", ""):
        raise ValueError("Native source is not covered by transfer verification")
    native = zarr.open_consolidated(str(source / "OM4.zarr"))
    coarse = zarr.open_consolidated(str(baseline / "OM4.zarr"))
    if float(native["lev"][0]) != 2.5:
        raise ValueError("Unexpected source surface velocity depth")
    times = native["time"][:]
    attrs = dict(native["time"].attrs)
    validate_shared_time(times, attrs, coarse["time"][:], dict(coarse["time"].attrs))
    dates = cftime.num2date(times, attrs["units"], attrs["calendar"])
    stamps = np.array([str(x)[:10] for x in dates])
    indices = np.flatnonzero((stamps >= "1975-01-03") & (stamps <= "2013-10-04"))
    sy, sx = rectilinear_bounds(native)
    ty, tx = rectilinear_bounds(coarse)
    aggregate = VarianceAggregator(sy, sx, ty, tx, native["mask_0"][:].astype(bool))
    valid = coarse["mask_0"][:].astype(bool) & (aggregate.fraction >= 0.5)
    if valid.sum() < 1000:
        raise ValueError("Insufficient target support")
    contract = dict(
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        source=str(source),
        global_om4=str(baseline),
        source_metadata_sha256=digest(source / "OM4.zarr/.zmetadata"),
        global_metadata_sha256=digest(baseline / "OM4.zarr/.zmetadata"),
        transfer_receipt_sha256=digest(args.transfer_success),
        fields=["uo_0", "vo_0"],
        depth_m=2.5,
        first=str(stamps[indices[0]]),
        last=str(stamps[indices[-1]]),
        frames=len(indices),
        source_indices=indices.tolist(),
        shape=list(valid.shape),
        target="0.5*(area_mean(u^2+v^2)-area_mean(u)^2-area_mean(v)^2)",
        time_policy="Variance of five-day-mean velocities, not total unresolved EKE; exact shared timestamps",
        geometry="Spherical rectangle intersections using stored lat_b/lon_b; common u/v native mask",
        minimum_native_wet_fraction=0.5,
        normalization="Training-only log1p(var/area_mean_var), then area-weighted mean/std",
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != contract:
        raise ValueError("Auxiliary cache resume contract differs")
    write_json(out / "manifest.json", contract)
    if (out / "READY.json").exists():
        ready = json.loads((out / "READY.json").read_text())
        for key, name in (
            ("manifest_sha256", "manifest.json"),
            ("grid_sha256", "grid.npz"),
            ("normalization_sha256", "normalization.json"),
        ):
            if ready[key] != digest(out / name):
                raise ValueError("Auxiliary cache changed")
        return
    np.savez(
        out / "grid.npz",
        mask=valid,
        area=aggregate.area,
        native_wet_fraction=aggregate.fraction,
        lat=coarse["y"][:],
        lon=coarse["x"][:],
        dates=stamps[indices],
        time=times[indices],
    )
    store = zarr.open_group(str(out / "targets.zarr"), mode="a")
    values = store.require_dataset(
        "variance",
        shape=(len(indices), *valid.shape),
        chunks=(1, *valid.shape),
        dtype="f4",
        fill_value=np.nan,
        compressor=numcodecs.Blosc(cname="lz4", clevel=3),
    )
    verified = store.require_dataset(
        "verified", shape=(len(indices),), chunks=(1,), dtype="bool", fill_value=False
    )

    def frame(i):
        if verified[i]:
            return
        j = int(indices[i])
        result = aggregate.variance(native["uo_0"][j], native["vo_0"][j])
        result = np.where(valid, result, np.nan).astype("f4")
        values[i] = result
        if not np.array_equal(values[i], result, equal_nan=True):
            raise ValueError("Auxiliary target readback differs")
        verified[i] = True

    todo = [i for i in range(len(indices)) if not verified[i]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, _ in enumerate(pool.map(frame, todo), 1):
            if n % 100 == 0 or n == len(todo):
                print(
                    json.dumps(
                        dict(
                            event="auxiliary_frames_verified",
                            new_frames=n,
                            total=len(indices),
                        )
                    ),
                    flush=True,
                )
    weights = aggregate.area * valid
    denominator = weights.sum() * len(indices)
    total = sum(
        float((np.nan_to_num(values[i]) * weights).sum()) for i in range(len(indices))
    )
    scale = total / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid auxiliary variance scale")
    s1 = s2 = 0.0
    for i in range(len(indices)):
        transformed = np.log1p(np.nan_to_num(values[i]).astype("f8") / scale)
        s1 += float((transformed * weights).sum())
        s2 += float((transformed**2 * weights).sum())
    mean = s1 / denominator
    std = np.sqrt(s2 / denominator - mean * mean)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid auxiliary transform scale")
    write_json(
        out / "normalization.json",
        dict(
            variance_scale=scale,
            log_mean=mean,
            log_std=float(std),
            training_frames=len(indices),
            valid_cells=int(valid.sum()),
        ),
    )
    zarr.consolidate_metadata(store.store)
    assert verified[:].all()
    write_json(
        out / "READY.json",
        dict(
            manifest_sha256=digest(out / "manifest.json"),
            grid_sha256=digest(out / "grid.npz"),
            normalization_sha256=digest(out / "normalization.json"),
            frames=len(indices),
            verification="All targets read back exactly after computation; all wet native velocities finite",
        ),
    )


if __name__ == "__main__":
    main()
