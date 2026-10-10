# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Extend the audited target cache to this baseline's exact training cutoff."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cftime
import numpy as np
import zarr  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--coarse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parent, out = args.parent, args.output
    ready = json.loads((parent / "READY.json").read_text())
    for key, name in [
        ("manifest", "manifest.json"),
        ("grid", "grid.npz"),
        ("normalization", "normalization.json"),
    ]:
        if digest(parent / name) != ready[key + "_sha256"]:
            raise ValueError("Parent metadata differs")
    manifest = json.loads((parent / "manifest.json").read_text())
    old = zarr.open_consolidated(str(parent / "targets.zarr"))
    if not old["verified"][:].all():
        raise ValueError("Unverified parent cache")
    native = zarr.open_consolidated(str(Path(manifest["source"]) / "OM4.zarr"))
    coarse = zarr.open_consolidated(str(args.coarse / "OM4.zarr"))
    original = zarr.open_consolidated(str(Path(manifest["global_om4"]) / "OM4.zarr"))
    for name in ["lat_b", "lon_b", "x", "y", "mask_0"]:
        np.testing.assert_array_equal(original[name][:], coarse[name][:])
    grid = dict(np.load(parent / "grid.npz", allow_pickle=False))
    attrs = dict(native["time"].attrs)
    stamps = np.array(
        [
            str(t)[:10]
            for t in cftime.num2date(
                native["time"][:], attrs["units"], attrs["calendar"]
            )
        ]
    )
    indices = np.flatnonzero((stamps >= "1975-01-03") & (stamps <= "2013-10-05"))
    np.testing.assert_array_equal(stamps[indices[:-1]], grid["dates"])
    if stamps[indices[-1]] != "2013-10-05":
        raise ValueError("Unexpected missing frame")
    sy, sx = rectilinear_bounds(native)
    ty, tx = rectilinear_bounds(coarse)
    aggregate = VarianceAggregator(sy, sx, ty, tx, native["mask_0"][:].astype(bool))
    np.testing.assert_allclose(aggregate.area, grid["area"], rtol=0, atol=0)
    np.testing.assert_array_equal(
        grid["mask"], coarse["mask_0"][:].astype(bool) & (aggregate.fraction >= 0.5)
    )
    expected = aggregate.variance(
        native["uo_0"][indices[-1]], native["vo_0"][indices[-1]]
    )
    expected = np.where(grid["mask"], expected, np.nan)
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(parent, out)
    (out / "READY.json").unlink()
    store = zarr.open_group(str(out / "targets.zarr"), mode="a")
    store["variance"].resize((len(indices), *grid["mask"].shape))
    store["verified"].resize((len(indices),))
    store["variance"][-1] = expected
    np.testing.assert_array_equal(store["variance"][:-1], old["variance"][:])
    np.testing.assert_array_equal(store["variance"][-1], expected)
    store["verified"][-1] = True
    grid["dates"] = stamps[indices]
    grid["time"] = native["time"][indices]
    np.savez(out / "grid.npz", **grid)
    # Preserve the original training-only normalization for a direct target transfer.
    manifest.update(
        parent_manifest_sha256=ready["manifest_sha256"],
        parent_root=str(parent),
        frames=len(indices),
        last="2013-10-05",
        source_indices=indices.tolist(),
        global_om4=str(args.coarse),
        normalization="Frozen original training-only transform through 2013-09-30; appended training date uses same transform",
        extension_script_sha256=digest(__file__),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    zarr.consolidate_metadata(str(out / "targets.zarr"))
    ready = {
        key + "_sha256": digest(out / name)
        for key, name in [
            ("manifest", "manifest.json"),
            ("grid", "grid.npz"),
            ("normalization", "normalization.json"),
        ]
    }
    ready.update(
        frames=len(indices),
        verification="All copied targets read back equal to verified parent; appended native-derived frame read back exactly",
    )
    (out / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(ready), flush=True)


if __name__ == "__main__":
    main()
