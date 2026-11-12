# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Build a small on-disk cache of OM4 (or synthetic) fields.

Cache layout (one .npy per split and channel so models can memory-map one
channel at a time on a small machine):

    cache/meta.json                 channels, level of each channel, split time stamps
    cache/static.npz                wet masks (L, H, W), lat (H, W), area (H, W)
    cache/<split>/<channel>.npy     float32 (T, H, W), NaN on land

Each split is a *contiguous* run of five-day states, so every interior index t
gives a (t-1, t, t+1) triple: two-state history in, next state out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

URL = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/{res}/OM4.zarr"
VARS3D = ("thetao", "so", "uo", "vo")
# Protocol v2 follows src/samudra/configs/data/om4_demo.yaml (the 2-degree preset):
# validation and inference windows are Samudra's; the ladder trains on the last
# four years of Samudra's training window (1975-01-03 to 2013-09-30).
DEFAULT_SPLITS = {
    "train": ("2009-10-01", "2013-09-30"),
    "val": ("2013-10-05", "2014-10-05"),
    "test": ("2014-10-10", "2022-12-24"),
}


def channel_list(levels) -> list[tuple[str, int]]:
    """(name, level) pairs; zos lives at level 0."""
    return [(f"{v}_{k}", k) for v in VARS3D for k in levels] + [("zos", 0)]


def _write_static(cache: Path, masks, lat, area, dz):
    np.savez(cache / "static.npz", masks=masks.astype(bool), lat=lat, area=area, dz=dz)


def load_om4(cache: Path, res: str, splits: dict, levels) -> None:
    import xarray as xr

    cache.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(URL.format(res=res))
    chans = channel_list(levels)
    masks = np.stack([ds[f"mask_{k}"].values for k in levels])
    dz = ds.dz.values[list(levels)].astype(np.float64)
    _write_static(cache, masks, ds.lat.values, ds.areacello.values, dz)
    meta: dict[str, Any] = {
        "source": URL.format(res=res),
        "attrs": {k: str(v) for k, v in ds.attrs.items() if k.startswith("m2lines/")},
        "channels": [c for c, _ in chans],
        "levels": {c: levels.index(k) for c, k in chans},
        "depth": {c: float(ds.lev.values[k]) for c, k in chans},
        "splits": {},
    }
    for split, (start, end) in splits.items():
        # Label-based and inclusive at both ends, as xarray .sel (and Samudra) use.
        sl = ds.indexes["time"].slice_indexer(start, end)
        idx = np.arange(len(ds.time))[sl]
        if len(idx) < 3:
            raise ValueError(f"split {split} {start}..{end} has {len(idx)} steps")
        (cache / split).mkdir(exist_ok=True)
        meta["splits"][split] = [str(t) for t in ds.time.values[idx]]
        t0 = time.time()
        for name, _ in chans:
            arr = (
                ds[name].isel(time=slice(idx[0], idx[-1] + 1)).values.astype(np.float32)
            )
            np.save(cache / split / f"{name}.npy", arr)
        print(
            f"[data] {split}: {len(idx)} steps x {len(chans)} channels in {time.time() - t0:.0f}s"
        )
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))


def make_synthetic(
    cache: Path, levels=(0, 5, 10), n_steps=(60, 24, 24), ny=90, nx=180, seed=0
) -> None:
    """Advected/diffused tracers with evolving smooth velocities on a lat-lon grid
    with continents. Only for exercising the pipeline; the numbers mean nothing
    about the ocean.
    """
    rng = np.random.default_rng(seed)
    cache.mkdir(parents=True, exist_ok=True)
    lat1 = np.linspace(-89, 89, ny)
    lat = np.repeat(lat1[:, None], nx, 1)
    area = np.cos(np.deg2rad(lat))
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    land = (
        ((xx - 40) ** 2 / 300 + (yy - 50) ** 2 / 400 < 1)
        | ((xx - 120) ** 2 / 200 + (yy - 35) ** 2 / 500 < 1)
        | (yy < 6)
    )

    def smooth(f, n=6):
        for _ in range(n):
            f = 0.2 * (
                f
                + np.roll(f, 1, 0)
                + np.roll(f, -1, 0)
                + np.roll(f, 1, 1)
                + np.roll(f, -1, 1)
            )
        return f

    def ddx(f):
        return 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))

    def ddy(f):
        return 0.5 * (np.roll(f, -1, 0) - np.roll(f, 1, 0))

    def lap(f):
        return (
            np.roll(f, 1, 0)
            + np.roll(f, -1, 0)
            + np.roll(f, 1, 1)
            + np.roll(f, -1, 1)
            - 4 * f
        )

    nl = len(levels)
    total = sum(n_steps) + 20
    psi = [smooth(rng.normal(size=(ny, nx)), 20) * 60 / (1 + li) for li in range(nl)]
    temp = [
        20 * np.cos(np.deg2rad(lat))
        - 4 * li
        + smooth(rng.normal(size=(ny, nx)), 10) * 8
        for li in range(nl)
    ]
    salt = [35 + smooth(rng.normal(size=(ny, nx)), 10) * 3 for _ in range(nl)]
    names = [f"{v}_{k}" for v in VARS3D for k in levels] + ["zos"]
    frames: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for _ in range(total):
        for li, k in enumerate(levels):
            psi[li] = 0.97 * psi[li] + 0.03 * smooth(
                rng.normal(size=(ny, nx)), 20
            ) * 60 / (1 + li)
            u, v = -ddy(psi[li]), ddx(psi[li])
            cfl = max(np.abs(u).max(), np.abs(v).max(), 1e-12)
            u, v = (
                u / cfl / (1 + li),
                v / cfl / (1 + li),
            )  # keep the toy advection stable
            for tracer in (temp, salt):
                f = tracer[li]
                tracer[li] = (
                    f
                    - 0.4 * (u * ddx(f) + v * ddy(f))
                    + 0.05 * lap(f)
                    + 0.02 * smooth(rng.normal(size=(ny, nx)), 3)
                )
            frames[f"thetao_{k}"].append(temp[li].copy())
            frames[f"so_{k}"].append(salt[li].copy())
            frames[f"uo_{k}"].append(u.copy())
            frames[f"vo_{k}"].append(v.copy())
        frames["zos"].append(psi[0] / 60)
    masks = np.stack([~land for _ in levels])
    _write_static(cache, masks, lat, area, np.ones(len(levels)))
    chans = channel_list(list(levels))
    meta: dict[str, Any] = {
        "source": "synthetic",
        "channels": [c for c, _ in chans],
        "levels": {c: list(levels).index(k) for c, k in chans},
        "depth": {c: float(k) for c, k in chans},
        "splits": {},
    }
    start = 20
    for split, n in zip(("train", "val", "test"), n_steps):
        (cache / split).mkdir(exist_ok=True)
        meta["splits"][split] = [f"step{t}" for t in range(start, start + n)]
        for name, _ in chans:
            arr = np.stack(frames[name][start : start + n]).astype(np.float32)
            arr[:, land] = np.nan
            np.save(cache / split / f"{name}.npy", arr)
        start += n
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cache", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--res", default="om4_twodeg")
    ap.add_argument(
        "--levels", default="all", help="comma-separated level indices, or 'all' (19)"
    )
    for split, (start, end) in DEFAULT_SPLITS.items():
        ap.add_argument(f"--{split}", default=f"{start}:{end}", help="START:END dates")
    a = ap.parse_args()
    if a.synthetic:
        make_synthetic(a.cache)
    else:
        levels = (
            list(range(19))
            if a.levels == "all"
            else [int(x) for x in a.levels.split(",")]
        )
        load_om4(
            a.cache,
            a.res,
            {s: tuple(getattr(a, s).split(":")) for s in DEFAULT_SPLITS},
            levels,
        )
