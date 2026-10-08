# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Stream paired early OM4 examples; coarse wet-area means use actual bounds."""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import torch
import zarr
from torch import nn

from samudra.experiments.observation_pilot import digest
from samudra.experiments.surface_state import geographic_features


class LocalReadCache(zarr.storage.DirectoryStore):
    """Job-local copies of immutable source chunks; verify every copied byte."""

    def __init__(self, source, cache):
        super().__init__(str(source))
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        target = self.cache / key
        if target.is_file():
            return target.read_bytes()
        value = super().__getitem__(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Duplicate reads may race, but readers never see a partial write.
        import tempfile

        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(value)
        try:
            if temporary.read_bytes() != value:
                raise OSError("Local OM4 cache failed byte-for-byte readback")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return value

    def __contains__(self, key):
        return (self.cache / key).is_file() or super().__contains__(key)


class SamplePrefetch:
    """Bounded CPU lookahead keyed by explicit seeds, independent of model RNG."""

    def __init__(self, read, limit=16):
        self.read = read
        self.limit = limit
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = {}

    def plan(self, seeds):
        seeds = list(dict.fromkeys(seeds))
        if len(seeds) > self.limit:
            raise ValueError("Prefetch plan exceeds its memory bound")
        # Replay qualification can move backwards; stale work is not state.
        for seed in list(self.pending):
            if seed not in seeds:
                self.pending.pop(seed).cancel()
        for seed in seeds:
            if seed not in self.pending:
                self.pending[seed] = self.pool.submit(self.read, seed)

    def take(self, seed):
        future = self.pending.pop(seed, None)
        return self.read(seed) if future is None else future.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def load_grid_bounds(store, path, expected_sha256=None):
    """Use published corner geometry only when its centers match the data store."""
    if expected_sha256 is not None and digest(path) != expected_sha256:
        raise ValueError("Published grid geometry differs from qualification")
    with np.load(path) as archive:
        grid = {k: archive[k] for k in ["lat_b", "lon_b", "y", "x"]}
    for center, corner, endpoints in [
        ("y", "lat_b", (-90, 90)),
        ("x", "lon_b", (0, 360)),
    ]:
        np.testing.assert_allclose(grid[center], store[center][:], rtol=0, atol=1e-10)
        b = grid[corner]
        if b.ndim != 1 or len(b) != len(grid[center]) + 1 or not np.all(np.diff(b) > 0):
            raise ValueError("Malformed published grid bounds")
        np.testing.assert_allclose(b[[0, -1]], endpoints, rtol=0, atol=1e-10)
        if not np.all((grid[center] > b[:-1]) & (grid[center] < b[1:])):
            raise ValueError("Grid centers lie outside published bounds")
    return grid


def rectilinear_edges(grid):
    lat, lon = np.asarray(grid["lat_b"]), np.asarray(grid["lon_b"])
    if lat.ndim == lon.ndim == 2:
        if not np.allclose(lat, lat[:, :1]) or not np.allclose(lon, lon[:1, :]):
            raise ValueError("Expected rectilinear bounds")
        lat, lon = lat[:, 0], lon[0, :]
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected one-dimensional cell edges")
    return lat, lon


def overlap(source, target):
    value = np.maximum(
        0,
        np.minimum(target[1:, None], source[None, 1:])
        - np.maximum(target[:-1, None], source[None, :-1]),
    )
    if not np.allclose(value.sum(1), np.diff(target), rtol=1e-6, atol=1e-12):
        raise ValueError("Native bounds do not cover the coarse grid")
    return value.astype("f4")


class WetCoarsener(nn.Module):
    def __init__(self, fine, coarse):
        super().__init__()
        sy, sx = rectilinear_edges(fine)
        ty, tx = rectilinear_edges(coarse)
        self.register_buffer(
            "y",
            torch.from_numpy(
                overlap(np.sin(np.deg2rad(sy)), np.sin(np.deg2rad(ty)))
            ).to_sparse(),
        )
        self.register_buffer(
            "x", torch.from_numpy(overlap(np.deg2rad(sx), np.deg2rad(tx))).to_sparse()
        )

    def integrate(self, value):
        shape = value.shape
        n = math.prod(shape[:-2])
        h, w = shape[-2:]
        # Sparse separable area overlap; no assumption of exactly nested latitudes.
        with torch.autocast(value.device.type, enabled=False):
            v = value.float().reshape(n * h, w)
            v = torch.sparse.mm(self.x, v.T).T.reshape(n, h, self.x.shape[0])
            v = torch.sparse.mm(self.y, v.permute(1, 0, 2).reshape(h, -1))
            return (
                v.reshape(self.y.shape[0], n, self.x.shape[0])
                .permute(1, 0, 2)
                .reshape(*shape[:-2], self.y.shape[0], self.x.shape[0])
            )

    def forward(self, value, wet):
        denominator = self.integrate(wet.float())
        return self.integrate(
            torch.where(wet.bool(), value, 0)
        ) / denominator.clamp_min(1e-20)


class EarlySamples:
    def __init__(self, root, global_data, fine):
        root = Path(root)
        ready = json.loads((root / "READY.json").read_text())
        self.fine = fine
        self.names = global_data.names
        self.device = global_data.device
        self.mean, self.std = global_data.mean, global_data.std
        self.surface_ids = [self.names.index("thetao_0"), self.names.index("zos")]
        source = Path(ready["fine" if fine else "coarse"])
        if (
            digest(source / "OM4.zarr/.zmetadata")
            != ready["metadata_sha256"]["fine" if fine else "coarse"]
        ):
            raise ValueError("Early data metadata differs from qualification")
        store = str(source / "OM4.zarr")
        if os.environ.get("SLURM_TMPDIR"):
            store = LocalReadCache(
                store,
                Path(os.environ["SLURM_TMPDIR"])
                / ("early-fine" if fine else "early-coarse"),
            )
        self.store = zarr.open_consolidated(store, mode="r")
        coarse = zarr.open_consolidated(
            str(Path(ready["coarse"]) / "OM4.zarr"), mode="r"
        )
        times = self.store["time"][:]
        attrs = dict(self.store["time"].attrs)
        self.dates = cftime.num2date(times, attrs["units"], attrs["calendar"])
        np.testing.assert_array_equal(times, coarse["time"][:])
        self.origins = [
            i
            for i in range(len(times) - 24)
            if self.dates[i].year >= 1958 and self.dates[i + 24].year < 1975
        ]
        if len(self.origins) != 1217:
            raise ValueError("Unexpected early-period trajectory coverage")
        masks = [
            self.store["mask_" + str(0 if n == "zos" else int(n.rsplit("_", 1)[1]))][:]
            for n in self.names
        ]
        self.mask = torch.tensor(np.stack(masks), device=self.device, dtype=torch.bool)
        self.lat = torch.tensor(
            self.store["y"][:], device=self.device, dtype=torch.float32
        )
        lon = torch.tensor(self.store["x"][:], device=self.device, dtype=torch.float32)
        self.geo = geographic_features(self.lat, lon)
        self.weights = self.mask.float() * self.lat.deg2rad().cos()[None, :, None]
        self.coarsen = None
        if fine:
            geometry = ready["grid_bounds"]
            grids = {
                key: load_grid_bounds(
                    store, geometry[key]["path"], geometry[key]["sha256"]
                )
                for key, store in [("fine", self.store), ("coarse", coarse)]
            }
            self.coarsen = WetCoarsener(grids["fine"], grids["coarse"]).to(self.device)
        # As in U-global, retain the original 1-degree forcing normalization.
        means = zarr.open_consolidated(str(Path(ready["coarse"]) / "OM4_means.zarr"))
        stds = zarr.open_consolidated(str(Path(ready["coarse"]) / "OM4_stds.zarr"))
        self.forcing_names = ["tauuo", "tauvo", "hfds"]
        self.fm = torch.tensor(
            [float(means[n][...]) for n in self.forcing_names], device=self.device
        )[None, :, None, None]
        self.fs = torch.tensor(
            [float(stds[n][...]) for n in self.forcing_names], device=self.device
        )[None, :, None, None]
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.prefetch = SamplePrefetch(self.read_cpu)

    def read_cpu(self, seed):
        start = self.origins[
            int(np.random.default_rng(seed).integers(len(self.origins)))
        ]
        requests = [(n, start + 17, start + 25) for n in self.names]
        requests += [(self.names[i], start, start + 19) for i in self.surface_ids]
        requests += [(n, start, start + 24) for n in self.forcing_names]

        def read(req):
            n, a, b = req
            return np.asarray(self.store[n][a:b], dtype="f4")

        values = list(self.pool.map(read, requests))
        return start, tuple(
            np.stack(v, axis=1) for v in [values[:77], values[77:79], values[79:]]
        )

    def sample(self, seed):
        start, arrays = self.prefetch.take(seed)
        state, surface, forcing = (
            torch.from_numpy(value).to(self.device) for value in arrays
        )
        for value, wet in [
            (state, self.mask),
            (surface, self.mask[self.surface_ids]),
            (forcing, self.mask[self.surface_ids[0]][None]),
        ]:
            if not torch.isfinite(torch.where(wet, value, 0)).all():
                raise ValueError("Nonfinite early OM4 field on wet support")
        state = torch.where(
            self.mask,
            (state - self.mean[None, :, None, None]) / self.std[None, :, None, None],
            0,
        )
        surface = torch.where(
            self.mask[self.surface_ids],
            (surface - self.mean[self.surface_ids][None, :, None, None])
            / self.std[self.surface_ids][None, :, None, None],
            0,
        )
        forcing = torch.where(
            self.mask[self.surface_ids[0]], (forcing - self.fm) / self.fs, 0
        )
        phase = 2 * math.pi * (self.dates[start + 18].dayofyr - 1) / 365.25
        season = state.new_tensor([math.sin(phase), math.cos(phase)])[
            :, None, None
        ].expand(2, *state.shape[-2:])
        return dict(
            surface=surface[None],
            past=forcing[None, :19],
            forcing=forcing[None, 18:24],
            truth=state[None, :2],
            labels=state[None, 2:],
            context=torch.cat((self.geo, season), 0)[None],
            start=start,
            first_date=str(self.dates[start]),
            last_date=str(self.dates[start + 24]),
        )
