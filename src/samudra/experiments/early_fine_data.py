# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Stream paired early OM4 examples; coarse wet-area means use actual bounds."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import torch
import zarr
from torch import nn

from samudra.experiments.observation_pilot import digest
from samudra.experiments.surface_state import geographic_features


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
        sy, sx = fine["lat_b"][:, 0], fine["lon_b"][0, :]
        ty, tx = coarse["lat_b"][:, 0], coarse["lon_b"][0, :]
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
        self.store = zarr.open_consolidated(str(source / "OM4.zarr"), mode="r")
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
        self.coarsen = (
            WetCoarsener(self.store, coarse).to(self.device) if fine else None
        )
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
        self.pool = ThreadPoolExecutor(max_workers=8)

    def sample(self, seed):
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
        state = torch.from_numpy(np.stack(values[:77], axis=1)).to(self.device)
        surface = torch.from_numpy(np.stack(values[77:79], axis=1)).to(self.device)
        forcing = torch.from_numpy(np.stack(values[79:], axis=1)).to(self.device)
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
