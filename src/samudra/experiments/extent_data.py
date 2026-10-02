# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-resolution patch samples with strictly training-period trajectories."""

import json
import math
from pathlib import Path

import cftime
import numpy as np
import torch
import zarr  # type: ignore[import-untyped]

from samudra.experiments.observation_pilot import digest
from samudra.experiments.surface_state import geographic_features


class PatchSamples:
    size, halo, leads = 128, 32, 6

    def __init__(self, root, global_data):
        root = Path(root)
        ready = json.loads((root / "CACHE_READY.json").read_text())
        if ready["manifest_sha256"] != digest(root / "manifest.json") or ready[
            "grid_sha256"
        ] != digest(root / "grid.npz"):
            raise ValueError("Patch cache contract changed")
        self.fields = zarr.open_consolidated(str(root / "fields.zarr"), mode="r")
        self.grid = dict(np.load(root / "grid.npz"))
        if self.grid["names"].tolist() != global_data.names:
            raise ValueError("Patch/global physical channel order differs")
        if not self.fields["verified"][:].all():
            raise ValueError("Unverified cache frames")
        self.device = global_data.device
        self.mean = global_data.mean[None, :, None, None]
        self.std = global_data.std[None, :, None, None]
        # Global OM4 normalization is retained for forcing as well as state.
        source = Path(global_data.args.data_root)
        names = ("tauuo", "tauvo", "hfds")
        self.forcing_mean = torch.tensor(
            [
                float(zarr.open_consolidated(str(source / "OM4_means.zarr"))[n][...])
                for n in names
            ],
            device=self.device,
        )[None, :, None, None]
        self.forcing_std = torch.tensor(
            [
                float(zarr.open_consolidated(str(source / "OM4_stds.zarr"))[n][...])
                for n in names
            ],
            device=self.device,
        )[None, :, None, None]
        self.names = global_data.names
        self.surface_ids = [self.names.index("thetao_0"), self.names.index("zos")]
        self.dates = cftime.num2date(
            self.grid["time"], str(self.grid["time_units"]), str(self.grid["calendar"])
        )
        # 19 input frames, last input at t+18, labels t+19..t+24.
        self.origins = [
            i
            for i in range(len(self.dates) - 24)
            if str(self.grid["dates"][i + 24]) <= "2013-10-04"
        ]
        if not self.origins:
            raise ValueError("No training patch trajectories")

    def sample(self, seed):
        rng = np.random.default_rng(seed)
        start = self.origins[int(rng.integers(len(self.origins)))]
        h, w = self.grid["mask"].shape[-2:]
        # Longitude wrapping is an extraction operation, not a regional boundary condition.
        for _ in range(1000):
            y = int(rng.integers(h - self.size + 1))
            x = int(rng.integers(w))
            columns = (np.arange(self.size) + x) % w
            mask_np = self.grid["mask"][:, y : y + self.size][:, :, columns]
            core = mask_np[
                self.surface_ids[0], self.halo : -self.halo, self.halo : -self.halo
            ]
            if core.mean() >= 0.2:
                break
        else:
            raise ValueError("Could not sample a wet patch core")

        def read(key, interval):
            # Orthogonal indexing reads only intersecting native spatial chunks.
            values = self.fields[key].oindex[
                interval, :, slice(y, y + self.size), columns
            ]
            return torch.as_tensor(values, device=self.device)

        mask = torch.as_tensor(mask_np, device=self.device)
        surface = read("surface", slice(start, start + 19))
        surface = (surface - self.mean[:, self.surface_ids]) / self.std[
            :, self.surface_ids
        ]
        surface = torch.where(mask[self.surface_ids], surface, 0)
        truth = read("prognostic", slice(start + 17, start + 25))
        truth = torch.where(mask, (truth - self.mean) / self.std, 0)
        forcing = read("boundary", slice(start, start + 24))
        forcing = torch.where(
            mask[self.surface_ids[0]],
            (forcing - self.forcing_mean) / self.forcing_std,
            0,
        )
        lat = torch.as_tensor(
            self.grid["lat"][y : y + self.size], device=self.device, dtype=torch.float32
        )
        lon = torch.as_tensor(
            self.grid["lon"][columns], device=self.device, dtype=torch.float32
        )
        geo = geographic_features(lat, lon)[None]
        phase = 2 * math.pi * (self.dates[start + 18].dayofyr - 1) / 365.25
        season = geo.new_tensor([math.sin(phase), math.cos(phase)])[
            None, :, None, None
        ].expand(1, 2, self.size, self.size)
        context = torch.cat((geo, season), 1)
        weights = mask.float() * torch.deg2rad(lat).cos()[None, :, None]
        core_mask = torch.zeros_like(mask)
        core_mask[:, self.halo : -self.halo, self.halo : -self.halo] = True
        weights = weights * core_mask
        for tensor in (surface, truth, forcing):
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite patch tensor after applying declared masks")
        return dict(
            surface=surface[None],
            past=forcing[None, :19],
            context=context,
            truth=truth[None, :2],
            labels=truth[None, 2:],
            forcing=forcing[None, 18:24],
            mask=mask,
            weights=weights,
            lat=lat,
            core=core_mask,
            provenance=dict(first=str(self.grid["dates"][start]), y=y, x=x),
        )
