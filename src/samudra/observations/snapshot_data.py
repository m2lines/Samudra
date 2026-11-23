# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepared six-hour ERA5, sparse daily surfaces and compact native LLC reads."""

from __future__ import annotations

import hashlib
import io
import json
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import torch
import xarray as xr

from samudra.observations.archive import context_planes
from samudra.observations.model_data import normalize_physical_fields
from samudra.observations.sample_types import ForecastSample
from samudra.observations.samples import Samples
from samudra.observations.snapshot import LLC_STATE_NAMES, eligible_origins
from samudra.observations.training import Sources, digest
from samudra.utils.data import CanonicalReadRequest, XarrayCanonicalReader
from samudra.utils.device import autocast

if TYPE_CHECKING:
    from samudra.models.surface_initialized import SurfaceInitializedModel
    from samudra.observations.config import ObservationTrainConfig

SURFACE = (102, 204)


def snapshot_manifest(cfg: ObservationTrainConfig) -> dict:
    root = cfg.data.local_path("snapshots", cfg.experiment)
    package = Path(__file__).resolve().parents[1]
    return {
        "version": "llc-obs-six-hour-v1",
        "config": cfg.scientific_config(),
        "selection_metric": cfg.selection_metric,
        "preparation_sha256": digest(root / "COMPLETE.json"),
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "code_sha256": {
            str(p.relative_to(package)): digest(p)
            for p in sorted(package.rglob("*.py"))
        },
    }


class SnapshotSamples(Samples):
    """Loss constants for snapshot windows; monthly sample methods are unused."""

    def __init__(self, root: Path, device: torch.device, task) -> None:
        self.root, self.device, self.task = root, device, task
        self.proof = json.loads((root / "COMPLETE.json").read_text())
        if self.proof["version"] != "llc-obs-six-hour-v1":
            raise ValueError("Unexpected snapshot preparation version")
        for filename in ("grid.npz", "statistics.npz"):
            if digest(root / filename) != self.proof["sha256"][filename]:
                raise ValueError(f"Prepared {filename} identity differs")
        self.grid = dict(np.load(root / "grid.npz", allow_pickle=False))
        self.stats = dict(np.load(root / "statistics.npz", allow_pickle=False))
        self.grid.update(mean=self.stats["state_mean"], std=self.stats["state_std"])
        self.names = self.grid["names"].tolist()
        if tuple(self.names) != LLC_STATE_NAMES:
            raise ValueError("Snapshot state order differs from the 205 LLC fields")
        self.mask = torch.as_tensor(self.grid["wet_area"] > 0, device=device)
        # Scale for numerical conditioning; relative wet-area weights are retained.
        self.area = self.tensor(self.grid["wet_area"] / np.max(self.grid["wet_area"]))
        self.mean = self.tensor(self.grid["mean"])[None, None, :, None, None]
        self.std = self.tensor(self.grid["std"])[None, None, :, None, None]
        for key in ("state_std", "atmosphere_std"):
            if (
                not np.isfinite(self.stats[key]).all()
                or not (self.stats[key] > 0).all()
            ):
                raise ValueError(f"Invalid {key}")
        if (
            not np.isfinite(self.stats["state_mean"]).all()
            or not np.isfinite(self.stats["atmosphere_mean"]).all()
        ):
            raise ValueError("Nonfinite normalization means")

    def to(self, device: torch.device) -> Self:
        result = copy(self)
        result.device = device
        for name in ("mask", "area", "mean", "std"):
            setattr(result, name, getattr(self, name).to(device))
        return result


class SnapshotSources(Sources):
    """Source adapter for the existing global planning, prefetch and resume loop."""

    def __init__(self, cfg: ObservationTrainConfig, device: torch.device) -> None:
        assert cfg.snapshot is not None
        self.obs = SnapshotSamples(
            cfg.data.local_path("snapshots", cfg.experiment), device, cfg.snapshot
        )
        self.names = ("llc", "obs")
        self.models = {}
        self.task = cfg.snapshot
        self.root = self.obs.root
        self.proof = self.obs.proof
        self.daily = Path(self.proof["daily_root"])
        llc = cfg.data.local_path("llc", cfg.experiment)
        if digest(llc / ".zmetadata") != self.proof["llc_metadata_sha256"]:
            raise ValueError("LLC metadata identity differs from preparation")
        ds = xr.open_zarr(llc, chunks=None).swap_dims({"y": "lat", "x": "lon"})
        ds = ds.rename(
            {
                "k": "lev",
                "U": "uo",
                "V": "vo",
                "Theta": "thetao",
                "Salt": "so",
                "Eta": "zos",
            }
        )
        self.reader = XarrayCanonicalReader(
            ds, xr.Dataset(), xr.Dataset(), LLC_STATE_NAMES
        )
        np.testing.assert_allclose(ds.lat.values, self.obs.grid["lat"], atol=1e-8)
        np.testing.assert_allclose(ds.lon.values, self.obs.grid["lon"], atol=1e-8)
        self.llc_times = pd.DatetimeIndex(ds.time.values)
        self.times = pd.date_range(
            self.proof["time_start"],
            self.proof["time_end"],
            freq="6h",
            inclusive="left",
        ).to_numpy()
        self.origins = {}
        for source in self.names:
            for split in ("train", "validation"):
                start, end = map(np.datetime64, self.proof["splits"][source][split])
                # Earliest frame uses the preceding six-hour forcing interval.
                start += np.timedelta64(6, "h")
                excluded = (
                    [
                        (
                            np.datetime64(p[0]),
                            np.datetime64(p[1]) + np.timedelta64(6, "h"),
                        )
                        for p in self.proof["excluded"]
                    ]
                    if split == "train"
                    else []
                )
                index = eligible_origins(
                    self.times,
                    start,
                    end,
                    self.task,
                    observation=source == "obs",
                    excluded=excluded,
                )
                if not len(index):
                    raise ValueError(f"No complete {source}/{split} windows")
                self.origins[source, split] = index

    def fresh_origin(self, source: str, rng: np.random.Generator) -> int:
        values = self.origins[source, "train"]
        return int(values[rng.integers(len(values))])

    def layout(self, source, origin, month):
        return None

    def auxiliary(self, source, origin, month, sample):
        return sample

    @lru_cache(maxsize=24)
    def shard(self, relative: str, product: str) -> np.ndarray:
        if product == "era5":
            path = self.root / relative
            expected = self.proof["era5_shards"][relative]
        else:
            path = self.daily / relative
            expected = self.proof["surface_shards"][relative]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Shard checksum differs: {path}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            return saved["values"].astype(np.float32)

    def load(self, source: str, origin: str | int, month: int = 0) -> ForecastSample:
        if (
            month
            or source not in self.names
            or not isinstance(origin, (int, np.integer))
        ):
            raise ValueError(
                "Snapshot loading requires an integer origin and no replay"
            )
        times = self.times[int(origin)] + self.task.offsets()
        shape = (len(times), 2, *self.obs.mask.shape[-2:])
        if source == "obs":
            surface = np.full(shape, np.nan, dtype=np.float32)
            for i, stamp in enumerate(pd.DatetimeIndex(times)):
                if stamp.hour == 0:
                    for channel, product in enumerate(("sst", "adt")):
                        surface[i, channel] = self.shard(
                            f"{product}/{stamp.year}/{stamp:%Y-%m-%d}.npz", product
                        )[0]
        else:
            indices = self.llc_times.get_indexer(times)
            if (indices < 0).any():
                raise ValueError("Missing native LLC snapshot timestamp")
            surface = self.reader.read(
                CanonicalReadRequest(indices, ("thetao_0", "zos"))
            )
        validity = np.isfinite(surface) & self.obs.grid["wet_area"][
            list(SURFACE)
        ].astype(bool)
        surface = np.where(
            validity,
            (surface - self.obs.grid["mean"][list(SURFACE), None, None])
            / self.obs.grid["std"][list(SURFACE), None, None],
            0,
        )
        forcing_frames = []
        for stamp in pd.DatetimeIndex(times):
            begin = stamp - pd.Timedelta(hours=6)
            forcing_frames.append(
                self.shard(f"era5/{begin.year}/{begin:%Y-%m-%d}.npz", "era5")[
                    begin.hour // 6
                ]
            )
        atmosphere = np.stack(forcing_frames)
        if not np.isfinite(atmosphere).all():
            raise ValueError("Nonfinite prepared ERA5 forcing")
        atmosphere = (
            atmosphere - self.obs.stats["atmosphere_mean"][None, :, None, None]
        ) / self.obs.stats["atmosphere_std"][None, :, None, None]
        result: ForecastSample = dict(
            surface=self.obs.tensor(surface)[None],
            validity=torch.as_tensor(validity, device=self.obs.device)[None],
            atmosphere=self.obs.tensor(atmosphere)[None],
            contexts=self.obs.tensor(
                context_planes(self.obs.grid["lat"], self.obs.grid["lon"], times)
            )[None],
            mask=self.obs.mask,
            start_lead=0,
            end_lead=self.task.forecast_steps,
            next_start_lead=self.task.forecast_steps,
        )
        if source == "llc":
            indices = self.llc_times.get_indexer(times[27:])
            fields = normalize_physical_fields(
                self.reader.read(CanonicalReadRequest(indices, LLC_STATE_NAMES)),
                self.obs.grid["wet_area"] > 0,
                self.obs.grid["mean"],
                self.obs.grid["std"],
            )
            result["initial_fields"] = self.obs.tensor(fields[:2])[None]
            result["target_fields"] = self.obs.tensor(fields[2:])[None]
        return result

    @torch.no_grad()
    def validate(self, model: SurfaceInitializedModel, count: int) -> dict[str, float]:
        from samudra.observations.pipeline import map_tensors

        device = next(model.parameters()).device
        model.eval()
        results: dict[str, list[float]] = {}
        area = self.obs.area.to(device)
        std = self.obs.std.to(device)
        for source in self.names:
            candidates = self.origins[source, "validation"]
            chosen = candidates[
                np.unique(np.linspace(0, len(candidates) - 1, count, dtype=int))
            ]
            for origin in chosen:
                sample = map_tensors(
                    self.load(source, int(origin)), lambda t: t.to(device)
                )
                with autocast(device.type == "cuda", torch.bfloat16):
                    prediction, _ = model.forecast(
                        sample["surface"],
                        sample["atmosphere"],
                        sample["contexts"],
                        sample["mask"],
                        sample["validity"],
                        "llc" if source == "llc" else "observation",
                    )
                target = (
                    sample["target_fields"]
                    if source == "llc"
                    else sample["surface"][:, 29:]
                )
                channels = list(range(205)) if source == "llc" else list(SURFACE)
                validity = (
                    sample["mask"][channels].expand_as(target)
                    if source == "llc"
                    else sample["validity"][:, 29:]
                )
                persistence = (
                    sample["initial_fields"][:, -1:]
                    if source == "llc"
                    else sample["surface"][:, 28:29]
                )
                for lead in (4, 8, 12):
                    valid = validity[:, lead - 1].bool()
                    if source == "obs":
                        valid &= sample["validity"][:, 28].bool()
                    weights = area[channels] * valid
                    denominator = weights.sum((-2, -1))
                    if (denominator <= 0).any():
                        raise ValueError("Validation cohort has an unsupported channel")
                    for method, estimate in (
                        ("model", prediction[:, lead - 1, channels]),
                        ("persistence", persistence[:, 0]),
                    ):
                        error = (
                            torch.where(valid, estimate - target[:, lead - 1], 0)
                            .float()
                            .square()
                        )
                        mse = (error * weights).sum((-2, -1)) / denominator
                        physical = mse * std[0, 0, channels, 0, 0].square()
                        for j, channel in enumerate(channels):
                            name = model.names[channel]
                            for metric, value in (
                                ("mse_normalized", mse[0, j]),
                                ("mse_physical", physical[0, j]),
                            ):
                                key = f"snapshot_val/{source}/{method}/{lead * 6}h/{name}/{metric}"
                                results.setdefault(key, []).append(float(value))
        metrics = {key: float(np.mean(values)) for key, values in results.items()}
        for key in list(metrics):
            if key.endswith("/mse_physical"):
                metrics[key.removesuffix("mse_physical") + "rmse"] = (
                    metrics.pop(key) ** 0.5
                )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite snapshot validation metric")
        # Aggregate square errors across origins first, then RMSE per channel/lead.
        for source in self.names:
            for method in ("model", "persistence"):
                values = [
                    value**0.5
                    for key, value in metrics.items()
                    if key.startswith(f"snapshot_val/{source}/{method}/")
                    and key.endswith("/mse_normalized")
                ]
                metrics[
                    f"snapshot_val/{source}/{method}/normalized_rmse/channel_mean"
                ] = float(np.mean(values))
        metrics["snapshot_val/obs/normalized_rmse/channel_mean"] = metrics[
            "snapshot_val/obs/model/normalized_rmse/channel_mean"
        ]
        return metrics
