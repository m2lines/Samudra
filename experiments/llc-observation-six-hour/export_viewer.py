# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export frozen EMA checkpoints and references for the LLC results page.

Run with torchrun over four GPUs against the unchanged training package. Each
rank owns independent origins; no training or inter-rank gradient work occurs.
"""

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from samudra.observations.archive import context_planes
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.snapshot import LLC_STATE_NAMES
from samudra.observations.snapshot_data import (
    SURFACE,
    SnapshotSources,
    snapshot_manifest,
)
from samudra.utils.data import CanonicalReadRequest
from samudra.utils.device import autocast
from samudra.utils.ema import EMATracker

SAVED_HOURS = (6, 12, 24, 48, 72, 120, 240, 360, 480, 600, 720)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def context_times(available, wanted):
    """Exact matches first; missing fields use one explicit calendar-year offset."""
    available, wanted = pd.DatetimeIndex(available), pd.DatetimeIndex(wanted)
    exact = available.get_indexer(wanted) >= 0
    actual = wanted.to_numpy().copy()
    missing = wanted[~exact]
    if len(missing):
        offsets = sorted(
            {year - int(missing[0].year) for year in available.year.unique()},
            key=lambda n: (abs(n), n),
        )
        for offset in offsets:
            try:
                candidates = pd.DatetimeIndex(
                    [stamp.replace(year=stamp.year + offset) for stamp in missing]
                )
            except ValueError:
                continue
            if (available.get_indexer(candidates) >= 0).all():
                actual[~exact] = candidates.to_numpy()
                break
        else:
            raise ValueError("No available calendar-date-matched LLC context")
    return pd.DatetimeIndex(actual), exact


def physical(states, mean, std, wet):
    result = states.detach().float().cpu().numpy() * std + mean
    return np.where(wet, result, np.nan).astype("f4")


def score(prediction, target, support, area, std):
    """One origin's spatial MSE, retained for the training-validation reduction."""
    valid = support & np.isfinite(target) & (area > 0)
    if not np.isfinite(prediction[valid]).all():
        raise FloatingPointError("Nonfinite prediction on scored support")
    weight = np.where(valid, area, 0).astype("f8")
    denominator = weight.sum((-2, -1))
    if (denominator <= 0).any():
        raise ValueError("Scored channel has no valid cells")
    error = np.where(valid, prediction - target, 0).astype("f8")
    mse = (error**2 * weight).sum((-2, -1)) / denominator
    return dict(
        mse_physical=mse.tolist(),
        mse_normalized=(mse / std**2).tolist(),
        cells=valid.sum((-2, -1)).tolist(),
    )


class Export:
    def __init__(self, cfg, root, extra, output, hours):
        self.cfg, self.root, self.extra, self.output = cfg, root, extra, output
        self.sources = SnapshotSources(cfg, torch.device("cpu"))
        self.data = self.sources.obs
        self.hours = hours
        self.proof = self.sources.proof
        self.extra_proof = json.loads((extra / "COMPLETE.json").read_text())
        if self.extra_proof["signature"]["training_proof_sha256"] != sha(
            self.data.root / "COMPLETE.json"
        ):
            raise ValueError("Evaluation extension belongs to another training cache")
        self.wet = self.data.grid["wet_area"] > 0
        self.mean = self.data.grid["mean"][:, None, None]
        self.std = self.data.grid["std"][:, None, None]
        self.area = self.data.grid["wet_area"].astype("f8")
        self.files = {}
        self.extra_reads = {}

    @lru_cache(maxsize=96)
    def shard(self, relative, product):
        key = "era5_shards" if product == "era5" else "surface_shards"
        if relative in self.proof[key]:
            return self.sources.shard(relative, product)
        expected = self.extra_proof[key].get(relative)
        if expected is None:
            raise ValueError("No verified evaluation support: " + relative)
        path = (self.extra if product == "era5" else self.sources.daily) / relative
        if sha(path) != expected:
            raise ValueError("Evaluation shard checksum differs")
        self.extra_reads[relative] = expected
        with np.load(path, allow_pickle=False) as saved:
            return saved["values"].astype("f4")

    def save(self, name, values):
        values = np.asarray(values)
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        temp.replace(path)
        read = np.load(path, mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(read, values)
        self.files[name] = dict(
            sha256=sha(path),
            shape=list(values.shape),
            dtype=str(values.dtype),
            bytes=path.stat().st_size,
        )
        return name

    def observations(self, times):
        result = np.full((len(times), 2, *self.wet.shape[-2:]), np.nan, dtype="f4")
        for i, stamp in enumerate(pd.DatetimeIndex(times)):
            if stamp.hour == 0:
                for c, product in enumerate(("sst", "adt")):
                    result[i, c] = self.shard(
                        f"{product}/{stamp.year}/{stamp:%Y-%m-%d}.npz", product
                    )[0]
        return np.where(self.wet[list(SURFACE)], result, np.nan)

    def forcing(self, times):
        result = []
        for stamp in pd.DatetimeIndex(times):
            begin = stamp - pd.Timedelta(hours=6)
            result.append(
                self.shard(f"era5/{begin.year}/{begin:%Y-%m-%d}.npz", "era5")[
                    begin.hour // 6
                ]
            )
        values = np.stack(result)
        return (
            values - self.data.stats["atmosphere_mean"][None, :, None, None]
        ) / self.data.stats["atmosphere_std"][None, :, None, None]

    def llc(self, times, names):
        actual, exact = context_times(self.sources.llc_times, times)
        ids = self.sources.llc_times.get_indexer(actual)
        values = self.sources.reader.read(CanonicalReadRequest(ids, tuple(names)))
        channels = [LLC_STATE_NAMES.index(n) for n in names]
        return np.where(self.wet[channels], values, np.nan).astype("f4"), actual, exact

    def run_case(self, model, case, source, origin, device):
        stamp = pd.Timestamp(self.sources.times[origin])
        times = pd.date_range(
            stamp + pd.Timedelta(hours=6), periods=self.hours // 6, freq="6h"
        )
        selected = [h for h in SAVED_HOURS if h <= self.hours]
        selected_times = pd.DatetimeIndex(
            [stamp + pd.Timedelta(hours=h) for h in selected]
        )
        initial_times = pd.DatetimeIndex([stamp - pd.Timedelta(hours=6), stamp])
        sample = self.sources.load(source, int(origin))
        history = np.where(
            sample["validity"][0, :29].numpy(),
            sample["surface"][0, :29].numpy() * self.std[list(SURFACE)]
            + self.mean[list(SURFACE)],
            np.nan,
        )
        record = dict(
            id=case,
            source=source,
            origin=stamp.isoformat(),
            initial_times=initial_times.astype(str).tolist(),
            times=times.astype(str).tolist(),
            saved_hours=selected,
            saved_times=selected_times.astype(str).tolist(),
            horizon_hours=self.hours,
            models={},
        )
        record["history"] = self.save(f"{case}/history.npy", history)
        record["history_times"] = (
            pd.date_range(stamp - pd.Timedelta(days=7), periods=29, freq="6h")
            .astype(str)
            .tolist()
        )
        record["observed_initial"] = self.save(
            f"{case}/observed-initial.npy", self.observations(initial_times)
        )
        obs = self.observations(times)
        record["observed_surface"] = self.save(f"{case}/observed-surface.npy", obs)
        for kind, dates, names in [
            ("initial", initial_times, LLC_STATE_NAMES),
            ("interior", selected_times, LLC_STATE_NAMES),
            ("surface", times, ("thetao_0", "zos")),
        ]:
            values, actual, exact = self.llc(dates, names)
            record["llc_" + kind] = self.save(f"{case}/llc-{kind}.npy", values)
            record["llc_" + kind + "_times"] = actual.astype(str).tolist()
            record["llc_" + kind + "_exact"] = exact.tolist()
        task = "llc" if source == "llc" else "observation"
        mask = self.data.mask.to(device)
        with autocast(device.type == "cuda", torch.bfloat16):
            initial = model.initialize(
                sample["surface"][:, :29].to(device),
                sample["atmosphere"][:, :29].to(device),
                sample["contexts"][:, 28].to(device),
                mask,
                sample["validity"][:, :29].to(device),
                task,
            )
        initial_physical = physical(initial[0], self.mean, self.std, self.wet)
        # Targets above are never passed to initialization or recurrence.
        del sample
        futures = self.forcing(times)
        states = initial
        surfaces = []
        interiors = []
        scores = {}
        initial_observed = self.observations([stamp])[0]
        llc_initial = np.load(self.output / record["llc_initial"], mmap_mode="r")[-1]
        for index, time in enumerate(times):
            hour = (index + 1) * 6
            atmosphere = torch.as_tensor(futures[index : index + 1], device=device)[
                None
            ]
            contexts = torch.as_tensor(
                context_planes(
                    self.data.grid["lat"],
                    self.data.grid["lon"],
                    pd.DatetimeIndex([time]).to_numpy(),
                ),
                device=device,
            )[None]
            with autocast(device.type == "cuda", torch.bfloat16):
                predicted, states = model.continue_forecast(
                    states, atmosphere, contexts, mask, start_lead=index + 1, task=task
                )
            raw = physical(predicted[0, 0], self.mean, self.std, self.wet)
            if not np.isfinite(raw[self.wet]).all():
                raise FloatingPointError(f"Nonfinite forecast: {case} hour{hour}")
            surfaces.append(raw[list(SURFACE)])
            if hour in selected:
                interiors.append(raw)
            if (source == "obs" and hour % 24 == 0) or (
                source == "llc" and hour in selected
            ):
                if source == "obs":
                    channels = list(SURFACE)
                    target = obs[index]
                    persist = initial_observed
                    support = np.isfinite(target) & np.isfinite(persist)
                else:
                    channels = list(range(205))
                    j = selected.index(hour)
                    if not record["llc_interior_exact"][j]:
                        if hour in (24, 48, 72):
                            raise ValueError("Validation lacks exact LLC gold")
                        continue  # Seasonal context is never a scored target.
                    target = np.load(
                        self.output / record["llc_interior"], mmap_mode="r"
                    )[j]
                    persist = llc_initial
                    support = self.wet
                scores[str(hour)] = {
                    method: score(
                        values,
                        target,
                        support,
                        self.area[channels],
                        self.std[channels, 0, 0],
                    )
                    for method, values in [
                        ("model", raw[channels]),
                        ("persistence", persist),
                    ]
                }
        arrays = dict(
            initial=self.save(f"{case}/prediction-initial.npy", initial_physical),
            surface=self.save(f"{case}/prediction-surface.npy", np.stack(surfaces)),
            interior=self.save(f"{case}/prediction-interior.npy", np.stack(interiors)),
        )
        record["prediction"] = arrays
        record["validation"] = {h: scores[h] for h in ("24", "48", "72")}
        record["diagnostics"] = scores
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoints", nargs="+", default=["main-10000.pt", "endpoint.pt"]
    )
    parser.add_argument("--hours", type=int, choices=[72, 720], default=720)
    parser.add_argument("--origins", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = ObservationTrainConfig.model_validate(
        yaml.safe_load((args.run_root / "train.yaml").read_text())
    )
    for protected in (
        args.run_root,
        args.extra,
        cfg.data.local_path("snapshots", cfg.experiment),
        cfg.data.local_path("llc", cfg.experiment),
    ):
        left, right = args.output.resolve(), Path(protected).resolve()
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Export output overlaps protected training/data paths")
    manifest = snapshot_manifest(cfg)
    export = Export(cfg, args.run_root, args.extra, args.output, args.hours)
    cases = []
    for source in export.sources.names:
        ids = export.sources.origins[source, "validation"]
        for origin in ids[
            np.unique(np.linspace(0, len(ids) - 1, args.origins, dtype=int))
        ]:
            cases.append((source, int(origin)))
    if not 1 <= args.origins <= 8:
        raise ValueError("One to eight fixed validation origins required")
    for checkpoint in args.checkpoints:
        if Path(checkpoint).name != checkpoint:
            raise ValueError("Checkpoint must be a basename")
        path = cfg.experiment.nets_dir / checkpoint
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"] != manifest:
            raise ValueError("Checkpoint runtime/data/config identity differs")
        model = cfg.model.build(LLC_STATE_NAMES).to(device)
        model.load_state_dict(saved["model"])
        EMATracker.from_state(saved["ema"], model).copy_to(model)
        model.eval()
        model.requires_grad_(False)
        identity = dict(
            checkpoint=str(path),
            checkpoint_sha256=sha(path),
            updates=saved["updates"],
            weights="EMA",
            exposure=saved["exposure"],
            manifest=manifest,
            export_script_sha256=sha(__file__),
            extra_proof_sha256=sha(args.extra / "COMPLETE.json"),
            hours=args.hours,
            origins=args.origins,
            world_size=world,
        )
        checkpoint_dir = args.output / f"update-{saved['updates']}"
        export.output = checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for i, (source, origin) in enumerate(cases):
            if i % world != rank:
                continue
            case = f"{source}-{pd.Timestamp(export.sources.times[origin]):%Y%m%dT%H%M}"
            receipt = checkpoint_dir / case / "COMPLETE.json"
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["identity"] != identity:
                    raise ValueError("Export identity differs")
                for name, info in old["files"].items():
                    if sha(checkpoint_dir / name) != info["sha256"]:
                        raise ValueError("Previously exported array changed")
                if sha(receipt.parent / "grid.npz") != old["grid_sha256"]:
                    raise ValueError("Previously exported grid changed")
                continue
            export.files = {}
            export.extra_reads = {}
            with torch.inference_mode():
                record = export.run_case(model, case, source, origin, device)
            grid = {
                k: export.data.grid[k]
                for k in ("lat", "lon", "depth", "thickness", "names", "wet_area")
            }
            grid_path = checkpoint_dir / case / "grid.npz"
            np.savez_compressed(grid_path, **grid)
            write_json(
                receipt,
                dict(
                    identity=identity,
                    case=record,
                    files=export.files,
                    grid_sha256=sha(grid_path),
                    extra_reads=export.extra_reads,
                ),
            )
            print(
                json.dumps(
                    dict(
                        event="case_complete",
                        rank=rank,
                        case=case,
                        updates=saved["updates"],
                    )
                ),
                flush=True,
            )
        del model, saved
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
