# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible, gradient-free decoder-edit experiment for checkpoint 3o112ot4.

Run with the repository environment. All downloads and generated tensors live
in the explicitly selected output directory, outside version control by default.
This script never overwrites the source checkpoint or sends data to W&B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import xarray as xr

PROG = [f"{v}_{d}" for v in ("uo", "vo", "thetao", "so") for d in range(19)] + ["zos"]
BOUNDARY = ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
ROOT = "s3://emulators/am16581/data/2025-11/om4_onedeg_v3"
SPLITS = {"discovery": 2015, "selection": 2017, "test": 2019}


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def windows(args, manifest):
    return [e for e in manifest["windows"] if e["split"] in args.splits][: args.limit]


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "source_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text())["data_root"] != args.data_root:
            raise ValueError("Use a separate output directory for each data source")
    elif (out / "data_manifest.json").exists():
        if (
            json.loads((out / "data_manifest.json").read_text())["data_root"]
            != args.data_root
        ):
            raise ValueError("Cached manifest belongs to another data source")
    save_json(identity_path, {"data_root": args.data_root})
    storage = {}
    if args.data_root.startswith("s3://"):
        from dotenv import dotenv_values

        env = dotenv_values(args.credentials_file)
        storage = {
            "key": env["OSN_KEY_ID"],
            "secret": env["OSN_KEY_SECRET"],
            "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"},
        }
    data = xr.open_zarr(
        args.data_root + "/OM4.zarr",
        consolidated=True,
        chunks=None,
        storage_options=storage,
    )
    means = xr.open_zarr(
        args.data_root + "/OM4_means.zarr",
        consolidated=True,
        chunks=None,
        storage_options=storage,
    )
    stds = xr.open_zarr(
        args.data_root + "/OM4_stds.zarr",
        consolidated=True,
        chunks=None,
        storage_options=storage,
    )
    log("Opened data and statistics")
    with ThreadPoolExecutor(max_workers=12) as pool:
        depth_masks = list(pool.map(lambda d: data[f"mask_{d}"].values, range(19)))
        mask = np.stack(
            [depth_masks[int(v.rsplit("_", 1)[1]) if "_" in v else 0] for v in PROG]
        )
        mean = np.array(
            list(pool.map(lambda v: means[v].item(), PROG)), dtype=np.float32
        )
        std = np.array(list(pool.map(lambda v: stds[v].item(), PROG)), dtype=np.float32)
    assert np.isfinite(mean).all() and (std > 0).all()
    np.savez(
        out / "static.npz",
        mask=mask,
        mean=mean,
        std=std,
        area=data.areacello.values,
        dz=data.dz.values,
        lev=data.lev.values,
        lat=data.y.values,
        lon=data.x.values,
        fraction=data.ocean_fraction.values,
    )

    # Reproduce compute_anomalies at the training commit: day-of-year climatology
    # over the entire source record, then one scalar mean/std. This legacy input
    # transform includes future forcing; it is not recomputed to favor edits.
    hfds_path = out / "hfds.npy"
    if not hfds_path.exists():
        log("Downloading full hfds record to reproduce legacy forcing anomalies")
        raw = np.lib.format.open_memmap(
            hfds_path.with_suffix(".partial.npy"),
            mode="w+",
            dtype=np.float32,
            shape=data.hfds.shape,
        )
        starts = list(range(0, len(data.time), 32))

        def fetch(start):
            stop = min(start + 32, len(data.time))
            raw[start:stop] = data.hfds.isel(time=slice(start, stop)).values

        with ThreadPoolExecutor(max_workers=16) as pool:
            for n, _ in enumerate(pool.map(fetch, starts), 1):
                if n % 20 == 0:
                    log(f"hfds chunks {n}/{len(starts)}")
        raw.flush()
        del raw
        hfds_path.with_suffix(".partial.npy").rename(hfds_path)
    raw = np.load(hfds_path, mmap_mode="r")
    doy = data.time.dt.dayofyear.values
    anom_path = out / "hfds_anomalies.npy"
    if not anom_path.exists():
        anomalies = np.lib.format.open_memmap(
            anom_path.with_suffix(".partial.npy"),
            mode="w+",
            dtype=np.float32,
            shape=raw.shape,
        )
        for day in np.unique(doy):
            indices = np.flatnonzero(doy == day)
            block = raw[indices]
            count = np.isfinite(block).sum(axis=0)
            clim = np.nansum(block, axis=0) / np.maximum(count, 1)
            clim = clim.astype(np.float32)
            anomalies[indices] = block - clim
        anomalies.flush()
        del anomalies
        anom_path.with_suffix(".partial.npy").rename(anom_path)
    anomalies = np.load(anom_path, mmap_mode="r")
    total = total2 = 0.0
    count = 0
    for start in range(0, len(data.time), 32):
        a = np.asarray(anomalies[start : start + 32], dtype=np.float64)
        total += float(np.nansum(a))
        total2 += float(np.nansum(a * a))
        count += np.isfinite(a).sum()
    amean = total / count
    astd = np.sqrt(total2 / count - amean**2)
    bmean = np.array(
        [means[v].item() for v in BOUNDARY[:-1]] + [amean], dtype=np.float32
    )
    bstd = np.array([stds[v].item() for v in BOUNDARY[:-1]] + [astd], dtype=np.float32)
    np.savez(out / "forcing_stats.npz", mean=bmean, std=bstd)
    log(f"Legacy hfds anomaly mean/std: {amean:.8g}, {astd:.8g}")
    manifest = {
        "data_root": args.data_root,
        "data_attrs": data.attrs,
        "prognostic_channels": PROG,
        "boundary_channels": BOUNDARY,
        "anomaly_transform": "full-record day-of-year climatology; scalar full-record mean/std",
        "normalization_before_mask": True,
        "masked_fill": 0.0,
        "input_steps": 2,
        "output_steps": 2,
        "windows": [],
    }
    for split, year in SPLITS.items():
        for month in (1, 4, 7, 10):
            indices = np.flatnonzero(
                (data.time.dt.year.values == year)
                & (data.time.dt.month.values == month)
            )
            start = int(indices[0])
            name = f"{split}_{year}_{month:02d}"
            # The first discovery window also supports six model calls (60 days).
            length = 14 if split == "discovery" and month == 1 else 4
            entry = {
                "name": name,
                "split": split,
                "start_index": start,
                "times": [str(v) for v in data.time.values[start : start + length]],
            }
            manifest["windows"].append(entry)
            target = out / f"{name}.npz"
            if target.exists():
                continue
            log(f"Fetching {name}, {length} timesteps")

            def read_var(v):
                return data[v].isel(time=slice(start, start + length)).values

            with ThreadPoolExecutor(max_workers=12) as pool:
                states = np.stack(list(pool.map(read_var, PROG)), axis=1)
                boundary = np.stack(
                    list(pool.map(read_var, BOUNDARY[:-1]))
                    + [anomalies[start : start + length]],
                    axis=1,
                )
            assert np.isfinite(states[:, mask]).all(), name
            states = np.where(
                mask[None],
                (states - mean[None, :, None, None]) / std[None, :, None, None],
                0,
            ).astype(np.float32)
            boundary = np.where(
                mask[0][None, None],
                (boundary - bmean[None, :, None, None]) / bstd[None, :, None, None],
                0,
            ).astype(np.float32)
            assert np.isfinite(states).all() and np.isfinite(boundary).all()
            np.savez(target, states=states, boundary=boundary)
    save_json(out / "data_manifest.json", manifest)
    log("Data preparation complete")


def model_from_checkpoint(args):
    import yaml

    from samudra.config import SamudraConfig

    torch.set_num_threads(4)
    cfg = SamudraConfig(
        **yaml.safe_load(
            Path("src/samudra/configs/samudra_om4_v2/model.yaml").read_text()
        )
    )
    cfg.checkpointing = None
    model = cfg.build(
        prog_channels=154,
        boundary_channels=8,
        out_channels=154,
        input_steps=2,
        grid_sizes=[(180, 360)],
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in checkpoint["model"].items()},
        strict=True,
    )
    model.eval().requires_grad_(False)
    return model.to(args.device)


def baseline(args):
    from samudra.utils.ctx import BatchGrid

    out = args.output
    static = np.load(out / "static.npz")
    manifest = json.loads((out / "data_manifest.json").read_text())
    model = model_from_checkpoint(args)
    ctx = BatchGrid(
        torch.from_numpy(np.tile(static["mask"], (2, 1, 1))).to(args.device),
        (torch.from_numpy(static["lat"]), torch.from_numpy(static["lon"])),
        (torch.from_numpy(static["lat"]), torch.from_numpy(static["lon"])),
    )
    captured = {}

    def hook(module, inputs):
        captured["features"] = inputs[0].detach().cpu().numpy()

    handle = model.decoder.register_forward_pre_hook(hook)
    for entry in windows(args, manifest):
        if entry["split"] not in args.splits:
            continue
        name = entry["name"]
        target = out / f"{name}_baseline.npy"
        if target.exists():
            continue
        data = np.load(out / f"{name}.npz")
        prog = torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
            args.device
        )
        boundary = torch.from_numpy(data["boundary"][:2].reshape(1, 8, 180, 360)).to(
            args.device
        )
        start = time.monotonic()
        with torch.inference_mode():
            pred = (
                model.forward_once(prog, boundary, ctx)
                .cpu()
                .numpy()
                .reshape(2, 77, 180, 360)
            )
        assert np.isfinite(pred).all()
        np.save(out / f"{name}_features.npy", captured.pop("features"))
        np.save(target, pred)
        log(f"{name}: forward {time.monotonic() - start:.1f}s")
    handle.remove()
    save_json(
        out / "baseline_provenance.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "torch": torch.__version__,
            "device": args.device,
            "precision": "float32",
            "gradients": False,
        },
    )


class Diagnostics:
    """Wet-edge, metric-distance diagnostics; no derivative crosses land."""

    def __init__(self, static):
        self.mask = static["mask"]
        self.std = static["std"]
        self.mean = static["mean"]
        self.area = static["area"]
        self.lev = static["lev"]
        self.lat = static["lat"]
        self.lon = static["lon"]
        m = self.mask
        interior = m & np.roll(m, 1, -1) & np.roll(m, -1, -1)
        interior[:, 1:] &= m[:, :-1]
        interior[:, :-1] &= m[:, 1:]
        self.coast = m & ~interior
        self.weights = m * self.area
        self.coast_weights = self.coast * self.area
        self.interior_weights = interior * self.area
        self.dx = (
            6371.0
            * np.cos(np.deg2rad(self.lat))[:, None]
            * np.deg2rad(np.diff(self.lon).mean())
        )
        self.dy = 6371.0 * np.deg2rad(np.diff(self.lat))[None, :, None]
        self.ex = m & np.roll(m, -1, -1)
        self.ey = m[:, :-1] & m[:, 1:]
        self.wx = self.ex * (self.area + np.roll(self.area, -1, -1)) / 2
        self.wy = self.ey * (self.area[:-1] + self.area[1:]) / 2
        self.cwx = self.wx * (self.coast | np.roll(self.coast, -1, -1))
        self.cwy = self.wy * (self.coast[:, :-1] | self.coast[:, 1:])

    @staticmethod
    def average(a, weights):
        numerator = np.sum(a * weights, axis=(-2, -1), dtype=np.float64)
        denominator = weights.sum(axis=(-2, -1))
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )

    def gradients(self, field, coastal=False):
        dx = (np.roll(field, -1, -1) - field) / self.dx
        dy = (field[..., 1:, :] - field[..., :-1, :]) / self.dy
        wx, wy = (self.cwx, self.cwy) if coastal else (self.wx, self.wy)
        num = np.sum(dx**2 * wx, axis=(-2, -1)) + np.sum(dy**2 * wy, axis=(-2, -1))
        den = wx.sum(axis=(-2, -1)) + wy.sum(axis=(-2, -1))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def evaluate(self, pred, truth, previous):
        error = pred.astype(np.float64) - truth
        mse = self.average(error**2, self.weights)
        cmse = self.average(error**2, self.coast_weights)
        imse = self.average(error**2, self.interior_weights)
        bias = self.average(error, self.weights)
        grad = self.gradients(error)
        cgrad = self.gradients(error, coastal=True)
        increment = pred - previous[None]
        actual_increment = truth - previous[None]
        delta = self.average(increment.astype(np.float64) ** 2, self.weights)
        actual_delta = self.average(
            actual_increment.astype(np.float64) ** 2, self.weights
        )
        pred_variance = self.average(
            (pred - self.average(pred, self.weights)[..., None, None]) ** 2,
            self.weights,
        )
        true_variance = self.average(
            (truth - self.average(truth, self.weights)[..., None, None]) ** 2,
            self.weights,
        )
        physical = pred * self.std[None, :, None, None] + self.mean[None, :, None, None]
        pair_error = self.average(
            ((pred[1] - pred[0]) - (truth[1] - truth[0])) ** 2, self.weights
        )
        pair_pred = self.average((pred[1] - pred[0]) ** 2, self.weights)
        pair_true = self.average((truth[1] - truth[0]) ** 2, self.weights)
        rows = []
        for c, name in enumerate(PROG):
            for lead in range(2):
                rows.append(
                    {
                        "channel": name,
                        "lead_days": (lead + 1) * 5,
                        "rmse_normalized": float(np.sqrt(mse[lead, c])),
                        "rmse_physical": float(np.sqrt(mse[lead, c]) * self.std[c]),
                        "coastal_rmse_normalized": float(np.sqrt(cmse[lead, c])),
                        "interior_rmse_normalized": float(np.sqrt(imse[lead, c])),
                        "bias_physical": float(bias[lead, c] * self.std[c]),
                        "gradient_rmse_normalized_per_km": float(
                            np.sqrt(grad[lead, c])
                        ),
                        "coastal_gradient_rmse_normalized_per_km": float(
                            np.sqrt(cgrad[lead, c])
                        ),
                        "increment_rms_physical": float(
                            np.sqrt(delta[lead, c]) * self.std[c]
                        ),
                        "true_increment_rms_physical": float(
                            np.sqrt(actual_delta[lead, c]) * self.std[c]
                        ),
                        "pair_increment_rmse_physical": float(
                            np.sqrt(pair_error[c]) * self.std[c]
                        ),
                        "pair_increment_rms_physical": float(
                            np.sqrt(pair_pred[c]) * self.std[c]
                        ),
                        "true_pair_increment_rms_physical": float(
                            np.sqrt(pair_true[c]) * self.std[c]
                        ),
                        "spatial_std_ratio": float(
                            np.sqrt(
                                pred_variance[lead, c]
                                / max(true_variance[lead, c], 1e-30)
                            )
                        ),
                        "min_physical": float(physical[lead, c][self.mask[c]].min()),
                        "max_physical": float(physical[lead, c][self.mask[c]].max()),
                    }
                )
        return rows


def diagnose(args):
    out = args.output
    diag = Diagnostics(np.load(out / "static.npz"))
    manifest = json.loads((out / "data_manifest.json").read_text())
    all_rows = []
    for entry in windows(args, manifest):
        if entry["split"] not in args.splits:
            continue
        name = entry["name"]
        data = np.load(out / f"{name}.npz")
        pred = np.load(out / f"{name}_baseline.npy")
        rows = diag.evaluate(pred, data["states"][2:4], data["states"][1])
        for row in rows:
            row.update(window=name, split=entry["split"], candidate="baseline")
        all_rows.extend(rows)
        log(f"Diagnosed {name}")
    save_json(out / ("baseline_metrics_" + "_".join(args.splits) + ".json"), all_rows)


def candidate_delta(weight, family, rows):
    """Return a reasoned, fixed tensor intervention, without fitting to targets."""
    delta = torch.zeros_like(weight)
    w = weight[rows]
    if family == "spatial_contrast":
        delta[rows] = -(w - w.mean(dim=(-2, -1), keepdim=True))
    elif family in ("zonal_curvature", "meridional_curvature", "checkerboard"):
        # Unit-norm zero-DC basis; preserve every input/output kernel's sum.
        constant = torch.ones(3, dtype=w.dtype) / np.sqrt(3)
        curvature = torch.tensor([1.0, -2.0, 1.0], dtype=w.dtype) / np.sqrt(6)
        y = curvature if family != "zonal_curvature" else constant
        x = curvature if family != "meridional_curvature" else constant
        basis = y[:, None] * x[None, :]
        delta[rows] = -(w * basis).sum(dim=(-2, -1), keepdim=True) * basis
    elif family == "velocity_direction":
        # The leading right singular vector of deep-velocity decoder rows is a
        # hypothesis about a shared velocity-writing direction, not proof of
        # imprinting. Remove its zero-DC component only from tracer rows.
        velocity_rows = [
            t * 77 + c
            for t in range(2)
            for c in list(range(10, 19)) + list(range(29, 38))
        ]
        v = (
            torch.linalg.svd(weight[velocity_rows].flatten(1), full_matrices=False)
            .Vh[0]
            .reshape(280, 3, 3)
        )
        v = v - v.mean(dim=(-2, -1), keepdim=True)
        v = v / v.norm()
        delta[rows] = -(w * v).sum(dim=(1, 2, 3), keepdim=True) * v
    else:
        raise ValueError(family)
    assert torch.allclose(
        delta.sum(dim=(-2, -1)), torch.zeros_like(delta[..., 0, 0]), atol=2e-6
    )
    return delta


def screen(args):
    """Evaluate decoder edits using cached features; the backbone stays fixed."""
    out = args.output
    torch.set_num_threads(4)
    static = np.load(out / "static.npz")
    diag = Diagnostics(static)
    manifest = json.loads((out / "data_manifest.json").read_text())
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weight = ckpt["model"]["module.decoder.weight"].clone()
    bias = ckpt["model"]["module.decoder.bias"].clone()
    # Deep tracer rows >=775 m, independently for both forecast times.
    rows = [
        t * 77 + c for t in range(2) for c in list(range(48, 57)) + list(range(67, 76))
    ]
    families = [
        "spatial_contrast",
        "zonal_curvature",
        "meridional_curvature",
        "checkerboard",
        "velocity_direction",
        "temporal_pair",
    ]
    deltas = {
        family: candidate_delta(weight, family, rows)
        for family in families
        if family != "temporal_pair"
    }
    pair_delta, pair_bias = temporal_pair_delta(weight, bias, rows)
    deltas["temporal_pair"] = pair_delta
    spatial_alphas = [0.005, 0.02, 0.05, 0.1, 0.2]
    temporal_alphas = [0.1, 0.2, 0.5, 1.0]
    selected = {
        "rows": rows,
        "families": families,
        "alphas": spatial_alphas,
        "temporal_pair_alphas": temporal_alphas,
    }
    save_json(out / "screen_design.json", selected)
    summaries = []
    for entry in windows(args, manifest):
        if entry["split"] not in args.splits:
            continue
        name = entry["name"]
        data = np.load(out / f"{name}.npz")
        pred = np.load(out / f"{name}_baseline.npy")
        fts = torch.from_numpy(np.load(out / f"{name}_features.npy")).to(args.device)
        truth = data["states"][2:4]
        base_error = pred.astype(np.float64) - truth
        base_mse = diag.average(base_error**2, diag.weights)
        base_grad = diag.gradients(base_error, coastal=True)
        base_pair_mse = diag.average((base_error[1] - base_error[0]) ** 2, diag.weights)
        for family, delta in deltas.items():
            with torch.inference_mode():
                db = (
                    pair_bias[rows].to(args.device)
                    if family == "temporal_pair"
                    else None
                )
                effect = (
                    torch.nn.functional.conv2d(fts, delta[rows].to(args.device), db)
                    .cpu()
                    .numpy()[0]
                )
            full_effect = np.zeros((154, 180, 360), dtype=np.float32)
            full_effect[rows] = effect
            full_effect = full_effect.reshape(2, 77, 180, 360) * static["mask"][None]
            np.save(
                out / f"{name}_{family}_effect.npy",
                full_effect[:, [*range(48, 57), *range(67, 76)]],
            )
            for alpha in (
                temporal_alphas if family == "temporal_pair" else spatial_alphas
            ):
                err = base_error + alpha * full_effect
                mse = diag.average(err**2, diag.weights)
                grad = diag.gradients(err, coastal=True)
                pair_mse = diag.average((err[1] - err[0]) ** 2, diag.weights)
                channels = [*range(48, 57), *range(67, 76)]
                summaries.append(
                    {
                        "window": name,
                        "split": entry["split"],
                        "family": family,
                        "alpha": alpha,
                        "global_rmse_ratio": float(
                            np.sqrt(mse.mean() / base_mse.mean())
                        ),
                        "deep_tracer_rmse_ratio": float(
                            np.sqrt(
                                mse[:, channels].mean() / base_mse[:, channels].mean()
                            )
                        ),
                        "deep_tracer_coastal_gradient_ratio": float(
                            np.sqrt(
                                grad[:, channels].mean() / base_grad[:, channels].mean()
                            )
                        ),
                        "deep_tracer_pair_increment_rmse_ratio": float(
                            np.sqrt(
                                pair_mse[channels].mean()
                                / base_pair_mse[channels].mean()
                            )
                        ),
                        "worst_channel_rmse_ratio": float(
                            np.sqrt(mse / np.maximum(base_mse, 1e-30)).max()
                        ),
                    }
                )
            log(f"Screened {name}: {family}")
    save_json(out / ("screen_" + "_".join(args.splits) + ".json"), summaries)


def temporal_pair_delta(weight, bias, rows):
    """Attenuate output-head disagreement while preserving their exact mean."""
    assert set(rows) == {((c + 77) % 154) for c in rows}
    delta = torch.zeros_like(weight)
    db = torch.zeros_like(bias)
    opposite = [(c + 77) % 154 for c in rows]
    delta[rows] = (weight[opposite] - weight[rows]) / 2
    db[rows] = (bias[opposite] - bias[rows]) / 2
    return delta, db


def edited_decoder(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weight = ckpt["model"]["module.decoder.weight"].clone()
    bias = ckpt["model"]["module.decoder.bias"].clone()
    rows = [t * 77 + c for t in range(2) for c in [*range(48, 57), *range(67, 76)]]
    if args.family == "temporal_pair":
        delta, db = temporal_pair_delta(weight, bias, rows)
    else:
        delta = candidate_delta(weight, args.family, rows)
        db = torch.zeros_like(bias)
    return weight + args.alpha * delta, bias + args.alpha * db, rows


def confirm(args):
    """Apply one frozen edit to cached features and record all channel metrics."""
    out = args.output
    static = np.load(out / "static.npz")
    diag = Diagnostics(static)
    manifest = json.loads((out / "data_manifest.json").read_text())
    weight, bias, rows = edited_decoder(args)
    device_weight, device_bias = weight.to(args.device), bias.to(args.device)
    all_rows = []
    tag = f"{args.family}_{args.alpha:g}"
    for entry in windows(args, manifest):
        if entry["split"] not in args.splits:
            continue
        name = entry["name"]
        data = np.load(out / f"{name}.npz")
        fts = torch.from_numpy(np.load(out / f"{name}_features.npy")).to(args.device)
        with torch.inference_mode():
            pred = (
                torch.nn.functional.conv2d(fts, device_weight, device_bias)
                .cpu()
                .numpy()
                .reshape(2, 77, 180, 360)
            )
        pred = np.where(static["mask"][None], pred, 0)
        assert np.isfinite(pred).all()
        np.save(out / f"{name}_{tag}.npy", pred)
        for row in diag.evaluate(pred, data["states"][2:4], data["states"][1]):
            row.update(window=name, split=entry["split"], candidate=tag)
            all_rows.append(row)
        log(f"Confirmed {name}: {tag}")
    save_json(out / (tag + "_metrics_" + "_".join(args.splits) + ".json"), all_rows)


def rollout(args):
    """Compare baseline and one edit on a six-call free-running trajectory."""
    from samudra.utils.ctx import BatchGrid

    out = args.output
    static = np.load(out / "static.npz")
    diag = Diagnostics(static)
    data = np.load(out / "discovery_2015_01.npz")
    ctx = BatchGrid(
        torch.from_numpy(np.tile(static["mask"], (2, 1, 1))).to(args.device),
        (torch.from_numpy(static["lat"]), torch.from_numpy(static["lon"])),
        (torch.from_numpy(static["lat"]), torch.from_numpy(static["lon"])),
    )
    model = model_from_checkpoint(args)
    weight, bias, _ = edited_decoder(args)
    tag = f"{args.family}_{args.alpha:g}"
    records = []
    for label in ("baseline", tag):
        if label != "baseline":
            model.decoder.load_state_dict({"weight": weight, "bias": bias})
        prog = torch.from_numpy(data["states"][:2].reshape(1, 154, 180, 360)).to(
            args.device
        )
        predictions = []
        for step in range(6):
            offset = 2 * step
            boundary = torch.from_numpy(
                data["boundary"][offset : offset + 2].reshape(1, 8, 180, 360)
            ).to(args.device)
            with torch.inference_mode():
                pred = model.forward_once(prog, boundary, ctx)
            prog = pred
            p = pred.cpu().numpy().reshape(2, 77, 180, 360)
            assert np.isfinite(p).all()
            predictions.append(p)
            for row in diag.evaluate(
                p, data["states"][offset + 2 : offset + 4], data["states"][1]
            ):
                row.update(
                    candidate=label,
                    model_call=step + 1,
                    lead_days=offset * 5 + row["lead_days"],
                )
                records.append(row)
            log(f"Rollout {label}: {step + 1}/6")
        np.save(out / f"rollout_{label}.npy", np.concatenate(predictions))
    save_json(out / f"rollout_{tag}_metrics.json", records)


def export(args):
    """Save an inference-only candidate, with a complete edit manifest."""
    out = args.output
    weight, bias, rows = edited_decoder(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    original_w = ckpt["model"]["module.decoder.weight"]
    original_b = ckpt["model"]["module.decoder.bias"]
    provenance = {
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "family": args.family,
        "alpha": args.alpha,
        "output_rows": rows,
        "changed_weight_values": int(torch.count_nonzero(weight - original_w)),
        "changed_bias_values": int(torch.count_nonzero(bias - original_b)),
        "decoder_relative_change": float(
            (weight - original_w).norm() / original_w.norm()
        ),
        "method": "direct tensor edit; no gradients, optimizer, or fitted regression",
        "run_id": "3o112ot4",
        "source_training_commit": "edd7b8c48cd6d835d0d77868aae15ce5c85dd4e4",  # pragma: allowlist secret
        "data_manifest": json.loads((out / "data_manifest.json").read_text()),
    }
    state = dict(ckpt["model"])
    state["module.decoder.weight"] = weight
    state["module.decoder.bias"] = bias
    tag = f"{args.family}_{args.alpha:g}"
    path = out / f"samudra2_onedeg_ema_{tag}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model": state, "edit_provenance": provenance}, path)
    provenance["edited_checkpoint_sha256"] = sha256(path)
    save_json(out / f"{tag}_edit_manifest.json", provenance)
    log(f"Saved {path}")


def plot(args):
    """Export matched-scale maps and depth profiles for a fixed candidate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    static = np.load(out / "static.npz")
    tag = f"{args.family}_{args.alpha:g}"
    name = args.window
    data = np.load(out / f"{name}.npz")
    baseline_pred = np.load(out / f"{name}_baseline.npy")
    candidate = np.load(out / f"{name}_{tag}.npy")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for row, c in enumerate((53, 72)):
        fields = [
            (data["states"][2, c] - data["states"][1, c]),
            (baseline_pred[0, c] - data["states"][1, c]),
            (candidate[0, c] - data["states"][1, c]),
        ]
        fields = [
            np.where(static["mask"][c], a * static["std"][c], np.nan) for a in fields
        ]
        limit = float(np.nanquantile(np.abs(np.stack(fields)), 0.99))
        for col, (field, title) in enumerate(
            zip(fields, ("OM4 reference", "Original checkpoint", tag), strict=True)
        ):
            im = axes[row, col].pcolormesh(
                static["lon"],
                static["lat"],
                field,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                rasterized=True,
            )
            axes[row, col].set_title(f"{PROG[c]} (3100 m): {title}")
            axes[row, col].set(xlabel="Longitude", ylabel="Latitude")
            axes[row, col].set_facecolor("#d8d8d8")
        fig.colorbar(
            im,
            ax=axes[row, :],
            label="5-day temperature change (°C)"
            if row == 0
            else "5-day salinity change (psu)",
            shrink=0.85,
        )
    fig.suptitle(
        f"Single-call increments from the most recent input — {name}\nShared color scale within each row; limits at pooled 99th percentile"
    )
    fig.savefig(out / f"{name}_{tag}_increments.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "baseline",
            "diagnose",
            "screen",
            "confirm",
            "rollout",
            "export",
            "plot",
        ],
    )
    parser.add_argument("--data-root", default=ROOT)
    parser.add_argument("--credentials-file", type=Path, default=Path("alex.env"))
    parser.add_argument(
        "--output", type=Path, default=Path(".LOCAL/weight-edit-original")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("samudra2_onedeg_ema_ckpt.pt")
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--family", default="temporal_pair")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--window", default="discovery_2015_01")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--splits", nargs="+", default=["discovery", "selection", "test"]
    )
    args = parser.parse_args()
    {
        "prepare": prepare,
        "baseline": baseline,
        "diagnose": diagnose,
        "screen": screen,
        "confirm": confirm,
        "rollout": rollout,
        "export": export,
        "plot": plot,
    }[args.action](args)


if __name__ == "__main__":
    main()
