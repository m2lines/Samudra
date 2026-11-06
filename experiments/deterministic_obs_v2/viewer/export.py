# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only export of frozen 5k EMA endpoints for the results viewer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.constants import build_om4_layout
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.observations.archive import context_planes
from samudra.observations.evaluate import complete_month_weights
from samudra.observations.samples import Samples
from samudra.utils.ema import EMATracker


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_origin(directory):
    manifest = json.loads((directory / "COMPLETE.json").read_text())
    assert (
        manifest["history_bins"],
        manifest["forecast_bins"],
        manifest["bin_days"],
    ) == (19, 73, 5)
    rows = []
    for i, record in enumerate(manifest["records"]):
        p = directory / record["file"]
        assert p.parent == directory and sha(p) == record["sha256"]
        with np.load(p) as z:
            row = dict(z)
        start = pd.Timestamp(manifest["origin"]) + pd.Timedelta(days=5 * (i - 19))
        assert pd.Timestamp(str(row["start"])) == start
        assert pd.Timestamp(str(row["end"])) == start + pd.Timedelta(days=5)
        rows.append(row)
    assert len(rows) == 92
    return manifest, {
        k: np.stack([r[k] for r in rows])
        for k in ("surface", "atmosphere", "start", "midpoint")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["C", "D"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path("/scratch/jr7309/deterministic-obs-v2")
    run = root / "screens-v1" / {"C": "C-obs-large", "D": "D-mixed-large"}[a.arm]
    checkpoint = run / "saved_nets/endpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert saved["updates"] == 5000 and saved["cooled"]
    data = Samples(
        "/scratch/jr7309/data/obs-d-pilot", "cuda", normalization=root / "normalization"
    )
    cfg = SurfaceInitializedConfig.model_validate(saved["manifest"]["config"]["model"])
    model = cfg.build(data.grid["names"].tolist()).cuda().eval()
    model.load_state_dict(saved["model"])
    # EMA uses named_parameters; this model implements that nn.Module interface.
    EMATracker.from_state(saved["ema"], model).copy_to(model)  # type: ignore[arg-type]
    model.requires_grad_(False)
    a.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        arm=a.arm,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        updates=5000,
        weights="EMA",
        exposure=saved["exposure"],
        training_manifest=saved["manifest"],
        exporter_sha256=sha(__file__),
        cases={},
    )
    layout = build_om4_layout()
    edges = np.r_[0, np.cumsum(layout.depth_thickness)]
    mask = data.grid["mask"].astype(bool)
    depths = np.array(layout.depth_levels)
    for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
        out = a.output / (origin + ".npz")
        directory = Path("/scratch/jr7309/data/obs-d-annual-instance-v1/test") / origin
        description, raw = read_origin(directory)
        starts = raw["start"][19:].astype("datetime64[ns]")
        ends = starts + np.timedelta64(5, "D")
        months, weights = complete_month_weights(starts, ends)
        assert len(months) == 12
        surface = raw["surface"][:19]
        valid = np.isfinite(surface) & mask[[38, 76]]
        normalized = np.where(
            valid,
            (np.where(valid, surface, 0) - data.grid["mean"][[38, 76], None, None])
            / data.grid["std"][[38, 76], None, None],
            0,
        )
        atmosphere = (
            raw["atmosphere"] - data.stats["atmosphere_mean"][None, :, None, None]
        ) / data.stats["atmosphere_std"][None, :, None, None]
        assert np.isfinite(atmosphere).all()
        forcing = data.tensor(atmosphere)[None]
        contexts = data.tensor(
            context_planes(data.grid["lat"], data.grid["lon"], raw["midpoint"])
        )[None]
        monthly = np.zeros((12, 2, 19, *mask.shape[-2:]), np.float64)
        surfaces, snapshots = [], []
        leads = [5, 15, 30, 90, 180, 365]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            states = model.initialize(
                data.tensor(normalized)[None],
                forcing[:, :19],
                contexts[:, 18],
                data.mask,
                data.tensor(valid)[None],
            )
            initial = data.physical(states)[0, -1].float().cpu().numpy()
            for step in range(73):
                prediction, states = model.continue_forecast(
                    states,
                    forcing[:, 19 + step : 20 + step],
                    contexts[:, 19 + step : 20 + step],
                    data.mask,
                    start_lead=step + 1,
                )
                field = data.physical(prediction)[0, 0].float().cpu().numpy()
                assert np.isfinite(field[mask]).all(), (origin, step)
                surfaces.append(field[[38, 76]])
                monthly += (
                    weights[:, step, None, None, None, None]
                    * field[38:76].reshape(2, 19, *mask.shape[-2:])[None]
                )
                if (step + 1) * 5 in leads:
                    snapshots.append(field)
        monthly = monthly.astype(np.float32)
        heat = []
        for lo, hi in [(0, 700), (700, 2000)]:
            dz = np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
            used = dz > 0
            wet = mask[38:57][used].all(0)
            heat.append(
                np.where(
                    wet,
                    (monthly[:, 0, used].astype(np.float64) * dz[used, None, None]).sum(
                        1
                    )
                    * 1035
                    * 3850,
                    np.nan,
                )
            )
        refs, ref_heat = [], []
        for month, record in zip(months, description["monthly_interiors"], strict=True):
            path = directory / record["file"]
            assert (
                path.parent == directory
                and sha(path) == record["sha256"]
                and path.name.startswith(month)
            )
            with np.load(path) as z:
                refs.append(z["values"])
                ref_heat.append(z["ohc"])
        fields = dict(
            initial=initial,
            surface=np.stack(surfaces),
            forecast_interior=np.stack(snapshots),
            monthly_ts=monthly,
            monthly_ohc=np.stack(heat, axis=1),
            reference_surface=raw["surface"][19:],
            reference_monthly_ts=np.stack(refs),
            reference_monthly_ohc=np.stack(ref_heat),
            months=np.array(months),
            starts=starts.astype(str),
            ends=ends.astype(str),
            interior_leads=np.array(leads),
            mask=mask,
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            depth=depths,
            depth_interfaces=edges,
            channel_names=data.grid["names"],
            initial_start=str(starts[0] - np.timedelta64(5, "D")),
            initial_end=origin,
        )
        prior = pd.Timestamp(origin) - pd.offsets.MonthBegin()
        context = Path("/scratch/jr7309/data/obs-d-pilot/test") / (
            prior.strftime("%Y-%m") + ".npz"
        )
        if context.exists():
            with np.load(context) as z:
                fields.update(
                    initial_iap_ts=z["interior"],
                    initial_context_month=prior.strftime("%Y-%m"),
                )
        np.savez_compressed(out, **fields)
        with np.load(out) as check:
            for key, value in fields.items():
                np.testing.assert_array_equal(check[key], value)
        receipt["cases"][origin] = dict(
            file=out.name,
            sha256=sha(out),
            annual_manifest_sha256=sha(directory / "COMPLETE.json"),
            context_sha256=sha(context) if context.exists() else None,
        )
        print("EXPORTED", a.arm, origin, flush=True)
    (a.output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
