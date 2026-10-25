# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-date one-call diagnostics: native gate versus matched seed15 control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from samudra.config import EvalConfig, SamudraConfig
from samudra.datasets import ModelBatch
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import BatchPreprocessor
from samudra.utils.location import LocalLocation


def masked_highpass(x, mask):
    # Mask-aware center minus four-neighbor mean; periodic longitude.
    wet = mask.float()
    xp = F.pad(F.pad(x * wet, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
    mp = F.pad(F.pad(wet, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
    neighbors = (
        xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1] + xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:]
    )
    count = (
        mp[..., :-2, 1:-1] + mp[..., 2:, 1:-1] + mp[..., 1:-1, :-2] + mp[..., 1:-1, 2:]
    )
    return (x - neighbors / count.clamp_min(1)) * wet


p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--baseline", type=Path)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--label", default="gated")
p.add_argument("--output", type=Path)
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
cfg = EvalConfig.from_yaml(a.root / "eval.yaml")
assert isinstance(cfg.model, SamudraConfig)
cfg.experiment.data_root = LocalLocation(path=a.data)
bundle = cfg.data.build(cfg.experiment.resolved_data_root)
source = bundle.inference_source
assert source is not None
layout = bundle.data_layout
names = list(layout.prognostic_var_names)
n = len(names)
device = torch.device("cuda")
pre = BatchPreprocessor(source, names, layout.boundary_var_names)
ctx = BatchGrid(
    source.masks.prognostic_for_steps(2).to(device),
    source.resolution,
    source.resolution,
)
mask = source.masks.prognostic.to(device)
lat, lon = source.resolution
weights = mask * torch.cos(torch.deg2rad(lat.to(device)))[:, None]
denom = weights.sum((-2, -1)).clamp_min(1e-12)
std = torch.as_tensor(source.statistics(names).std, device=device)
mean = torch.as_tensor(source.statistics(names).mean, device=device)
dates = np.random.default_rng(1718).choice(len(source.time) - 3, 16, replace=False)
if a.checkpoint:
    checkpoints = [(a.label, a.checkpoint)]
else:
    assert a.baseline is not None, "--baseline required without --checkpoint"
    outputs = list((a.root / "output").glob("native-gated-1deg-seed15-*"))
    assert len(outputs) == 1, outputs
    checkpoints = [
        ("control_raw", a.baseline / "saved_nets" / "ckpt_70.pt"),
        ("control_ema", a.baseline / "saved_nets" / "ema_ckpt.pt"),
        ("gated_raw", outputs[0] / "saved_nets" / "ckpt_70.pt"),
        ("gated_ema", outputs[0] / "saved_nets" / "ema_ckpt.pt"),
    ]
out = a.output or a.root / "diagnostics"
out.mkdir(parents=True, exist_ok=True)
surface = [
    names.index(k)
    for k in ["zos", "thetao_0", "so_0", "uo_0", "vo_0", "thetao_11", "so_11"]
]
records = {}
for label, path in checkpoints:
    cfg.model.persistence_gate = label.startswith("gated")
    model = cfg.model.build(
        prog_channels=2 * n,
        boundary_channels=2 * len(layout.boundary_var_names),
        out_channels=2 * n,
        input_steps=2,
        grid_sizes=[source.grid_size],
    ).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in state["model"].items()}
    )
    model.eval()
    rows = []
    maps = {}
    with torch.no_grad():
        for j, start in enumerate(dates):
            past = np.array([start, start + 1])
            future = np.array([start + 2, start + 3])
            x = pre.prepare_planes(
                torch.as_tensor(source.read(past, names)), device
            ).reshape(1, 2 * n, *mask.shape[-2:])
            forcing = pre.prepare_planes(
                torch.as_tensor(source.read(past, layout.boundary_var_names)),
                device,
                boundary=True,
            ).reshape(1, -1, *mask.shape[-2:])
            truth = pre.prepare_planes(
                torch.as_tensor(source.read(future, names)), device
            )
            batch = ModelBatch(ctx)
            batch.append(x, forcing, truth.reshape(1, 2 * n, *mask.shape[-2:]))
            pred = model(batch)[0].float().reshape(2, n, *mask.shape[-2:])
            assert torch.isfinite(pred).all()
            hp = masked_highpass(pred, mask)
            ht = masked_highpass(truth, mask)

            def average(x):
                return (x * weights).sum((-2, -1)) / denom

            p0 = hp - average(hp)[..., None, None]
            t0 = ht - average(ht)[..., None, None]
            correlation = average(p0 * t0) / (
                average(p0.square()) * average(t0.square())
            ).sqrt().clamp_min(1e-12)
            rows.append(
                dict(
                    start_index=int(start),
                    target_dates=[str(source.time.values[i]) for i in future],
                    persistence_mse_normalized=average(
                        (x.reshape(2, n, *mask.shape[-2:])[-1:] - truth).square()
                    )
                    .cpu()
                    .tolist(),
                    mse_normalized=average((pred - truth).square()).cpu().tolist(),
                    hf_mse_normalized=average((hp - ht).square()).cpu().tolist(),
                    hf_prediction_energy=average(hp.square()).cpu().tolist(),
                    hf_truth_energy=average(ht.square()).cpu().tolist(),
                    hf_correlation=correlation.cpu().tolist(),
                )
            )
            if j < 2:
                maps[f"previous_{j}"] = (
                    (
                        x.reshape(2, n, *mask.shape[-2:])[-1, surface]
                        * std[surface, None, None]
                        + mean[surface, None, None]
                    )
                    .cpu()
                    .numpy()
                )
                maps[f"prediction_{j}"] = (
                    (
                        pred[0, surface] * std[surface, None, None]
                        + mean[surface, None, None]
                    )
                    .cpu()
                    .numpy()
                )
                maps[f"truth_{j}"] = (
                    (
                        truth[0, surface] * std[surface, None, None]
                        + mean[surface, None, None]
                    )
                    .cpu()
                    .numpy()
                )
            print("GATED_DIAGNOSTIC", label, j, flush=True)
    mse = np.mean([r["mse_normalized"] for r in rows], axis=0)
    hf = np.mean([r["hf_mse_normalized"] for r in rows], axis=0)
    records[label] = dict(
        checkpoint=str(path),
        rows=rows,
        rmse_physical=(np.sqrt(mse) * std.cpu().numpy()).tolist(),
        hf_rmse_physical=(np.sqrt(hf) * std.cpu().numpy()).tolist(),
    )
    np.savez_compressed(
        out / f"{label}-maps.npz",
        **maps,
        mask=source.masks.prognostic[surface].numpy(),
        lat=lat.numpy(),
        lon=lon.numpy(),
    )
    del model, state
    torch.cuda.empty_cache()
(out / "comparison.json").write_text(
    json.dumps(
        dict(channels=names, leads_days=[5, 10], date_seed=1718, models=records),
        indent=2,
    )
    + "\n"
)
print("GATED_DIAGNOSTICS_COMPLETE", flush=True)
