# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Day-5 and day-360 continuous rollouts of the final native gate and control."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from samudra.config import EvalConfig, SamudraConfig
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
p.add_argument("--baseline", type=Path, required=True)
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
start = int(
    np.random.default_rng(1718).choice(len(source.time) - 3, 16, replace=False)[0]
)
assert start + 73 < len(source.time)
outputs = list((a.root / "output").glob("native-gated-1deg-seed15-*"))
assert len(outputs) == 1
out = a.root / "maps360"
out.mkdir(exist_ok=True)
selected = [
    names.index(k)
    for k in ["zos", "thetao_0", "so_0", "uo_0", "vo_0", "thetao_11", "so_11"]
]
record: dict[str, Any] = dict(
    channels=[names[i] for i in selected],
    start_index=start,
    origin_time=str(source.time.values[start + 1]),
    days=[5, 360],
    models={},
)
for label, path in [
    ("control", a.baseline / "saved_nets" / "ema_ckpt.pt"),
    ("gated", outputs[0] / "saved_nets" / "ema_ckpt.pt"),
]:
    cfg.model.persistence_gate = label == "gated"
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
    x = pre.prepare_planes(
        torch.as_tensor(source.read(np.array([start, start + 1]), names)), device
    ).reshape(1, 2 * n, *mask.shape[-2:])
    maps = {
        "initial": (
            x.reshape(2, n, *mask.shape[-2:])[-1, selected] * std[selected, None, None]
            + mean[selected, None, None]
        )
        .cpu()
        .numpy()
    }
    metrics = {}
    with torch.no_grad():
        for call in range(36):
            past = np.array([start + 2 * call, start + 2 * call + 1])
            future = past + 2
            forcing = pre.prepare_planes(
                torch.as_tensor(source.read(past, layout.boundary_var_names)),
                device,
                boundary=True,
            ).reshape(1, -1, *mask.shape[-2:])
            pred = (
                model.forward_once(x, forcing, ctx)
                .float()
                .reshape(2, n, *mask.shape[-2:])
            )
            assert torch.isfinite(pred).all()
            for lead in range(2):
                day = (2 * call + lead + 1) * 5
                if day not in (5, 360):
                    continue
                truth = pre.prepare_planes(
                    torch.as_tensor(source.read(future, names)), device
                )[lead]
                error = pred[lead] - truth
                metrics[str(day)] = dict(
                    date=str(source.time.values[future[lead]]),
                    rmse_physical=(
                        ((error.square() * weights).sum((-2, -1)) / denom).sqrt() * std
                    )
                    .cpu()
                    .tolist(),
                )
                maps[f"prediction_{day}"] = (
                    (
                        pred[lead, selected] * std[selected, None, None]
                        + mean[selected, None, None]
                    )
                    .cpu()
                    .numpy()
                )
                maps[f"truth_{day}"] = (
                    (
                        truth[selected] * std[selected, None, None]
                        + mean[selected, None, None]
                    )
                    .cpu()
                    .numpy()
                )
            x = pred.reshape(1, 2 * n, *mask.shape[-2:])
            print("MAP_ROLLOUT", label, call + 1, flush=True)
    np.savez_compressed(
        out / f"{label}.npz",
        **maps,
        mask=mask[selected].cpu().numpy(),
        lat=lat.numpy(),
        lon=lon.numpy(),
    )
    record["models"][label] = metrics
    del model, state
    torch.cuda.empty_cache()
(out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
