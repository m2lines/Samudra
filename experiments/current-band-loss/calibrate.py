# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit band energies on16fixed training dates with the exact model preprocessing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from samudra.config import TrainConfig
from samudra.utils.band_loss import BandLoss, masked_highpass
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import BatchPreprocessor
from samudra.utils.location import LocalLocation

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
cfg = TrainConfig.from_yaml(a.config)
cfg.experiment.data_root = LocalLocation(path=a.data)
bundle = cfg.data.build(cfg.experiment.resolved_data_root)
source = bundle.train_sources[0]
layout = bundle.data_layout
names = list(layout.prognostic_var_names)
pre = BatchPreprocessor(
    source,
    names,
    layout.boundary_var_names,
    normalize_before_mask=cfg.data.normalize_before_mask,
    masked_fill_value=cfg.data.masked_fill_value,
)
valid = np.array(
    [i for i, t in enumerate(source.time.values) if str(t)[:10] < "2013-10-05"]
)
indices = np.random.default_rng(2718).choice(valid, size=16, replace=False)
energies = []
first = None
for i in indices:
    raw = torch.as_tensor(source.read(np.array([i]), names))
    values = pre.prepare_planes(raw, torch.device("cpu"))
    hp = masked_highpass(values, source.masks.prognostic)
    energies.append(hp.double().square().mean((0, 2, 3)).numpy())
    if first is None:
        first = values
    print("CALIBRATION_DATE", int(i), str(source.time.values[i]), flush=True)
energy = np.mean(energies, axis=0)
scales = np.maximum(energy, 1e-6).astype("float32")
result = dict(
    scales=scales.tolist(),
    raw_energies=energy.tolist(),
    floor=1e-6,
    channels=names,
    dates=[str(source.time.values[i]) for i in indices],
    training_indices=indices.tolist(),
    seed=2718,
    method="Full-grid masked high-pass target mean-square in model-normalized units. Gaussian sigma1radius3,periodic longitude,replicated latitude.16training dates; separate process preserves training RNG.",
    dataset=str(a.data),
    mean=source.statistics(names).mean.tolist(),
    std=source.statistics(names).std.tolist(),
)
# CUDA loss/backprop smoke on real77-channel fields and two output times.
device = torch.device("cuda")
assert first is not None
x = first.repeat(1, 2, 1, 1).to(device).detach().requires_grad_(True)
ctx = BatchGrid(
    label_mask=source.masks.prognostic_for_steps(2).to(device),
    input_resolution_cpu=source.resolution,
    output_resolution_cpu=source.resolution,
)
loss = BandLoss(
    device=device,
    num_channels=len(names),
    band_scales=scales,
    band_weight=0.1,
)
target = x.detach() + 0.01 * torch.randn_like(x)
for _ in range(64):
    x.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        objective = loss(x, target, ctx).mean()
    objective.backward()
    assert x.grad is not None
    if not torch.isfinite(objective) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA band qualification failed")
result["cuda_smoke"] = dict(
    updates=64,
    loss=float(objective.detach()),
    peak_gb=torch.cuda.max_memory_allocated() / 1024**3,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    "BAND_CALIBRATION_OK", hashlib.sha256(a.output.read_bytes()).hexdigest(), flush=True
)
