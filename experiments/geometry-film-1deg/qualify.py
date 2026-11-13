# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Full-shape CUDA four-step backward qualification and compute-only timing."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from samudra.config import SamudraConfig, TrainConfig
from samudra.rust_reader import RustIoRuntime
from samudra.utils.ctx import BatchGrid

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("output")
a = p.parse_args()
cfg = TrainConfig.from_yaml(a.config)
assert isinstance(cfg.model, SamudraConfig)
layout = cfg.data.sources[0].build_layout()
RustIoRuntime(
    8
)  # Fail before training if the installed native extension is incompatible.
data = xr.open_zarr(Path(str(cfg.experiment.resolved_data_root)) / "OM4.zarr")
lat, lon = torch.tensor(data.y.values), torch.tensor(data.x.values)
shape = (len(lat), len(lon))
assert shape == (180, 360)
torch.manual_seed(15)
net = cfg.model.build(
    2 * len(layout.prognostic_var_names),
    2 * len(layout.boundary_var_names),
    2 * len(layout.prognostic_var_names),
    2,
    [shape],
).cuda()
x = torch.randn(
    cfg.batch_size, 2 * len(layout.prognostic_var_names), *shape, device="cuda"
)
b = torch.randn(
    cfg.batch_size, 2 * len(layout.boundary_var_names), *shape, device="cuda"
)
ctx = BatchGrid(
    torch.ones(x.shape[1], *shape, dtype=torch.bool, device="cuda"),
    (lat, lon),
    (lat, lon),
)
timings = []
torch.cuda.reset_peak_memory_stats()
for i in range(8):
    net.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = x
    loss = torch.zeros((), device="cuda")
    for _ in range(4):
        y = net.forward_once(y, b, ctx)
        loss = loss + y.square().mean() / 4
    assert torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )
    if i >= 3:
        timings.append(elapsed)
result = dict(
    arm=cfg.model.unet.geometry_mode,
    gpu=torch.cuda.get_device_name(),
    torch=torch.__version__,
    parameters=sum(p.numel() for p in net.parameters()),
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    batch_size=cfg.batch_size,
    rollout_calls=4,
    samples_seconds=timings,
    median_seconds=float(np.median(timings)),
    scope="synthetic forward/backward; excludes data I/O, optimizer and DDP",
)
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
