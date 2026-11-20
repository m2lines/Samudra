# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check real input masking, full-shape CUDA backward, and empty-layer loss safety."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from samudra.config import TrainConfig
from samudra.datasets import InferenceDataset
from samudra.rust_reader import RustIoRuntime
from samudra.utils.distributed import set_seed
from samudra.utils.loss import DynamicLoss, loss_fn_from_metric

p = argparse.ArgumentParser()
p.add_argument("config", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
cfg = TrainConfig.from_yaml(a.config)
set_seed(15)
RustIoRuntime(8)
bundle = cfg.data.build(cfg.experiment.resolved_data_root, include_inference=True)
source = bundle.train_sources[0]
layout = bundle.data_layout
n = len(layout.prognostic_var_names)
dataset = InferenceDataset(
    source,
    layout.prognostic_var_names,
    layout.boundary_var_names,
    2,
    2,
    True,
    0.0,
    False,
)
x, b, target = dataset[0]
x = x.cuda().expand(cfg.batch_size, -1, -1, -1).contiguous()
b = b.cuda().expand(cfg.batch_size, -1, -1, -1).contiguous()
target = target.cuda().expand(cfg.batch_size, -1, -1, -1).contiguous()
ctx = dataset.ctx.to(torch.device("cuda"))
assert (
    torch.isfinite(x).all() and torch.isfinite(b).all() and torch.isfinite(target).all()
)
assert torch.count_nonzero(x[:, ~ctx.label_mask]) == 0
assert torch.count_nonzero(target[:, ~ctx.label_mask]) == 0
net = cfg.model.build(
    2 * n, 2 * len(layout.boundary_var_names), 2 * n, 2, [source.grid_size]
).cuda()
initial_hash = hashlib.sha256()
for v in net.state_dict().values():
    initial_hash.update(v.detach().cpu().numpy().tobytes())
loss_fn = DynamicLoss(
    loss_fn_from_metric("mse"), limit=20, device=torch.device("cuda"), num_channels=n
)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
y = x
loss = torch.zeros((), device="cuda")
for _ in range(4):
    y = net.forward_once(y, b, ctx)
    assert torch.count_nonzero(y[:, ~ctx.label_mask]) == 0
    per_channel = loss_fn(y, target, ctx)
    loss_fn.update(per_channel)
    loss = loss + per_channel.mean() / 4
assert torch.isfinite(loss)
loss.backward()
assert all(
    p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
)
torch.cuda.synchronize()
result = dict(
    config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
    initial_model_sha256=initial_hash.hexdigest(),
    parameters=sum(p.numel() for p in net.parameters()),
    finite_loss=float(loss),
    seconds=time.perf_counter() - t0,
    torch=torch.__version__,
    gpu=torch.cuda.get_device_name(),
    peak_bytes=torch.cuda.max_memory_allocated(),
    empty_channels=(source.masks.prognostic.sum(dim=(-1, -2)) == 0)
    .nonzero()
    .flatten()
    .tolist(),
    scope="actual source input/target masking; four full-shape autoregressive calls and backward; dynamic loss including empty channels",
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
