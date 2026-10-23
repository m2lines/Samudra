# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit matched config and full-sized GPU forward/backward before training."""

import json
import sys
from pathlib import Path

import torch
import yaml

from samudra.config import SamudraConfig, TrainConfig
from samudra.utils.ctx import BatchGrid

p = Path(__file__).parent
control = yaml.safe_load((p / "control-train.yaml").read_text())
treatment = yaml.safe_load((p / "train.yaml").read_text())
expected = json.loads(json.dumps(control))
expected["model"]["persistence_gate"] = True
expected["experiment"]["name"] = treatment["experiment"]["name"]
expected["experiment"]["wandb"]["project"] = "tocean"
expected["post_train_eval"]["eval_config_path"] = treatment["post_train_eval"][
    "eval_config_path"
]
assert expected == treatment
cfg = TrainConfig.from_yaml(p / "train.yaml")
assert isinstance(cfg.model, SamudraConfig)
assert (
    cfg.model.persistence_gate and cfg.data.sources[0].boundary_vars_key == "tau_hfds"
)
if "--cpu" in sys.argv:
    print("CONFIG_AUDIT_PASSED")
    raise SystemExit()
torch.manual_seed(15)
model = cfg.model.build(
    prog_channels=154,
    boundary_channels=6,
    out_channels=154,
    input_steps=2,
    grid_sizes=[(180, 360)],
).cuda()
x = torch.randn(1, 154, 180, 360, device="cuda")
b = torch.randn(1, 6, 180, 360, device="cuda")
mask = torch.ones(154, 180, 360, dtype=torch.bool, device="cuda")
mask[:, -2:] = False
coords = (torch.arange(180), torch.arange(360))
ctx = BatchGrid(mask, coords, coords)
assert model.persistence_gate is not None
opt = torch.optim.Adam(model.parameters(), lr=6e-4)
for _ in range(2):
    opt.zero_grad()
    out = model.forward_once(x, b, ctx)
    out2 = model.forward_once(out, b, ctx)
    loss = out2.square().mean()
    loss.backward()
    for parameter in (
        model.persistence_gate.weight,
        model.decoder.weight,
        next(model.unet.parameters()),
    ):
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
    opt.step()
    assert torch.isfinite(out2).all() and not out2[:, :, -2:].count_nonzero()
record = dict(
    config_audit="passed",
    gpu=torch.cuda.get_device_name(),
    loss=float(loss.detach()),
    parameters=sum(x.numel() for x in model.parameters()),
    gate_parameters=sum(x.numel() for x in model.persistence_gate.parameters()),
    two_call_gradients="finite nonzero",
    torch=torch.__version__,
)
Path("QUALIFIED.json").write_text(json.dumps(record, indent=2) + "\n")
print(record)
