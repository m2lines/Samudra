# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

root = Path("/scratch/jr7309/runs/current-sd-seeds16-17/diagnostics")
if len(sys.argv) == 1:
    results = []
    for mode in ["original", "random", "channels_last", "fp32", "no_cudnn"]:
        r = subprocess.run(
            [sys.executable, __file__, mode],
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(mode, r.returncode, r.stdout, r.stderr, flush=True)
        results.append(
            dict(mode=mode, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        )
        (root / "conv-suite-results.json").write_text(json.dumps(results, indent=2))
    sys.exit(0)
import torch
import torch.nn.functional as F

mode = sys.argv[1]
d = torch.load(root / "failing-conv.pt", map_location="cpu", weights_only=False)
print(
    dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),
        mode=mode,
        shape=list(d["input"].shape),
        autocast=d["autocast"],
        dtype=str(d["autocast_dtype"]),
        finite_input=bool(d["input"].isfinite().all()),
        finite_weight=bool(d["weight"].isfinite().all()),
    ),
    flush=True,
)
torch.backends.cudnn.benchmark = True
x = d["input"].cuda().requires_grad_()
w = d["weight"].cuda().requires_grad_()
b = d["bias"].cuda().requires_grad_() if d["bias"] is not None else None
if mode == "random":
    x = torch.randn_like(x, requires_grad=True)
    w = torch.randn_like(w, requires_grad=True)
if mode == "channels_last":
    x = x.contiguous(memory_format=torch.channels_last)
    w = w.contiguous(memory_format=torch.channels_last)
if mode == "no_cudnn":
    torch.backends.cudnn.enabled = False
with torch.autocast(
    "cuda", dtype=d["autocast_dtype"], enabled=d["autocast"] and mode != "fp32"
):
    y = F.conv2d(
        x,
        w,
        b,
        stride=d["stride"],
        padding=d["padding"],
        dilation=d["dilation"],
        groups=d["groups"],
    )
    torch.cuda.synchronize()
    print("FORWARD_OK", flush=True)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    print("BACKWARD_OK", flush=True)
