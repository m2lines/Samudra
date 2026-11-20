# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Use all four allocated GPUs for four independent held-out forecast origins."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("arm")
a = p.parse_args()
checkpoint = a.root / "output" / a.arm / "saved_nets/ema_ckpt.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["epoch"] == 70, state["epoch"]
del state
children = []
for gpu, year in enumerate([2015, 2017, 2019, 2021]):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = (a.root / a.arm / f"eval-{year}.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "samudra.eval",
            str(a.root / a.arm / f"eval-{year}.yaml"),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    children.append((proc, log, year))
failures = []
for proc, log, year in children:
    rc = proc.wait()
    log.close()
    if rc:
        failures.append((year, rc))
if failures:
    raise RuntimeError(f"Evaluation failures: {failures}")
subprocess.run(
    [sys.executable, str(Path(__file__).with_name("score.py")), str(a.root), a.arm],
    check=True,
)
