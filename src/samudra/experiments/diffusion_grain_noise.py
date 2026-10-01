# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate only the initial sampling draws; no model inference or training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.diffusion_endpoint_maps import CHANNELS
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    protocols = [
        json.loads(
            (root / "reports/grain-diagnostics-v1" / mode / "protocol.json").read_text()
        )
        for mode in ["pretrained", "adapted"]
    ]
    for protocol in protocols:
        if protocol["seed"] != 4041729 or protocol["members"] != 8:
            raise ValueError("Unexpected sampling contract")
        if protocol["signature"].get("noise_correlation", 0) != 0 or protocol[
            "signature"
        ].get("paired_noise_draws", False):
            raise ValueError("Expected original white-noise model")
    grid = np.load(root / "data/observations/grid.npz")
    channels = [list(grid["names"]).index(c) for c in CHANNELS]
    mask = torch.tensor(grid["mask"], device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(4041729)
    selected = []
    for _ in range(8):
        noise = diffusion_noise((1, 154, *mask.shape[-2:]), mask.repeat(2, 1, 1), rng)
        selected.append(
            noise.reshape(1, 2, 77, *mask.shape[-2:])[0, -1, channels].cpu().numpy()
        )
    path = root / "reports/grain-diagnostics-v1/initial-noise.npz"
    np.savez_compressed(
        path, noise=np.array(selected), channels=np.array(CHANNELS), seed=4041729
    )
    atomic_json(
        dict(
            sha256=digest(path),
            seed=4041729,
            members=8,
            scope="Unscaled initial Gaussian fields; identical draws across all solver step counts, dates and checkpoints",
        ),
        path.with_suffix(".json"),
    )


if __name__ == "__main__":
    main()
