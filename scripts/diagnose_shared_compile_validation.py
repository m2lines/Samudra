#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Separate inference compilation from training-weight changes after a failed gate."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.root / args.source / "paths.json").read_text())
    path = (
        args.root
        / args.source
        / "qualification/A-shared-era5-8k/serialization-expected.pt"
    )
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=16,
    )
    data.cache_prepared = True
    state = torch.load(path, weights_only=False, map_location="cpu")
    model = MonthlyLatentOcean().cuda().eval()
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    if args.compile:
        model.decoder.core.forward = torch.compile(  # type: ignore[method-assign]
            model.decoder.core.forward, mode="default", fullgraph=False
        )
    result = evaluate(model, data, data.paths("validation")[:1])
    out = args.root / "inference-diagnosis"
    out.mkdir(exist_ok=True)
    name = args.source + ("-compiled" if args.compile else "-eager")
    (out / (name + ".json")).write_text(
        json.dumps(
            dict(
                source_sha256=sha, step=step, compiled=args.compile, validation=result
            ),
            indent=2,
        )
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
