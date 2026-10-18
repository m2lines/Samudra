#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only paired raw/EMA and conditioning sensitivity diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from samudra.experiments.monthly_forcing_adapter import ModelForcingAdapter
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_training import Samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--producer", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((a.source / "clip1/protocol.json").read_text())
    data = Samples(
        protocol["observations"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=256,
    )
    data.cache_prepared = True
    paths = data.paths("validation")
    results = {}
    for arm in ["clip1", "clip5", "adapter"]:
        for step in [2000, 3000]:
            path = a.source / arm / f"endpoint-{step:05d}.pt"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["step"] == step and checkpoint["protocol"]["seed"] == 1730
            model = MonthlyLatentOcean().cuda()
            if arm == "adapter":
                model.add_module("model_forcing_adapter", ModelForcingAdapter().cuda())
            control = evaluate(model, data, paths, climatology=True)
            for weights in ["model", "ema"]:
                model.load_state_dict(checkpoint[weights])
                modes = [None] + (
                    ["hold_latent", "shuffle_history", "zero_forcing"]
                    if step == 3000 and weights == "ema"
                    else []
                )
                for mode in modes:
                    key = f"{arm}-{step}-{weights}-{mode or 'standard'}"
                    metrics = evaluate(
                        model,
                        data,
                        paths,
                        diagnostic=mode,
                        prediction_path=a.out / f"{key}.npz",
                    )
                    if weights == "ema" and mode is None:
                        reference = json.loads(
                            (a.source / arm / f"validation-{step:05d}.json").read_text()
                        )
                        if json.dumps(metrics, sort_keys=True) != json.dumps(
                            reference, sort_keys=True
                        ):
                            raise ValueError(
                                "Saved EMA validation did not reproduce exactly: " + key
                            )
                    record = dict(
                        metrics=metrics,
                        blend=selection_score(metrics, control, [], metric="rmse-only"),
                        checkpoint_sha256=sha,
                        producer=a.producer,
                    )
                    (a.out / f"{key}.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
                    results[key] = record
                    print(json.dumps(dict(key=key, blend=record["blend"])), flush=True)
            del model, checkpoint
    (a.out / "COMPLETE.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
