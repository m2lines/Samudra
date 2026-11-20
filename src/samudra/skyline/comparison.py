# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched validation of selected skyline checkpoints and frozen Samudra."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .baseline import file_sha256
from .model import ModelConfig, PointTransformer, Vocabulary
from .run import evaluate


def compare(config, source, output):
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / "baseline"
    # Separate process releases the full-grid baseline model before loading the
    # point specialists. It uses the published evaluator's checkpoint['model'].
    subprocess.run(
        [
            sys.executable,
            "-m",
            "samudra.skyline.baseline",
            "--config",
            config["sampling_config"],
            "--baseline-config",
            config["baseline_config"],
            "--checkpoint",
            config["baseline_checkpoint"],
            "--output",
            str(baseline),
        ],
        check=True,
    )
    results = []
    reference_locations = None
    device = torch.device("cuda")
    for selection in config["selections"]:
        checkpoint = Path(selection["checkpoint"])
        state = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        model_config = ModelConfig(**state["config"]["model"])
        model = PointTransformer(model_config).to(device)
        model.load_state_dict(state["model"], strict=True)
        vocab = Vocabulary(np.ones(19), model_config.bins).to(device)
        vocab.load_state_dict(state["vocabulary"], strict=True)
        evaluation_config = dict(state["config"], baseline_cache=str(baseline))
        destination = output / selection["label"]
        destination.mkdir(exist_ok=True)
        result = evaluate(
            model, vocab, source, evaluation_config, device, destination, state["step"]
        )
        predictions = destination / f"predictions-{state['step']:07d}.jsonl"
        locations = [
            (r["origin"], r["row"], r["col"], r["depth"], r["truth_c"])
            for r in map(json.loads, predictions.read_text().splitlines())
        ]
        if reference_locations is None:
            reference_locations = locations
        elif locations != reference_locations:
            raise ValueError("Selected checkpoints did not evaluate identical targets")
        receipt = {
            **selection,
            "checkpoint_sha256": file_sha256(checkpoint),
            "step": state["step"],
            "sampling_config": evaluation_config,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "metrics": result,
        }
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        results.append(receipt)
        print(
            json.dumps(
                {
                    "selection": selection["label"],
                    "step": state["step"],
                    "nmse": result["mean_depth_nmse"],
                }
            ),
            flush=True,
        )
        del model, vocab, state
        torch.cuda.empty_cache()
    (output / "COMPLETE.json").write_text(
        json.dumps({"selections": results}, indent=2) + "\n"
    )
