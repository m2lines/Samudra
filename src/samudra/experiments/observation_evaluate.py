# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out evaluation after validation-only checkpoint selection is complete."""

import argparse
import json
from pathlib import Path

import torch

from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import Pilot, atomic_json, digest
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if not (run / "TRAIN_COMPLETE.json").exists():
        raise ValueError("Selection must finish before held-out evaluation")
    manifest = json.loads((run / "manifest.json").read_text())
    checkpoint = run / "best.pt"
    best = json.loads((run / "best.json").read_text())
    if digest(checkpoint) != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    evaluator = Pilot.__new__(Pilot)
    evaluator.data = Samples(manifest["arguments"]["data"], "cuda")
    if manifest["arguments"].get("from_scratch", False):
        evaluator.data.use_observation_normalization()
    evaluator.model = ObservationTransfer(evaluator.data.grid["names"].tolist()).cuda()
    evaluator.model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=False)["model"],
        strict=True,
    )
    paths = evaluator.data.paths(args.split)
    expected = 96 if args.split == "test" else 9
    if len(paths) != expected:
        raise ValueError("Incomplete reporting cohort")
    result = evaluator.evaluate(paths, export=output / "selected-predictions.npz")
    atomic_json(
        {
            "selected_checkpoint": str(checkpoint),
            "sha256": digest(checkpoint),
            "validation_selection_score": best["score"],
            "split": args.split,
            "metrics": result,
        },
        output / "selected.json",
    )
    source = manifest["arguments"]["checkpoint"]
    evaluator.data = Samples(manifest["arguments"]["data"], "cuda")
    evaluator.model.load_core(
        torch.load(source, map_location="cpu", weights_only=False)["model"]
    )
    # Restore the initial zero-output adapter, independently of selected weights.
    final_adapter = evaluator.model.adapter[-1]
    assert isinstance(final_adapter, torch.nn.Conv2d) and final_adapter.bias is not None
    torch.nn.init.zeros_(final_adapter.weight)
    torch.nn.init.zeros_(final_adapter.bias)
    for label, persistence in [
        ("source-with-zero-forcing", False),
        ("source-inferred-persistence", True),
    ]:
        reference = evaluator.evaluate(
            paths, persistence=persistence, export=output / (label + ".npz")
        )
        atomic_json(reference, output / (label + ".json"))
    for label, options in [
        ("seasonal-climatology", {"climatology": True}),
        ("inferred-anomaly-persistence", {"anomaly": True}),
    ]:
        reference = evaluator.evaluate(
            paths, export=output / (label + ".npz"), **options
        )
        atomic_json(reference, output / (label + ".json"))
    atomic_json(
        {
            "split": args.split,
            "origins": expected,
            "source_sha256": digest(source),
            "selected_sha256": digest(checkpoint),
        },
        output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
