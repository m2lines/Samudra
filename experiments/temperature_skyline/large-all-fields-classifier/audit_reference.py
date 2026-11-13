# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify the saved reference and seeded initialization inside the run container."""

import hashlib
import json
from pathlib import Path

import torch

from samudra.skyline.baseline import file_sha256
from samudra.skyline.comparison_train import parameter_digest
from samudra.skyline.model import ModelConfig, PointTransformer

ROOT = Path("/scratch/jr7309/skyline-large-classifier-20261112")
REFERENCE = Path("/scratch/jr7309/skyline-comparison-20261106/all_fields-25k")


def nonhead_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not name.startswith("head."):
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    provenance = json.loads((REFERENCE / "provenance-restart-0.json").read_text())
    state = torch.load(
        REFERENCE / "latest.pt", map_location="cpu", mmap=True, weights_only=False
    )
    assert state["config"] == provenance["config"]
    assert state["source"] == provenance["source"]
    assert state["step"] == 25000 and state["ema"]["num_updates"] == 25000
    assert len(state["rank_states"]) == 4
    config = json.loads((ROOT / "train.json").read_text())
    differences = {
        key: {"reference": provenance["config"].get(key), "classifier": value}
        for key, value in config.items()
        if value != provenance["config"].get(key)
    }
    torch.manual_seed(config["seed"])
    regression = PointTransformer(ModelConfig(**provenance["config"]["model"]))
    assert parameter_digest(regression) == provenance["initial_parameter_sha256"]
    original = nonhead_digest(regression)
    del regression
    torch.manual_seed(config["seed"])
    classifier = PointTransformer(ModelConfig(**config["model"]))
    assert original == nonhead_digest(classifier)
    receipt = dict(
        reference_checkpoint_sha256=file_sha256(REFERENCE / "latest.pt"),
        reference_source=provenance["source_commit"],
        reference_step=state["step"],
        reference_initial_hash=provenance["initial_parameter_sha256"],
        nonhead_initial_hash=original,
        classifier_parameters=sum(p.numel() for p in classifier.parameters()),
        config_differences=differences,
        source=state["source"],
    )
    (ROOT / "reference-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Saved reference and initial parameters verified", flush=True)


if __name__ == "__main__":
    main()
