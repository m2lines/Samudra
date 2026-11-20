# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare final sampler and frame-stream states with the original regressor."""

import json
from pathlib import Path

import numpy as np
import torch

from samudra.skyline.baseline import file_sha256

root = Path("/scratch/jr7309/skyline-large-classifier-20261112")
reference = Path("/scratch/jr7309/skyline-comparison-20261106/all_fields-25k")
new = torch.load(
    root / "all-fields-25k/latest.pt", map_location="cpu", mmap=True, weights_only=False
)
old = torch.load(
    reference / "latest.pt", map_location="cpu", mmap=True, weights_only=False
)
assert new["step"] == old["step"] == 25000
assert new["source"] == old["source"]
assert new["ema"]["num_updates"] == 25000
for left, right in zip(new["rank_states"], old["rank_states"], strict=True):
    np.testing.assert_equal(left["rng"], right["rng"])
    np.testing.assert_equal(left["stream"], right["stream"])


def training_records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["step"]: row for row in rows if "importance_mean" in row}


actual = training_records(root / "all-fields-25k/training.jsonl")
original = training_records(reference / "training.jsonl")
shared = sorted(actual.keys() & original.keys())
statistics = {}
for key in ("importance_mean", "importance_ess_fraction", "accepted_examples"):
    statistics[key] = max(
        abs(actual[step][key] - original[step][key]) for step in shared
    )
receipt = dict(
    step=25000,
    all_four_sampler_rng_states_equal=True,
    all_four_frame_stream_states_equal=True,
    source_receipts_equal=True,
    shared_logged_updates=len(shared),
    maximum_logged_statistic_difference=statistics,
    checkpoint_sha256=file_sha256(root / "all-fields-25k/latest.pt"),
)
(root / "final-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
config = json.loads((root / "evaluation-template.json").read_text())
config["classifier"]["sha256"] = receipt["checkpoint_sha256"]
(root / "evaluation.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
