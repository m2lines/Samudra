# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Score candidate absolute temperatures without loading future ocean truth.

Input JSONL rows: origin (time index of newest history), row, col, depth
(zero based), temperature_c. Output retains these and appends bin probability,
physical-unit density, and support status. No anomaly threshold is inferred.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import OceanSource
from .model import ModelConfig, PointTransformer, Vocabulary
from .run import to_device


@torch.no_grad()
def score_values(model, vocab, batch, candidates_c, means, stds):
    depth = batch["depth"]
    if model.config.objective == "regression":
        raise ValueError("Scalar regression does not define a likelihood")
    device = batch["values"].device
    if not torch.isfinite(candidates_c).all():
        raise ValueError("Nonfinite candidate temperature")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(batch)
    normalized = (candidates_c - means[depth]) / stds[depth]
    if model.config.objective == "residual_ce":
        normalized = normalized - batch["current"]
    ids = vocab.encode(normalized, depth)
    outside = vocab.outside(normalized, depth)
    log_mass = logits.log_softmax(-1).gather(1, ids[:, None]).squeeze(1)
    log_density = vocab.log_density(logits, normalized, depth, stds)
    return {
        "bin": ids,
        "outside_support": outside,
        "log_probability": log_mass.masked_fill(outside, -torch.inf),
        "log_density_per_c": log_density,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    torch.set_num_threads(config["torch_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointTransformer(ModelConfig(**config["model"]))
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    vocab = Vocabulary(state["vocabulary"]["centers"][:, -1], model.config.bins)
    vocab.load_state_dict(state["vocabulary"], strict=True)
    vocab = vocab.to(device)
    del state
    source = OceanSource(
        config["source"], config["readers"], config.get("reference_root")
    )
    try:
        groups = defaultdict(list)
        for line in args.candidates.read_text().splitlines():
            row = json.loads(line)
            groups[int(row["origin"])].append(row)
        means = torch.tensor(source.mean[38:57], device=device)
        stds = torch.tensor(source.std[38:57], device=device)
        rng = np.random.default_rng(0)  # Explicit points use no random draws.
        with args.output.open("x") as output:
            for origin, records in sorted(groups.items()):
                frame = source.read(origin, with_target=False)
                for start in range(0, len(records), config["microbatch"]):
                    rows = records[start : start + config["microbatch"]]
                    depths = [int(row["depth"]) for row in rows]
                    points = [(int(row["row"]), int(row["col"])) for row in rows]
                    batch = to_device(
                        source.sample(
                            frame, rng, len(rows), config["crop"], depths, points
                        ),
                        device,
                    )
                    values = torch.tensor(
                        [row["temperature_c"] for row in rows],
                        dtype=torch.float32,
                        device=device,
                    )
                    scores = score_values(model, vocab, batch, values, means, stds)
                    for i, row in enumerate(rows):
                        outside = bool(scores["outside_support"][i])
                        result = {
                            **row,
                            "bin": int(scores["bin"][i]),
                            "outside_support": outside,
                            "log_probability": None
                            if outside
                            else float(scores["log_probability"][i]),
                            "log_density_per_c": None
                            if outside
                            else float(scores["log_density_per_c"][i]),
                        }
                        output.write(json.dumps(result, allow_nan=False) + "\n")
    finally:
        source.close()


if __name__ == "__main__":
    main()
