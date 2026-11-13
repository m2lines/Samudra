# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise concurrent CPU initialization with the production source readers."""

import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from samudra.skyline.model import ModelConfig, PointTransformer
from samudra.skyline.multitask import ComparisonSource

root = Path("/scratch/jr7309/skyline-large-classifier-20261112")
config = json.loads((root / "train.json").read_text())
rank = int(os.environ["LOCAL_RANK"])
torch.set_num_threads(4)
blosc.set_nthreads(1)
source = ComparisonSource(
    config["source"], 4, config["reference_root"], arm="all_fields"
)
hashes = []
for repeat in range(4):
    torch.manual_seed(config["seed"])
    model = PointTransformer(ModelConfig(**config["model"]))
    assert isinstance(model.head, torch.nn.Linear)
    assert torch.isfinite(model.head.weight).all()
    hashes.append(
        hashlib.sha256(model.head.weight.detach().numpy().tobytes()).hexdigest()
    )
    del model
    gc.collect()
assert len(set(hashes)) == 1
(root / f"cpu-initialization-rank-{rank}.json").write_text(
    json.dumps(dict(rank=rank, hashes=hashes)) + "\n"
)
source.close()
print(f"Rank {rank} completed four identical finite initializations", flush=True)
