# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Migration changes rank ownership explicitly without weakening normal restore."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "migration", Path(__file__).parents[1] / "scripts/migrate_observation_checkpoint.py"
)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


@pytest.fixture
def saved(monkeypatch):
    code = {"fixture.py": "test-only"}
    monkeypatch.setattr(
        migration,
        "AUDITED_CODE_DIGEST",
        hashlib.sha256(
            json.dumps(code, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    backbone = {"drop_path_rate": 0, "core_block": {"block_type": "conv_next_block"}}
    rng = {
        "python_rng": (3, (1,), None),
        "numpy_rng": ("test", np.arange(2)),
        "torch_rng": torch.arange(3, dtype=torch.uint8),
        "cuda_rng": [torch.arange(4, dtype=torch.uint8)],
        "cuda_rng_scope": "current",
    }
    result: dict[str, Any] = {
        key: {}
        for key in ("model", "optimizer", "ema", "exposure", "sampler_rng", "pools")
    }
    result.update(
        manifest={
            "code_sha256": code,
            "config": {
                "update_prefetch": True,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "model": {
                    "forcing_mode": "shared_era5",
                    **{
                        name: copy.deepcopy(backbone)
                        for name in (
                            "backbone",
                            "initializer_backbone",
                            "evolution_backbone",
                        )
                    },
                },
            },
        },
        updates=25,
        cooled=False,
        best_score=0.5,
        rank_rng=[copy.deepcopy(rng) for _ in range(4)],
        pending_plan={
            "update": 25,
            "source": "obs",
            "examples": [
                {
                    "origin": "2000-01",
                    "month": 0,
                    "coverage": None,
                    "seed": i,
                    "entry": None,
                }
                for i in range(8)
            ],
        },
        **copy.deepcopy(rng),
    )
    result["model"] = {"weight": torch.tensor([1.0, 2.0])}
    return result


def test_only_rank_mapping_changes_with_source_readback(tmp_path, saved):
    source, target = tmp_path / "parent.pt", tmp_path / "derived.pt"
    torch.save(saved, source)
    before = migration.digest(source)
    receipt = migration.migrate(source, target, [0, 1])
    actual = torch.load(target, weights_only=False)
    assert len(actual["rank_rng"]) == 2
    for key in saved.keys() - {"rank_rng"}:
        migration.equal_tree(saved[key], actual[key])
    assert migration.digest(source) == before == receipt["source_sha256"]
    assert receipt["changed_fields"] == ["rank_rng"]
    assert (
        receipt["manifest_unchanged"]
        and not receipt["cross_layout_bitwise_trajectory_claim"]
    )
    with pytest.raises(ValueError, match="new output"):
        migration.migrate(source, target, [0, 1])
    with pytest.raises(ValueError, match="new output"):
        migration.migrate(source, source, [0, 1])


@pytest.mark.parametrize(
    "mapping", [[], [0, 1, 2, 3], [0, 0], [1, 2], [0, 4], [0], [0, 2]]
)
def test_reject_invalid_rank_mapping(saved, mapping):
    with pytest.raises(ValueError):
        migration.migrated_payload(saved, mapping)


def test_reject_unknown_runtime_stochastic_model_and_unseeded_plan(saved):
    saved["manifest"]["code_sha256"]["new.py"] = "changed"
    with pytest.raises(ValueError, match="not been audited"):
        migration.migrated_payload(saved, [0, 1])
    del saved["manifest"]["code_sha256"]["new.py"]
    saved["manifest"]["config"]["model"]["initializer_backbone"]["drop_path_rate"] = 0.1
    with pytest.raises(ValueError, match="stochastic drop path"):
        migration.migrated_payload(saved, [0, 1])
    saved["manifest"]["config"]["model"]["initializer_backbone"]["drop_path_rate"] = 0
    del saved["pending_plan"]["examples"][0]["seed"]
    with pytest.raises(ValueError, match="example fields"):
        migration.migrated_payload(saved, [0, 1])
