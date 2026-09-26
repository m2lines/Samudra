# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "continuation",
    Path(__file__).parents[1] / "scripts/prepare_observation_continuation.py",
)
assert spec and spec.loader
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "random"
    root.mkdir()
    args = dict(
        from_scratch=True,
        fixed_updates=True,
        joint_steps=8000,
        seed=1729,
        accumulate=8,
        output=str(root),
        name="random",
        milestone_steps=[4000, 6000, 8000],
        core_lr=1e-4,
        warmup_steps=50,
        validate_every=1000,
    )
    write(
        root / "manifest.json",
        {"arguments": args, "code_commit": "pinned", "data_manifest_sha256": "fixed"},
    )
    model = {"weight": torch.tensor([1.0])}
    torch.save({"model": model}, root / "best.pt")
    checksum = continuation.digest(root / "best.pt")
    write(root / "best.json", {"step": 7000, "checkpoint_sha256": checksum})
    write(root / "TRAIN_COMPLETE.json", {"best_checkpoint_sha256": checksum})
    write(root / "joint-complete.json", {"step": 8000})
    e = tmp_path / "random-evaluation"
    e.mkdir()
    write(e / "COMPLETE.json", {"origins": 96, "selected_sha256": checksum})
    torch.save(
        {
            "model": model,
            "state": {"step": 8000},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.3])}}},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.tensor([3], dtype=torch.uint8),
        },
        root / "joint-last.pt",
    )
    for step in (4000, 6000, 8000):
        torch.save({"model": model, "step": step}, root / f"joint-{step:05d}.pt")
        (root / f"joint-{step:05d}-best.pt").write_bytes(
            (root / "best.pt").read_bytes()
        )
        write(
            root / f"joint-{step:05d}-best.json",
            {"step": min(step, 7000), "checkpoint_sha256": checksum},
        )
    for name in ("joint-best.pt", "reconstruction-best.pt"):
        (root / name).write_bytes((root / "best.pt").read_bytes())
    for name in (
        "reconstruction-complete.json",
        "selection-reference.json",
        "baseline-validation.json",
    ):
        write(root / name, {"unchanged": True})
    (root / "events.jsonl").write_text('{"step":8000}\n')
    return root


def test_continuation_preserves_optimizer_rng_and_completed_parent(source):
    before = {p.name: continuation.digest(p) for p in source.iterdir()}
    target = source.with_name("random-16000")
    record = continuation.prepare(source, target)
    assert continuation.digest(source / "joint-last.pt") == continuation.digest(
        target / "joint-last.pt"
    )
    assert before == {p.name: continuation.digest(p) for p in source.iterdir()}
    assert not (target / "joint-complete.json").exists()
    assert not (target / "TRAIN_COMPLETE.json").exists()
    assert (target / "reconstruction-complete.json").exists()
    a = continuation.read(source / "manifest.json")
    b = continuation.read(target / "manifest.json")
    changed = {k for k in a["arguments"] if a["arguments"][k] != b["arguments"][k]}
    assert changed == {"output", "name", "joint_steps", "milestone_steps"}
    assert record["resume_step"] == 8000 and b["arguments"]["joint_steps"] == 16000
    with pytest.raises(FileExistsError):
        continuation.prepare(source, target)


def test_continuation_rejects_terminal_weight_mismatch(source):
    torch.save(
        {"model": {"weight": torch.tensor([2.0])}, "step": 8000},
        source / "joint-08000.pt",
    )
    with pytest.raises(ValueError, match="Resume weights differ"):
        continuation.prepare(source, source.with_name("random-16000"))


def test_continuation_requires_finished_evaluation(source):
    write(
        source.parent / "random-evaluation/COMPLETE.json",
        {"origins": 9, "selected_sha256": "wrong"},
    )
    with pytest.raises(ValueError, match="held-out evaluation"):
        continuation.prepare(source, source.with_name("random-16000"))
