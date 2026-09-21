# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SUBMIT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/submit_surface_adaptation.py")
)["submit"]


def setup_submission(tmp_path, monkeypatch, stage):
    wave1 = tmp_path / "wave1"
    for relative in [
        "ar/pretrain-best.pt",
        "ar/joint-best.pt",
        "initializer/initializer-best.pt",
    ]:
        path = wave1 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    layer, wrapper = tmp_path / "layer", tmp_path / "wrapper"
    layer.touch()
    wrapper.touch()
    qualification = tmp_path / "qualification"
    for name, arm in [("A", "initializer"), ("B", "evolution"), ("C", "joint")]:
        directory = qualification / name
        directory.mkdir(parents=True)
        (directory / "COMPLETE.json").write_text(json.dumps({"arm": arm}))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "code_commit": "qualified",
                    "arguments": {"max_steps": 30},
                    "world_size": 4,
                }
            )
        )
    for name in ["GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"]:
        monkeypatch.setenv(name, "test-placeholder")
    calls = []

    def record(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(stdout=str(100 + len(calls)))

    monkeypatch.setattr("subprocess.run", record)
    args = SimpleNamespace(
        root=str(tmp_path / "outputs"),
        wave1_root=str(wave1),
        code_layer=str(layer),
        wrapper=str(wrapper),
        qualification_root=str(qualification),
        code_commit="qualified",
        container_hash="container",
        deadline="2099-01-01T00:00:00Z",
        stage=stage,
        after_jobs=[],
        control_seed=1729,
    )
    return args, calls


def test_production_submits_exactly_four_arms_with_two_dependency_chains(
    tmp_path, monkeypatch
):
    args, calls = setup_submission(tmp_path, monkeypatch, "production")
    SUBMIT(args)
    assert len(calls) == 4
    assert "--dependency=afterok:101" in calls[2][0]
    assert "--dependency=afterok:102" in calls[3][0]
    assert all("--gres=gpu:rtx6000:4" in command for command, _ in calls)
    jobs = json.loads((Path(args.root) / "jobs.json").read_text())
    assert set(jobs) == {"A", "B", "C", "D"}


def test_rate_control_uses_higher_rate_and_waits_for_primary_jobs(
    tmp_path, monkeypatch
):
    args, calls = setup_submission(tmp_path, monkeypatch, "rate-control")
    with pytest.raises(ValueError, match="requires dependencies"):
        SUBMIT(args)
    assert not calls
    args.after_jobs = ["103", "104"]
    SUBMIT(args)
    assert len(calls) == 1
    assert "--dependency=afterok:103:104" in calls[0][0]
    arguments = json.loads((Path(args.root) / "jobs.json").read_text())["E"][
        "module_args"
    ]
    assert arguments[arguments.index("--learning-rate") + 1] == "1e-4"
    assert arguments[arguments.index("--seed") + 1] == "1729"


def test_code_mismatch_prevents_any_production_submission(tmp_path, monkeypatch):
    args, calls = setup_submission(tmp_path, monkeypatch, "production")
    args.code_commit = "different"
    with pytest.raises(ValueError, match="Qualification mismatch"):
        SUBMIT(args)
    assert not calls


def test_alternate_rate_control_uses_matched_second_order(tmp_path, monkeypatch):
    args, calls = setup_submission(tmp_path, monkeypatch, "rate-control")
    args.control_seed = 1730
    args.after_jobs = ["103", "104"]
    SUBMIT(args)
    jobs = json.loads((Path(args.root) / "jobs.json").read_text())
    assert set(jobs) == {"F"}
    assert jobs["F"]["seed"] == 1730
    assert len(calls) == 1
