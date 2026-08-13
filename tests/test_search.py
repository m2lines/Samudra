# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from samudra.search import SearchConfig, build_search
from samudra.search.executors import LocalExecutor, SlurmExecutor


def config(tmp_path: Path, *, executor: str = "local") -> SearchConfig:
    executor_config = {"type": "local", "output_dir": tmp_path / "runs"}
    if executor == "slurm":
        harness = tmp_path / "train.sbatch"
        harness.touch()
        executor_config = {
            "type": "slurm",
            "output_dir": tmp_path / "runs",
            "harness": harness,
            "account": "account",
            "partition": "gpu",
        }
    return SearchConfig.model_validate(
        {
            "name": "test-search",
            "algorithm": {"type": "successive_halving", "rungs": [1, 3]},
            "objective": {"metric": "validation_loss", "mode": "min"},
            "metrics": ["validation_loss", "train_loss"],
            "executor": executor_config,
            "allow_dirty": True,
            "candidates": [
                {"name": "anchor", "config": "anchor.yaml", "fixed": True},
                {"name": "a", "config": "a.yaml"},
                {"name": "b", "config": "b.yaml"},
                {"name": "c", "config": "c.yaml"},
            ],
        }
    )


def test_config_validates_objective_and_rungs(tmp_path):
    value = config(tmp_path).model_dump(mode="json")
    value["metrics"] = ["train_loss"]
    with pytest.raises(ValidationError, match="objective.metric"):
        SearchConfig.model_validate(value)

    value["metrics"] = ["validation_loss"]
    value["algorithm"]["rungs"] = [3, 1]
    with pytest.raises(ValidationError, match="strictly increasing"):
        SearchConfig.model_validate(value)


def test_executor_dictionary_is_an_explicit_extension_point(tmp_path):
    local = build_search(config(tmp_path))
    slurm = build_search(config(tmp_path, executor="slurm"))
    assert isinstance(local.executor, LocalExecutor)
    assert isinstance(slurm.executor, SlurmExecutor)


def test_start_snapshots_resolved_candidate_configs(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.executor.dry_run = True
    source = str(Path("tests/configs/train_default.yaml").resolve())
    for candidate in search_config.candidates:
        candidate.config = source
        candidate.args = ["--batch_size=1"]
    monkeypatch.setattr(
        "samudra.search.successive_halving._git_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "dirty": False,
            "package_version": "1.0",
        },
    )
    search = build_search(search_config)

    state_path = search.start()

    bundled = SearchConfig.from_yaml_and_cli([str(search.config_path)])
    assert state_path.is_file()
    assert all(not candidate.args for candidate in bundled.candidates)
    assert all(Path(candidate.config).is_file() for candidate in bundled.candidates)


def write_result(search, name: str, rung: int, validation: float) -> None:
    output = search.output_dir(name, rung)
    (output / "saved_nets").mkdir(parents=True)
    (output / "saved_nets/ckpt.pt").touch()
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "epoch": search.rungs[rung],
                "complete": True,
                "validation_loss": validation,
                "train_loss": validation + 0.1,
                "optimizer_steps": 10,
                "executor": "local",
                "code_commit": "f" * 40,
            }
        ),
        encoding="utf-8",
    )


def test_successive_halving_promotes_with_dataframe_and_reports_metrics(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search = build_search(search_config)
    search.search_dir.mkdir(parents=True)
    search.config_path.write_text("test", encoding="utf-8")
    state = {
        "name": search_config.name,
        "status": "running",
        "provenance": {"commit": "f" * 40},
        "anchors": {"candidates": ["anchor"], "results": []},
        "rungs": [
            {
                "index": 0,
                "epochs": 1,
                "candidates": ["a", "b", "c"],
                "results": [],
                "promoted": [],
                "advanced": False,
            },
            {
                "index": 1,
                "epochs": 3,
                "candidates": [],
                "results": [],
                "promoted": [],
                "advanced": False,
            },
        ],
    }
    search.write_state(state)
    for name, score in {"a": 0.3, "b": 0.1, "c": 0.2}.items():
        write_result(search, name, 0, score)
    submitted = []
    monkeypatch.setattr(
        search.executor,
        "submit_rung",
        lambda state, rung: submitted.append(
            (rung, state["rungs"][rung]["candidates"])
        ),
    )

    search.advance(0)

    updated = search.read_state()
    assert updated["rungs"][0]["promoted"] == ["b", "c"]
    assert submitted == [(1, ["b", "c"])]
    report = search.results_path.read_text(encoding="utf-8")
    assert "validation_loss" in report
    assert "train_loss" in report


def test_local_executor_runs_tasks_then_advances(tmp_path, monkeypatch):
    search = build_search(config(tmp_path))
    search.search_dir.mkdir(parents=True)
    state = {
        "status": "prepared",
        "anchors": {"candidates": []},
        "rungs": [{"candidates": ["a", "b"]}],
    }
    search.write_state(state)
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        search,
        "train_task",
        lambda rung, task, anchor: events.append(("train", rung, task, anchor)),
    )
    monkeypatch.setattr(
        search, "advance", lambda rung: events.append(("advance", rung))
    )

    search.executor.submit_rung(state, 0)

    assert events == [
        ("train", 0, 0, False),
        ("train", 0, 1, False),
        ("advance", 0),
    ]


def test_task_builds_train_config_and_calls_trainer_directly(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.candidates = [
        search_config.candidates[1].model_copy(
            update={"config": str(Path("tests/configs/train_default.yaml").resolve())}
        )
    ]
    search = build_search(search_config)
    search.search_dir.mkdir(parents=True)
    search.write_state(
        {
            "provenance": {"commit": "f" * 40},
            "anchors": {"candidates": []},
            "rungs": [{"candidates": ["a"]}, {"candidates": []}],
        }
    )
    received = []

    class FakeTrainer:
        def __init__(self, train_config):
            received.append(train_config)

        def run(self):
            received.append("run")

    monkeypatch.setattr("samudra.search.successive_halving.Trainer", FakeTrainer)
    monkeypatch.setattr(
        "samudra.search.successive_halving.handle_logging", lambda *args: None
    )
    monkeypatch.setattr(
        "samudra.search.successive_halving.handle_warnings", lambda: None
    )

    search.train_task(0, 0, anchor=False)

    train_config = received[0]
    assert train_config.epochs == 1
    assert train_config.experiment.search.candidate == "a"
    assert train_config.experiment.wandb.group == "test-search"
    assert "search" in train_config.experiment.wandb.tags
    assert received[1] == "run"


def test_slurm_executor_submits_array_and_automatic_advance(tmp_path, monkeypatch):
    search = build_search(config(tmp_path, executor="slurm"))
    search.search_dir.mkdir(parents=True)
    state = {
        "name": "test-search",
        "status": "prepared",
        "anchors": {"candidates": []},
        "rungs": [{"candidates": ["a", "b"]}],
    }
    search.write_state(state)
    commands = []

    def submit(command):
        commands.append(command)
        return str(len(commands))

    monkeypatch.setattr(
        search.executor,
        "_submit",
        submit,
    )

    search.executor.submit_rung(state, 0)

    assert "--array=0-1%2" in commands[0]
    assert any(value.startswith("--dependency=afterany:1") for value in commands[1])
    assert "samudra.search.worker" in commands[1][-1]
