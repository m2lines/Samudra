# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from samudra.search import SearchConfig
from samudra.search.config import ArtifactConfig
from samudra.search.executors import LocalExecutor, SlurmExecutor
from samudra.utils.training_summary import write_search_metrics


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
            "apptainer_module": "singularity-ce/4.3.3",
        }
    return SearchConfig.model_validate(
        {
            "name": "test-search",
            "run_id": "test-search--run",
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


def test_config_rejects_candidate_names_with_colliding_resource_slugs(tmp_path):
    value = config(tmp_path).model_dump(mode="json")
    value["candidates"][1]["name"] = "same name"
    value["candidates"][2]["name"] = "same-name"

    with pytest.raises(ValidationError, match="unique after slug normalization"):
        SearchConfig.model_validate(value)


def test_executor_dictionary_is_an_explicit_extension_point(tmp_path):
    local = config(tmp_path).build()
    slurm = config(tmp_path, executor="slurm").build()
    assert isinstance(local.executor, LocalExecutor)
    assert isinstance(slurm.executor, SlurmExecutor)


def test_search_generates_a_readable_run_id_once(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.run_id = None
    monkeypatch.setattr(
        "samudra.search.successive_halving._new_run_id",
        lambda name: f"{name}--20260813T192612.123456Z",
    )

    search = search_config.build()

    assert search.run_id == "test-search--20260813T192612.123456Z"
    assert search_config.run_id == search.run_id
    assert search.search_dir.name == search.run_id


def test_immutable_environment_uses_verified_commit_provenance(monkeypatch):
    from samudra.search.successive_halving import _git_provenance

    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "A" * 40)
    provenance = _git_provenance(allow_dirty=False)

    assert provenance["commit"] == "a" * 40
    assert provenance["dirty"] is False


def test_immutable_environment_rejects_ambiguous_commit(monkeypatch):
    from samudra.search.successive_halving import _git_provenance

    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "main")

    with pytest.raises(ValueError, match="full 40-character Git SHA"):
        _git_provenance(allow_dirty=False)


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
    search = search_config.build()

    state_path = search.start()

    bundled = SearchConfig.from_yaml_and_cli([str(search.config_path)])
    assert state_path.is_file()
    assert all(not candidate.args for candidate in bundled.candidates)
    assert all(Path(candidate.config).is_file() for candidate in bundled.candidates)


def test_local_artifact_publisher_writes_queryable_research_record(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search_config.executor.dry_run = True
    search_config.artifacts = ArtifactConfig.model_validate(
        {
            "destination": {"type": "local", "path": tmp_path / "published"},
            "checkpoints": "final",
            "public_url": "https://example.test/experiments",
        }
    )
    source = str(Path("tests/configs/train_default.yaml").resolve())
    for candidate in search_config.candidates:
        candidate.config = source
    monkeypatch.setattr(
        "samudra.search.successive_halving._git_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "dirty": False,
            "package_version": "1.0",
        },
    )
    search = search_config.build()
    search.start()
    published = tmp_path / "published/test-search--run"
    published_state = json.loads((published / "state.json").read_text(encoding="utf-8"))
    assert published_state["status"] == "running"
    output = search.output_dir("a", 1)
    (output / "saved_nets").mkdir(parents=True)
    (output / "saved_nets/ckpt.pt").write_bytes(b"checkpoint")
    (output / "config.yaml").write_text("epochs: 3\n", encoding="utf-8")
    (output / "experiment.log").write_text("trained\n", encoding="utf-8")
    (output / "analysis").mkdir()
    (output / "analysis/loss.png").write_bytes(b"figure")
    write_search_metrics(output, {"candidate": "a", "epoch": 1, "loss": 0.5})
    write_search_metrics(output, {"candidate": "a", "epoch": 2, "loss": 0.25})
    state = search.read_state()
    state["rungs"][1]["results"] = [
        {
            "candidate": "a",
            "rung": 1,
            "epochs": 3,
            "eligible": True,
            "validation_loss": 0.25,
            "train_loss": 0.2,
            "output_dir": str(output),
        }
    ]
    search._write_results(state)

    search.publish(state)

    assert (published / "results.parquet").is_file()
    assert (published / "epochs.parquet").is_file()
    assert (published / f"runs/{output.name}/saved_nets/ckpt.pt").is_file()
    epochs = pd.read_parquet(published / "epochs.parquet")
    assert epochs[["epoch", "loss"]].to_dict("records") == [
        {"epoch": 1, "loss": 0.5},
        {"epoch": 2, "loss": 0.25},
    ]
    catalog = pd.read_parquet(published / "artifacts.parquet")
    checkpoint = catalog[catalog["artifact"].str.endswith("ckpt.pt")].iloc[0]
    assert checkpoint["sha256"]
    assert checkpoint["public_url"].startswith(
        "https://example.test/experiments/test-search--run/"
    )
    figure = catalog[catalog["artifact"].str.endswith("loss.png")].iloc[0]
    assert figure["kind"] == "figure"
    assert figure["candidate"] == "a"
    report = catalog[catalog["artifact"] == "analysis/report.md"].iloc[0]
    assert report["kind"] == "report"
    assert report["media_type"] == "text/markdown"


def test_start_records_and_publishes_submission_failure(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.executor.dry_run = True
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    source = str(Path("tests/configs/train_default.yaml").resolve())
    for candidate in search_config.candidates:
        candidate.config = source
    monkeypatch.setattr(
        "samudra.search.successive_halving._git_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "dirty": False,
            "package_version": "1.0",
        },
    )
    search = search_config.build()
    monkeypatch.setattr(search.executor, "submit_anchors", lambda state: None)

    def fail_submission(state, rung):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(search.executor, "submit_rung", fail_submission)

    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        search.start()

    local_state = search.read_state()
    published_state = json.loads(
        (tmp_path / "published/test-search--run/state.json").read_text(encoding="utf-8")
    )
    assert local_state["status"] == "failed"
    assert local_state["failure"]["stage"] == "submission"
    assert local_state["failure"]["type"] == "RuntimeError"
    assert local_state["failure"]["message"] == "scheduler unavailable"
    assert published_state == local_state


def test_s3_publication_is_executor_independent_and_uses_configured_endpoint(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search_config.executor.dry_run = True
    search_config.artifacts = ArtifactConfig.model_validate(
        {
            "destination": {
                "type": "s3",
                "endpoint_url": "https://osn.example.test",
                "bucket": "public",
                "path": "experiments/searches",
                "anon": False,
            },
            "checkpoints": "none",
        }
    )
    source = str(Path("tests/configs/train_default.yaml").resolve())
    for candidate in search_config.candidates:
        candidate.config = source
    monkeypatch.setattr(
        "samudra.search.successive_halving._git_provenance",
        lambda allow_dirty: {
            "commit": "f" * 40,
            "dirty": False,
            "package_version": "1.0",
        },
    )
    uploaded = []

    class FakeS3:
        def exists(self, path):
            assert path == "public/experiments/searches/test-search--run"
            return False

        def put_file(self, source, destination):
            uploaded.append(destination)

    fake = FakeS3()
    monkeypatch.setattr(
        "samudra.search.artifacts.ArtifactPublisher._s3",
        staticmethod(lambda destination: fake),
    )

    search_config.build().start()

    assert "public/experiments/searches/test-search--run/config.yaml" in uploaded
    assert "public/experiments/searches/test-search--run/artifacts.parquet" in uploaded


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


def test_ineligible_result_captures_bounded_scheduler_context(tmp_path):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    logs = search.search_dir / "logs"
    logs.mkdir()
    search.write_state(
        {
            "provenance": {"commit": "f" * 40},
            "anchors": {"candidates": ["anchor"], "results": []},
            "rungs": [
                {
                    "candidates": ["a", "b", "c"],
                    "job_id": "123",
                    "results": [],
                },
                {"candidates": [], "results": []},
            ],
        }
    )
    (logs / "r0-123_1.out").write_text("starting\n", encoding="utf-8")
    (logs / "r0-123_1.err").write_text(
        "container runtime unavailable\n", encoding="utf-8"
    )

    result = search._result("b", 0)

    assert result["eligible"] is False
    assert "training_summary.json" in result["error"]
    assert result["scheduler_task_id"] == "123_1"
    assert result["scheduler_stdout_log"] == "logs/r0-123_1.out"
    assert result["scheduler_stdout_tail"] == "starting\n"
    assert result["scheduler_stderr_log"] == "logs/r0-123_1.err"
    assert result["scheduler_stderr_tail"] == "container runtime unavailable\n"


def test_successive_halving_promotes_with_dataframe_and_reports_metrics(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search = search_config.build()
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
    markdown = (search.search_dir / "analysis/report.md").read_text(encoding="utf-8")
    assert "## Latest eligible result per candidate" in markdown
    assert "`b` | 0 | 1 | 0.1" in markdown
    assert "`b`, `c`" in markdown


def test_local_executor_runs_tasks_then_advances(tmp_path, monkeypatch):
    search = config(tmp_path).build()
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


def test_advance_retry_finishes_publication_and_submission(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = {
        "status": "running",
        "anchors": {"candidates": [], "results": []},
        "rungs": [
            {
                "advanced": True,
                "results": [{"candidate": "a"}],
                "candidates": ["a"],
            },
            {
                "advanced": False,
                "results": [],
                "candidates": ["a"],
                "job_id": "orphaned-array",
            },
        ],
    }
    search.write_state(state)
    events: list[object] = []
    monkeypatch.setattr(search, "_write_results", lambda state: events.append("write"))
    monkeypatch.setattr(search, "publish", lambda state: events.append("publish"))
    monkeypatch.setattr(
        search.executor,
        "submit_rung",
        lambda state, rung: events.append(("submit", rung)),
    )

    search.advance(0)

    assert events == ["write", "publish", ("submit", 1)]


def test_task_builds_train_config_and_calls_trainer_directly(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    search_config.candidates = [
        search_config.candidates[1].model_copy(
            update={"config": str(Path("tests/configs/train_default.yaml").resolve())}
        )
    ]
    search = search_config.build()
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
    assert train_config.experiment.search.artifacts_uri == str(
        tmp_path / "published/test-search--run"
    )
    assert train_config.experiment.wandb.group == "test-search--run"
    assert "search" in train_config.experiment.wandb.tags
    assert received[1] == "run"


def test_slurm_executor_submits_array_and_automatic_advance(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm").build()
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
    assert f"--chdir={tmp_path / 'runs'}" in commands[0]
    assert "APPTAINER_MODULE=singularity-ce/4.3.3" in next(
        value for value in commands[0] if value.startswith("--export=")
    )
    assert any(value.startswith("--dependency=afterany:1") for value in commands[1])
    assert f"--chdir={tmp_path / 'runs'}" in commands[1]
    assert "--export=ALL" in commands[1]
    assert "samudra.search.worker" in commands[1][-1]
    saved = search.read_state()["rungs"][0]
    assert saved["job_id"] == "1"
    assert saved["controller_job_id"] == "2"


def test_slurm_does_not_persist_array_without_controller(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm").build()
    search.search_dir.mkdir(parents=True)
    state = {
        "name": "test-search",
        "status": "prepared",
        "anchors": {"candidates": []},
        "rungs": [{"candidates": ["a"]}],
    }
    search.write_state(state)
    submissions = 0

    def submit(command):
        nonlocal submissions
        submissions += 1
        if submissions == 2:
            raise RuntimeError("controller submission failed")
        return "array-job"

    monkeypatch.setattr(search.executor, "_submit", submit)

    with pytest.raises(RuntimeError, match="controller submission failed"):
        search.executor.submit_rung(state, 0)

    saved = search.read_state()
    assert saved["status"] == "prepared"
    assert "job_id" not in saved["rungs"][0]
    assert "controller_job_id" not in saved["rungs"][0]
