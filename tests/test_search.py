# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from samudra.config import TrainConfig
from samudra.search import SearchConfig
from samudra.search.config import (
    AdaptiveDataParallelResourceConfig,
    ArtifactConfig,
    SlurmExecutorConfig,
)
from samudra.search.executors import (
    LocalExecutor,
    SlurmAllocationExecutor,
    SlurmExecutor,
)
from samudra.search.executors.pool import PoolExecutor, Task
from samudra.search.node_launcher import main as node_launcher_main
from samudra.utils.location import S3Location
from samudra.utils.schedule import CosineSchedulerConfig
from samudra.utils.training_summary import (
    write_search_metrics,
    write_search_worker_status,
    write_training_summary,
)


def config(tmp_path: Path, *, executor: str = "local") -> SearchConfig:
    executor_config = {"type": "local", "output_dir": tmp_path / "runs"}
    if executor == "slurm_allocation":
        executor_config = {
            "type": "slurm_allocation",
            "output_dir": tmp_path / "runs",
        }
    elif executor == "slurm":
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
            "allow_dirty": executor != "slurm",
            "candidates": [
                {"name": "anchor", "config": "anchor.yaml", "fixed": True},
                {"name": "a", "config": "a.yaml"},
                {"name": "b", "config": "b.yaml"},
                {"name": "c", "config": "c.yaml"},
            ],
        }
    )


def search_state(search, *, status="running") -> dict:
    """Build the complete durable state shape used by production controllers."""
    return {
        "name": search.config.name,
        "run_id": search.run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": status,
        "provenance": {
            "commit": "f" * 40,
            "dirty": False,
            "package_version": "1.0",
        },
        "anchors": {"candidates": [], "results": []},
        "rungs": [
            {
                "index": index,
                "epochs": epochs,
                "candidates": [],
                "results": [],
                "promoted": [],
                "advanced": False,
            }
            for index, epochs in enumerate(search.rungs)
        ],
    }


def test_config_validates_objective_and_rungs(tmp_path):
    value = config(tmp_path).model_dump(mode="json")
    value["metrics"] = ["train_loss"]
    with pytest.raises(ValidationError, match="objective.metric"):
        SearchConfig.model_validate(value)

    value["metrics"] = ["validation_loss"]
    value["algorithm"]["rungs"] = [3, 1]
    with pytest.raises(ValidationError, match="strictly increasing"):
        SearchConfig.model_validate(value)

    value["algorithm"]["rungs"] = [1, 3]
    value["algorithm"]["minimum_promoted"] = 4
    with pytest.raises(ValidationError, match=r"non-fixed candidates \(3\)"):
        SearchConfig.model_validate(value)


def test_config_rejects_candidate_names_with_colliding_resource_slugs(tmp_path):
    value = config(tmp_path).model_dump(mode="json")
    value["candidates"][1]["name"] = "same name"
    value["candidates"][2]["name"] = "same-name"

    with pytest.raises(ValidationError, match="unique after slug normalization"):
        SearchConfig.model_validate(value)


def test_slurm_rejects_dirty_worktree_before_submission(tmp_path):
    value = config(tmp_path, executor="slurm").model_dump(mode="json")
    value["allow_dirty"] = True

    with pytest.raises(ValidationError, match="not supported by the submitting Slurm"):
        SearchConfig.model_validate(value)


def test_submitted_slurm_rejects_adaptive_data_parallelism(tmp_path):
    value = config(tmp_path, executor="slurm").model_dump(mode="json")
    value["resources"] = {
        "strategy": "adaptive_data_parallel",
        "max_gpus_per_candidate": 8,
        "effective_global_batch_size": 64,
    }

    with pytest.raises(ValidationError, match="do not share an allocation"):
        SearchConfig.model_validate(value)


def test_executor_dictionary_is_an_explicit_extension_point(tmp_path):
    local = config(tmp_path).build()
    allocation = config(tmp_path, executor="slurm_allocation").build()
    slurm = config(tmp_path, executor="slurm").build()
    assert isinstance(local.executor, LocalExecutor)
    assert isinstance(allocation.executor, SlurmAllocationExecutor)
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


def test_slurm_probe_gates_fixed_anchors_at_search_start(tmp_path, monkeypatch):
    search_config = config(tmp_path, executor="slurm")
    assert isinstance(search_config.executor, SlurmExecutorConfig)
    search_config.executor.dry_run = True
    search_config.executor.rung0_probe = True
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

    state = search.read_state()
    assert "job_id" not in state["anchors"]
    assert state["rungs"][0]["probe"]["status"] == "submitted"


def test_start_snapshot_preserves_structured_s3_locations(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.executor.dry_run = True
    train_config = TrainConfig.from_yaml_and_cli(
        [str(Path("tests/configs/train_default.yaml").resolve())]
    )
    train_config.data.sources[0].data_location = S3Location(
        endpoint_url="https://osn.example.test",
        anon=True,
        bucket="public",
        path="data/test.zarr",
    )
    source_path = tmp_path / "s3-train.yaml"
    source_path.write_text(
        yaml.safe_dump(train_config.model_dump(mode="json")), encoding="utf-8"
    )
    source = str(source_path)
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

    bundled = SearchConfig.from_yaml_and_cli([str(search.config_path)])
    train_config = TrainConfig.from_yaml_and_cli([bundled.candidates[0].config])
    assert isinstance(train_config.data.sources[0].data_location, S3Location)
    assert train_config.data.sources[0].data_location.anon is True


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
    initial_results = pd.read_parquet(published / "results.parquet")
    assert initial_results.empty
    assert {"candidate", "rung", "validation_loss"}.issubset(initial_results.columns)
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
    assert figure["media_type"] == "image/png"
    assert figure["candidate"] == "a"
    report = catalog[catalog["artifact"] == "analysis/report.md"].iloc[0]
    assert report["kind"] == "report"
    assert report["media_type"] == "text/markdown"


def test_all_checkpoint_publication_includes_epoch_checkpoints(tmp_path):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {
            "destination": {"type": "local", "path": tmp_path / "published"},
            "checkpoints": "all",
        }
    )
    search = search_config.build()
    state = search_state(search)
    output = search.output_dir("a", 0)
    saved_nets = output / "saved_nets"
    saved_nets.mkdir(parents=True)
    epoch_checkpoint = saved_nets / "ckpt_1.pt"
    epoch_checkpoint.touch()
    state["rungs"][0]["results"] = [
        {
            "candidate": "a",
            "rung": 0,
            "eligible": True,
            "output_dir": str(output),
        }
    ]

    assert search.publisher is not None
    files = search.publisher._files(state)
    relative = f"runs/{output.name}/saved_nets/ckpt_1.pt"
    assert (epoch_checkpoint, relative) in files
    assert search.publisher._kind(relative) == "checkpoint"


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

    def fail_submission(state):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(search.executor, "submit_initial", fail_submission)

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
    state = search_state(search)
    state["anchors"]["candidates"] = ["anchor"]
    state["rungs"][0].update(candidates=["a", "b", "c"], job_id="123")
    search.write_state(state)
    (logs / "r0-123_1.out").write_text("starting\n", encoding="utf-8")
    (logs / "r0-123_1.err").write_text(
        "container runtime unavailable\n", encoding="utf-8"
    )

    result = search._result("b", 0)

    assert result["eligible"] is False
    assert "training_summary.json" in result["error"]
    assert result["scheduler_task_id"] == "123_1"
    assert result["scheduler_stdout_log"] == "logs/r0-123_1.out"
    assert result["scheduler_stderr_log"] == "logs/r0-123_1.err"


def test_nonfinite_metric_is_reported_as_divergence_not_missing_summary(tmp_path):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0]["candidates"] = ["a"]
    search.write_state(state)
    output = search.output_dir("a", 0)
    (output / "saved_nets").mkdir(parents=True)
    (output / "saved_nets/ckpt.pt").touch()
    write_training_summary(
        output,
        {
            "epoch": 1,
            "complete": True,
            "validation_loss": float("nan"),
            "train_loss": 1.0,
            "code_commit": "f" * 40,
        },
    )

    result = search._result("a", 0)

    assert result["eligible"] is False
    assert result["error"] == "metric 'validation_loss' is not finite"


def test_worker_status_path_matches_published_catalog_key(tmp_path):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0]["candidates"] = ["a"]
    search.write_state(state)
    write_result(search, "a", 0, 0.1)
    output = search.output_dir("a", 0)
    write_search_worker_status(output, "completed", optimizer_steps=10)

    result = search._result("a", 0)
    state["rungs"][0]["results"] = [result]
    assert result["worker_status_log"] == (
        f"runs/{output.name}/search_worker_status.json"
    )
    assert search.publisher is not None
    published_keys = {relative for _, relative in search.publisher._files(state)}
    assert result["worker_status_log"] in published_keys


def test_public_artifacts_exclude_raw_logs_by_default(tmp_path):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    (search.search_dir / "logs").mkdir()
    (search.search_dir / "logs/r0.err").write_text(
        "WANDB_API_KEY=secret", encoding="utf-8"
    )
    state = search_state(search)

    assert search.publisher is not None
    published_keys = {relative for _, relative in search.publisher._files(state)}
    assert "logs/r0.err" not in published_keys


def test_artifact_retry_skips_objects_with_already_published_hashes(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    search.write_state(state)
    search._write_results(state)
    calls: list[str] = []
    assert search.publisher is not None
    monkeypatch.setattr(
        search.publisher,
        "_put",
        lambda source, relative: calls.append(relative),
    )

    search.publish(state)
    first_counts = {relative: calls.count(relative) for relative in calls}
    search.publish(state)

    assert calls.count("state.json") == first_counts["state.json"] == 1
    assert calls.count("results.parquet") == first_counts["results.parquet"] == 1


def test_successive_halving_promotes_with_dataframe_and_reports_metrics(
    tmp_path, monkeypatch
):
    search_config = config(tmp_path)
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    search.config_path.write_text("test", encoding="utf-8")
    state = search_state(search)
    state["anchors"]["candidates"] = ["anchor"]
    state["rungs"][0]["candidates"] = ["a", "b", "c"]
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
    state = search_state(search, status="prepared")
    state["rungs"][0]["candidates"] = ["a", "b"]
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


def test_local_executor_coschedules_anchors_and_first_rung(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="prepared")
    state["anchors"]["candidates"] = ["anchor"]
    state["rungs"][0]["candidates"] = ["a", "b"]
    search.write_state(state)
    batches = []
    monkeypatch.setattr(
        search.executor,
        "_run_tasks",
        lambda tasks: batches.append(
            [(task.rung, task.task, task.anchor) for task in tasks]
        ),
    )
    monkeypatch.setattr(search, "advance", lambda rung: batches.append(rung))

    search.executor.submit_initial(state)

    assert batches == [[(1, 0, True), (0, 0, False), (0, 1, False)], 0]
    saved = search.read_state()
    assert saved["anchors"]["job_id"] == "local"
    assert saved["rungs"][0]["job_id"] == "local"


def test_pool_stops_launching_queued_tasks_after_worker_failure():
    started = []

    def run(task: Task) -> None:
        started.append(task.task)
        if task.task == 0:
            raise RuntimeError("candidate failed")

    tasks = [Task(rung=0, task=task, anchor=False) for task in range(5)]

    with pytest.raises(RuntimeError, match="candidate failed"):
        PoolExecutor._run_concurrently(tasks, concurrency=2, runner=run)

    assert 0 in started
    assert set(started) <= {0, 1}


def test_slurm_allocation_executor_uses_exclusive_gpu_steps(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm_allocation").build()
    assert isinstance(search.executor, SlurmAllocationExecutor)
    search.config.executor.max_concurrent = 2
    commands = []
    environments = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_NNODES", "2")
    # Some Slurm installations append a socket/topology suffix.
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8(S:0-1)")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "128")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1048576")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)

    def run(command, *, check, env):
        commands.append(command)
        environments.append(env)

    monkeypatch.setattr("samudra.search.executors.slurm_allocation.subprocess.run", run)

    search.executor._run_tasks(
        [
            type("Task", (), {"rung": 0, "task": task, "anchor": False})()
            for task in range(3)
        ]
    )

    assert len(commands) == 3
    assert all(command[0] == "srun" for command in commands)
    assert all(
        "--exclusive" in command and "--exact" in command for command in commands
    )
    assert all("--gpus-per-task=1" in command for command in commands)
    assert all("--cpus-per-task=16" in command for command in commands)
    assert all("--mem=131072M" in command for command in commands)
    assert all(env["SAMUDRA_DISABLE_DISTRIBUTED"] == "1" for env in environments)
    assert all("RANK" not in env and "WORLD_SIZE" not in env for env in environments)


def test_slurm_allocation_executor_requires_an_allocation(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm_allocation").build()
    assert isinstance(search.executor, SlurmAllocationExecutor)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="must run inside a Slurm allocation"):
        search.executor._run_tasks(
            [type("Task", (), {"rung": 0, "task": 0, "anchor": False})()]
        )


def test_local_executor_preserves_visible_gpu_identifiers(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    assert isinstance(search.executor, LocalExecutor)
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "ignored-by-local-executor")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")

    def run(command, *, check, env):
        calls.append((command, env["CUDA_VISIBLE_DEVICES"]))

    monkeypatch.setattr("samudra.search.executors.local.subprocess.run", run)
    search.executor._run_tasks(
        [
            type("Task", (), {"rung": 0, "task": task, "anchor": False})()
            for task in range(2)
        ]
    )

    assert {device for _, device in calls} == {"GPU-first", "GPU-second"}
    assert all(command[0] != "srun" for command, _ in calls)
    assert all("samudra.search.worker" in command for command, _ in calls)


def test_local_executor_launches_one_torchrun_across_reserved_gpus(
    tmp_path, monkeypatch
):
    search = config(tmp_path).build()
    assert isinstance(search.executor, LocalExecutor)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    calls = []
    monkeypatch.setattr(
        "samudra.search.executors.local.subprocess.run",
        lambda command, *, check, env: calls.append((command, env)),
    )

    search.executor._run_tasks([Task(rung=1, task=0, anchor=False, world_size=4)])

    command, environment = calls[0]
    assert "torch.distributed.run" in command
    assert "--nproc-per-node=4" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert "SAMUDRA_DISABLE_DISTRIBUTED" not in environment


def test_slurm_allocation_launches_multi_node_torchrun(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm_allocation").build()
    assert isinstance(search.executor, SlurmAllocationExecutor)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "128")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1048576")
    calls = []
    monkeypatch.setattr(
        "samudra.search.executors.slurm_allocation.subprocess.run",
        lambda command, *, check, env: calls.append((command, env)),
    )

    search.executor._run_slurm_task(Task(rung=1, task=0, anchor=False, world_size=16))

    command, environment = calls[0]
    assert "--nodes=2" in command
    assert "--ntasks=2" in command
    assert "--gpus-per-task=8" in command
    assert "--mem=1048576M" in command
    assert "samudra.search.node_launcher" in command
    assert "--nnodes=2" in command
    assert "--nproc-per-node=8" in command
    assert "SAMUDRA_DISABLE_DISTRIBUTED" not in environment


def test_node_launcher_maps_slurm_node_rank_to_torchrun(monkeypatch):
    monkeypatch.setenv("SLURM_STEP_NODELIST", "node-[01-02]")
    monkeypatch.setenv("SLURM_NODEID", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "node_launcher",
            "--nnodes=2",
            "--nproc-per-node=8",
            "--master-port=23456",
            "task",
            "config.yaml",
            "state.json",
            "1",
            "0",
        ],
    )
    commands = []

    def run(command, **kwargs):
        if command[0] == "scontrol":
            return SimpleNamespace(stdout="node-01\nnode-02\n")
        commands.append(command)
        return SimpleNamespace()

    monkeypatch.setattr("samudra.search.node_launcher.subprocess.run", run)

    node_launcher_main()

    command = commands[0]
    assert "--node-rank=1" in command
    assert "--master-addr=node-01" in command
    assert "--master-port=23456" in command
    assert command[-5:] == ["task", "config.yaml", "state.json", "1", "0"]


def test_adaptive_resource_plan_is_reused_on_submission_retry(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.config.resources = AdaptiveDataParallelResourceConfig(
        max_gpus_per_candidate=4,
        effective_global_batch_size=32,
    )
    search.config.executor.dry_run = True
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="prepared")
    state["rungs"][0]["candidates"] = ["a"]
    search.write_state(state)
    calls = 0

    def plan_resources(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "a": {
                "world_size": 4,
                "local_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "effective_global_batch_size": 32,
            }
        }

    monkeypatch.setattr(search, "plan_resources", plan_resources)

    search.executor.submit_rung(state, 0)
    search.executor.submit_rung(search.read_state(), 0)

    assert calls == 1
    assert search.read_state()["rungs"][0]["resources"]["a"]["world_size"] == 4


def test_local_executor_keeps_later_rung_trials_on_one_gpu(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    assert isinstance(search.executor, LocalExecutor)
    state = search_state(search)
    state["rungs"][1]["candidates"] = ["a", "b", "c", "d"]
    search.search_dir.mkdir(parents=True)
    search.write_state(state)
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", ",".join(f"GPU-{index}" for index in range(8))
    )
    visible_devices = []

    def run(command, *, check, env):
        visible_devices.append(env["CUDA_VISIBLE_DEVICES"])

    monkeypatch.setattr("samudra.search.executors.local.subprocess.run", run)
    monkeypatch.setattr(search, "advance", lambda rung: None)

    search.executor.submit_rung(state, 1)

    assert len(visible_devices) == 4
    assert all("," not in device for device in visible_devices)
    assert len(set(visible_devices)) == 4


def test_advance_retry_finishes_publication_and_submission(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0].update(
        advanced=True,
        results=[{"candidate": "a"}],
        candidates=["a"],
        promoted=["a"],
    )
    state["rungs"][1].update(candidates=["a"], job_id="orphaned-array")
    search.write_state(state)
    events: list[object] = []
    monkeypatch.setattr(search, "_write_results", lambda state: events.append("write"))
    monkeypatch.setattr(search, "_write_report", lambda state: events.append("report"))
    monkeypatch.setattr(search, "publish", lambda state: events.append("publish"))
    monkeypatch.setattr(
        search.executor,
        "submit_rung",
        lambda state, rung: events.append(("submit", rung)),
    )

    search.advance(0)

    assert events == [
        "write",
        "publish",
        ("submit", 1),
        "report",
        "publish",
    ]


def test_report_failure_does_not_block_next_rung_submission(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0]["candidates"] = ["a", "b", "c"]
    search.write_state(state)
    for name, score in {"a": 0.3, "b": 0.1, "c": 0.2}.items():
        write_result(search, name, 0, score)
    submitted = []
    monkeypatch.setattr(
        search.executor,
        "submit_rung",
        lambda state, rung: submitted.append(rung),
    )
    monkeypatch.setattr(
        search,
        "_write_report",
        lambda state: (_ for _ in ()).throw(RuntimeError("report failed")),
    )

    with pytest.raises(RuntimeError, match="report failed"):
        search.advance(0)

    assert submitted == [1]
    assert search.read_state()["rungs"][0]["advanced"] is True


def test_final_rung_with_failed_jobs_is_partial_not_complete(tmp_path):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    search.config_path.write_text("test", encoding="utf-8")
    state = search_state(search)
    state["rungs"][0].update(candidates=["a", "b"], promoted=["a", "b"], advanced=True)
    state["rungs"][1]["candidates"] = ["a", "b"]
    search.write_state(state)
    write_result(search, "a", 1, 0.1)

    search.advance(1)

    updated = search.read_state()
    assert updated["status"] == "partial"
    assert updated["rungs"][1]["results"][0]["eligible"] is True
    assert updated["rungs"][1]["results"][1]["eligible"] is False
    report = (search.search_dir / "analysis/report.md").read_text(encoding="utf-8")
    assert "Best completed finalist" in report


def test_intermediate_worker_failure_makes_terminal_search_partial(tmp_path):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    search.config_path.write_text("test", encoding="utf-8")
    state = search_state(search)
    state["rungs"][0].update(
        candidates=["a", "b"],
        promoted=["a"],
        advanced=True,
        results=[
            {"candidate": "a", "rung": 0, "eligible": True},
            {"candidate": "b", "rung": 0, "eligible": False, "error": "OOM"},
        ],
    )
    state["rungs"][1]["candidates"] = ["a"]
    search.write_state(state)
    write_result(search, "a", 1, 0.1)

    search.advance(1)

    assert search.read_state()["status"] == "partial"
    report = (search.search_dir / "analysis/report.md").read_text(encoding="utf-8")
    assert "Best completed finalist" in report


def test_rank_zero_refuses_to_overwrite_existing_worker_output(tmp_path, monkeypatch):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0]["candidates"] = ["a"]
    search.write_state(state)
    search.output_dir("a", 0).mkdir(parents=True)
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        search.train_task(0, 0, anchor=False)


def test_state_schema_rejects_incomplete_persistent_state(tmp_path):
    search = config(tmp_path).build()
    search.search_dir.mkdir(parents=True)

    with pytest.raises(ValidationError, match="rungs.0.epochs"):
        search.write_state(
            {
                **search_state(search),
                "rungs": [{"candidates": ["a"]}],
            }
        )


def test_task_builds_train_config_and_calls_trainer_directly(tmp_path, monkeypatch):
    search_config = config(tmp_path)
    search_config.artifacts = ArtifactConfig.model_validate(
        {"destination": {"type": "local", "path": tmp_path / "published"}}
    )
    candidate_config = TrainConfig.from_yaml_and_cli(
        [str(Path("tests/configs/train_default.yaml").resolve())]
    )
    candidate_config.scheduler = CosineSchedulerConfig()
    effective_batch_size = candidate_config.batch_size * 4 * 3
    search_config.resources = AdaptiveDataParallelResourceConfig(
        max_gpus_per_candidate=4,
        effective_global_batch_size=effective_batch_size,
    )
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate_config.model_dump(mode="json")), encoding="utf-8"
    )
    search_config.candidates = [
        search_config.candidates[1].model_copy(update={"config": str(candidate_path)})
    ]
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search)
    state["rungs"][0]["candidates"] = ["a"]
    state["rungs"][0]["resources"] = {
        "a": {
            "world_size": 4,
            "local_batch_size": candidate_config.batch_size,
            "gradient_accumulation_steps": 3,
            "effective_global_batch_size": effective_batch_size,
        }
    }
    search.write_state(state)
    received = []

    class FakeTrainer:
        def __init__(self, train_config):
            received.append(train_config)
            self.num_batches_seen = 0
            self.train_progress = type("Progress", (), {"optimizer_steps": 0})()

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
    assert train_config.scheduler is not None
    assert train_config.scheduler.target_epochs == 3
    assert train_config.batch_size == candidate_config.batch_size
    assert train_config.gradient_accumulation_steps == 3
    assert train_config.experiment.search.candidate == "a"
    assert train_config.experiment.search.world_size == 4
    assert train_config.experiment.search.effective_global_batch_size == (
        effective_batch_size
    )
    assert train_config.experiment.search.adaptive_data_parallel is True
    assert train_config.experiment.search.artifacts_uri == str(
        tmp_path / "published/test-search--run"
    )
    assert train_config.experiment.wandb.group == "test-search--run"
    assert "search" in train_config.experiment.wandb.tags
    assert "test-search--run" not in train_config.experiment.wandb.tags
    assert received[1] == "run"


def test_slurm_executor_submits_array_and_automatic_advance(tmp_path, monkeypatch):
    search = config(tmp_path, executor="slurm").build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="prepared")
    state["rungs"][0]["candidates"] = ["a", "b"]
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
    assert f"--export=ALL,SAMUDRA_CODE_COMMIT={'f' * 40}" in commands[1]
    assert "module load singularity-ce/4.3.3" in commands[1][-1]
    assert "samudra.search.worker" in commands[1][-1]
    saved = search.read_state()["rungs"][0]
    assert saved["job_id"] == "1"
    assert saved["controller_job_id"] == "2"


def test_slurm_reuses_array_when_controller_submission_is_retried(
    tmp_path, monkeypatch
):
    search = config(tmp_path, executor="slurm").build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="prepared")
    state["rungs"][0]["candidates"] = ["a"]
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
    assert saved["status"] == "running"
    assert saved["rungs"][0]["job_id"] == "array-job"
    assert "controller_job_id" not in saved["rungs"][0]

    search.executor.submit_rung(saved, 0)

    assert submissions == 3
    saved = search.read_state()
    assert saved["rungs"][0]["job_id"] == "array-job"
    assert saved["rungs"][0]["controller_job_id"] == "array-job"


def test_slurm_probe_gates_first_rung_array(tmp_path, monkeypatch):
    search_config = config(tmp_path, executor="slurm")
    assert isinstance(search_config.executor, SlurmExecutorConfig)
    search_config.executor.rung0_probe = True
    search = search_config.build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="prepared")
    state["rungs"][0]["candidates"] = ["a", "b"]
    search.write_state(state)
    commands = []

    def submit(command):
        commands.append(command)
        return str(len(commands))

    monkeypatch.setattr(search.executor, "_submit", submit)

    search.executor.submit_rung(state, 0)

    assert len(commands) == 2
    assert "--array=0-0%1" in commands[0]
    assert "r0-probe" in next(
        value for value in commands[0] if value.startswith("--output=")
    )
    assert "SAMUDRA_MODULE_ARGS=probe " in next(
        value for value in commands[0] if value.startswith("--export=")
    )
    assert "module load singularity-ce/4.3.3" in commands[1][-1]
    assert "release-probe" in commands[1][-1]
    saved = search.read_state()
    assert saved["status"] == "validating"
    assert saved["rungs"][0]["probe"] == {
        "candidate": "a",
        "job_id": "1",
        "controller_job_id": "2",
        "status": "submitted",
    }


def test_successful_probe_releases_full_rung(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "f" * 40)
    search = config(tmp_path, executor="slurm").build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="validating")
    state["rungs"][0].update(
        candidates=["a", "b"],
        probe={
            "candidate": "a",
            "job_id": "1",
            "controller_job_id": "2",
            "status": "submitted",
        },
    )
    search.write_state(state)
    output = search.search_dir / "probe/a"
    output.mkdir(parents=True)
    write_search_worker_status(
        output,
        "completed",
        optimizer_steps=1,
        batches_seen=32,
    )
    released: list[object] = []
    monkeypatch.setattr(
        search.executor,
        "submit_anchors",
        lambda state: released.append("anchors"),
    )
    monkeypatch.setattr(
        search.executor,
        "submit_validated_rung",
        lambda state, rung: released.append(("rung", rung)),
    )

    search.release_probe(0)

    updated = search.read_state()
    assert updated["rungs"][0]["probe"]["status"] == "complete"
    assert updated["rungs"][0]["probe"]["optimizer_steps"] == 1
    assert released == ["anchors", ("rung", 0)]


def test_failed_probe_terminates_search_before_array(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "f" * 40)
    search = config(tmp_path, executor="slurm").build()
    search.search_dir.mkdir(parents=True)
    state = search_state(search, status="validating")
    state["rungs"][0].update(
        candidates=["a", "b"],
        probe={
            "candidate": "a",
            "job_id": "1",
            "controller_job_id": "2",
            "status": "submitted",
        },
    )
    search.write_state(state)
    output = search.search_dir / "probe/a"
    output.mkdir(parents=True)
    write_search_worker_status(output, "failed", error="bad data")
    monkeypatch.setattr(search, "publish", lambda state: None)

    with pytest.raises(ValueError, match="probe failed"):
        search.release_probe(0)

    updated = search.read_state()
    assert updated["status"] == "failed"
    assert updated["failure"]["stage"] == "rung_probe"
    assert updated["rungs"][0]["probe"]["status"] == "failed"
