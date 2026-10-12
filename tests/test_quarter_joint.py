# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from samudra.experiments import quarter_joint as runner
from samudra.experiments.task_schedule import TaskSchedule


def args(tmp_path, mode="qualify", arm="observation-only"):
    return SimpleNamespace(
        mode=mode,
        arm=arm,
        data=str(tmp_path / "data"),
        output=str(tmp_path / "output"),
        name="test-quarter",
        qualification=None,
        selection_reference=str(tmp_path / "reference.json"),
        runtime_contract=str(tmp_path / "runtime.json"),
        om4_data=None,
        compact_audit=None,
        train_hours=22.0,
        validate_every=100,
    )


def test_frozen_science_and_production_never_bypasses_fit(tmp_path):
    qualification = runner.pilot_arguments(args(tmp_path))
    replay = runner.pilot_arguments(args(tmp_path, "qualify-replay"))
    assert vars(replay) == vars(qualification)
    production = runner.pilot_arguments(args(tmp_path, "train"))
    assert qualification.fit_probe is True
    assert production.fit_probe is False
    assert production.wandb_mode == "online"
    assert qualification.wandb_mode == "disabled"
    assert production.from_scratch and production.observation_normalization
    assert production.accumulate == 8 and production.seed == 1729
    assert production.core_lr == 1e-4
    assert production.completion_weight == production.reconstruction_weight == 0.1
    assert production.evolution_architecture == "extent-unet"
    assert production.selection_metric == "rmse-only"
    assert production.global_observations and production.strict_velocity_support


def test_schedules_are_exact_matched_exposure():
    baseline = TaskSchedule(*runner.ARMS["u-global"])
    obs = TaskSchedule(*runner.ARMS["observation-only"])
    assert baseline.total == obs.total == 6000
    assert baseline.counts(6000) == {"om4": 3000, "observation": 3000}
    assert baseline.counts(3000)["observation"] < 1500
    assert obs.counts(6000) == {"om4": 0, "observation": 6000}
    assert all(obs.task(step) == "observation" for step in range(6000))


def test_empty_or_mismatched_qualification_cannot_authorize_train():
    contract = {"arm": "observation-only"}
    with pytest.raises(ValueError, match="contract differs"):
        runner.check_qualification({}, contract)
    with pytest.raises(ValueError, match="Missing measured.*fit"):
        runner.check_qualification({"quarter_contract": contract}, contract)
    evidence: dict[str, dict[str, Any]] = {
        name: {"passed": True} for name in ["fit", "ddp", "resume", "io", "throughput"]
    }
    record = {"quarter_contract": contract, "quarter_evidence": evidence}
    with pytest.raises(ValueError, match="four real ranks"):
        runner.check_qualification(record, contract)
    evidence["ddp"]["world_size"] = 4
    with pytest.raises(ValueError, match="throughput"):
        runner.check_qualification(record, contract)


def test_om4_adapter_delegates_one_bounded_window():
    calls = []

    def sample(ids):
        calls.append(ids)
        return tuple(range(6))

    window = SimpleNamespace(model_sample=sample)
    adapter = object.__new__(runner.Om4Adapter)
    adapter.reads, adapter.read_seconds = 0, 0.0
    assert adapter.model_sample(window, [7]) == tuple(range(6))
    assert calls == [[7]]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def set_phase(self, phase):
        assert phase == "joint"

    def to(self, *args, **kwargs):
        return self


def test_observation_only_never_constructs_om4(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(runner.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        runner,
        "Pilot",
        lambda *args, **kwargs: SimpleNamespace(
            model=TinyModel(), data=object(), args=args[0]
        ),
    )
    monkeypatch.setattr(
        runner, "distributed_task_loss", lambda *args, **kwargs: object()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation-only constructed an OM4 reader")

    monkeypatch.setattr(runner, "Om4Adapter", forbidden)
    actual = runner.QuarterRunner(args(tmp_path), {})
    assert actual.om4 is None
    with pytest.raises(ValueError, match="cannot request OM4"):
        actual.update("om4")


def test_pilot_nonwriter_does_not_create_artifacts_or_wandb(tmp_path, monkeypatch):
    from samudra.experiments import observation_pilot

    options = runner.pilot_arguments(args(tmp_path))
    options.wandb_mode = "online"
    data = Path(options.data)
    data.mkdir()
    (data / "DATA_READY.json").write_text(
        json.dumps(
            {"verification": "all 351 NPZ files checked against source SHA256SUMS"}
        )
    )
    for name in ["SHA256SUMS", "grid.npz", "statistics.npz"]:
        (data / name).write_bytes(b"fixture")
    sample = SimpleNamespace(
        grid={"names": np.array(["thetao_0"]), "mean": np.zeros(1), "std": np.ones(1)},
        dual_grid=False,
        use_observation_normalization=lambda: None,
        paths=lambda split: [
            Path(str(index)) for index in range(243 if split == "train" else 9)
        ],
    )
    monkeypatch.setattr(observation_pilot, "Samples", lambda *args, **kwargs: sample)
    monkeypatch.setattr(
        observation_pilot.ObservationTransfer,
        "from_arguments",
        lambda *args: TinyModel(),
    )
    pilot = observation_pilot.Pilot(options, write_artifacts=False)
    assert pilot.wandb is None
    assert not Path(options.output).exists()
    options.wandb_mode = "disabled"
    observation_pilot.Pilot(options)
    assert (Path(options.output) / "manifest.json").is_file()


def test_determinism_policy_precedes_cuda_and_is_not_silently_overridden(monkeypatch):
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        runner.configure_determinism()
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match=":4096:8"):
        runner.configure_determinism()


def test_replica_hash_handles_scalar_buffers_and_detects_changed_weights():
    state = {"weight": torch.ones(2, dtype=torch.bfloat16), "counter": torch.tensor(0)}
    first = runner.tensor_hash(state)
    assert (
        runner.tensor_hash({key: value.clone() for key, value in state.items()})
        == first
    )
    state["counter"] += 1
    assert runner.tensor_hash(state) != first


def test_qualification_requires_both_real_task_curves_and_replays():
    tasks = ["observation", "om4"]
    hashes = ["a" * 64] * 4
    evidence: dict[str, dict[str, Any]] = {
        "fit": {
            "passed": True,
            "losses": {task: list(range(11, 0, -1)) for task in tasks},
            "gradients": {
                f"{task}/{component}": True
                for task in tasks
                for component in [
                    "initializer",
                    "evolution",
                    "completion_head",
                    "adapter",
                ]
            },
        },
        "ddp": {
            "passed": True,
            "world_size": 4,
            "model_hashes": hashes,
            "task_model_hashes": {task: hashes for task in tasks},
        },
        "resume": {"passed": True, "tasks": tasks, "fresh_process": True},
        "retention": {
            "passed": True,
            "seconds": 1.0,
            "metrics": {
                "origins": 12,
                "ts_mse": 1.0,
                "reconstruction_ts_mse": 2.0,
                "full_mse": 3.0,
            },
        },
        "io": {"passed": True, "observation_samples": 2, "om4_windows": 2},
        "throughput": {
            "passed": True,
            "seconds_per_update": 10.0,
            "task_seconds_per_update": {task: 10.0 for task in tasks},
        },
    }
    contract = {"arm": "u-global"}
    record = {"quarter_contract": contract, "quarter_evidence": evidence}
    runner.check_qualification(record, contract)
    evidence["resume"]["fresh_process"] = False
    with pytest.raises(ValueError, match="fresh-process"):
        runner.check_qualification(record, contract)
    evidence["resume"]["fresh_process"] = True
    evidence["resume"]["tasks"] = ["observation"]
    with pytest.raises(ValueError, match="every trained task"):
        runner.check_qualification(record, contract)
    evidence["resume"]["tasks"] = tasks
    evidence["fit"]["losses"]["om4"] = [1.0] * 11
    with pytest.raises(ValueError, match="om4 fitting curve"):
        runner.check_qualification(record, contract)


@pytest.mark.parametrize(
    "drift", [None, "version", "origin", "binary", "unlisted_extension"]
)
def test_runtime_binds_actual_import_origin_and_loaded_binary(
    tmp_path, monkeypatch, drift
):
    import sys
    from types import ModuleType

    package = tmp_path / "fixture_runtime"
    package.mkdir()
    origin, binary = package / "__init__.py", package / "_native.so"
    origin.write_text("# package")
    binary.write_bytes(b"native binary")
    module: Any = ModuleType("fixture_runtime")
    child = ModuleType("fixture_runtime._native")
    module.__file__, module.__version__ = str(origin), "1.0"
    child.__file__ = str(binary)
    monkeypatch.setitem(sys.modules, "fixture_runtime", module)
    monkeypatch.setitem(sys.modules, "fixture_runtime._native", child)
    monkeypatch.setattr(runner, "RUNTIME_MODULES", frozenset({"fixture_runtime"}))
    expected: dict[str, Any] = {
        "version": "1.0",
        "origin": str(origin),
        "binaries": {
            str(origin): runner.digest(origin),
            str(binary): runner.digest(binary),
        },
    }
    if drift == "version":
        module.__version__ = "2.0"
    elif drift == "origin":
        module.__file__ = str(tmp_path / "shadow" / "__init__.py")
    elif drift == "binary":
        binary.write_bytes(b"changed")
    elif drift == "unlisted_extension":
        del expected["binaries"][str(binary)]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"modules": {"fixture_runtime": expected}}))
    monkeypatch.setenv("SAMUDRA_RUNTIME_SHA256", runner.digest(path))
    if drift is None:
        runner.verify_runtime(path)
    else:
        with pytest.raises(ValueError, match="differs|missing from runtime"):
            runner.verify_runtime(path)


def test_compact_audit_binds_bytes_across_mount_aliases(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    native = ModuleType("samudra_rust_loader")
    binary = tmp_path / "actual-runtime.so"
    binary.write_bytes(b"qualified extension")
    native.__file__ = str(binary)
    monkeypatch.setitem(sys.modules, "samudra_rust_loader", native)
    store = tmp_path / "actual-source" / "OM4.zarr"
    store.mkdir(parents=True)
    (store / ".zmetadata").write_text("metadata")
    audit = tmp_path / "compact-audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "source": "/data/OM4.zarr",
                "module_file": "/build/runtime/extension.so",
                "module_binary_sha256": {binary.name: runner.digest(binary)},
                "module_file_sha256": runner.digest(binary),
                "source_metadata_sha256": runner.digest(store / ".zmetadata"),
                "cases": [{"equal_including_nan": True}],
            }
        )
    )
    assert runner.verify_compact_audit(audit, store.parent) == runner.digest(audit)
    binary.write_bytes(b"unqualified replacement")
    with pytest.raises(ValueError, match="decoder audit"):
        runner.verify_compact_audit(audit, store.parent)


def test_walltime_and_signal_stop_are_rank_zero_broadcast_decisions(monkeypatch):
    import signal

    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner.dist, "all_reduce", lambda *args, **kwargs: None)
    decisions = []
    monkeypatch.setattr(
        runner.dist,
        "broadcast_object_list",
        lambda value, **kwargs: decisions.append(value[0]),
    )
    monkeypatch.setattr(runner.time, "monotonic", lambda: 101.0)
    stop = runner.StopController(1.0, 0, "cpu", started=100.0)
    assert stop.check() is None
    stop.handle_signal(signal.SIGUSR1, None)
    assert stop.check() == "SIGUSR1"
    stop.reason = None
    stop.deadline = 100.0
    assert stop.check() == "train_hours"
    assert decisions == [None, "SIGUSR1", "train_hours"]
    stop.close()


def test_partial_checkpoint_resumes_pending_validation_before_next_update(tmp_path):
    events = []
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.out, subject.rank = tmp_path, 0
    subject.args = SimpleNamespace(validate_every=100)
    subject.schedule = TaskSchedule(0, 6000, "scratch")
    subject.completed, subject.last_validated = 100, 0
    subject.contract = {"fixture": True}
    subject.pilot = SimpleNamespace(emit=lambda value: events.append(value["event"]))
    last = tmp_path / "quarter-last.pt"
    last.write_bytes(b"checkpoint")
    subject.restore = lambda path: events.append("restore")
    subject.checkpoint = lambda path: events.append("collective_checkpoint")

    def validate():
        events.append("validate100")
        subject.last_validated = 100

    subject.validate = validate
    decisions = iter([None, "train_hours"])
    subject.train(SimpleNamespace(check=lambda: next(decisions)))
    assert events == [
        "restore",
        "validate100",
        "collective_checkpoint",
        "collective_checkpoint",
        "quarter_partial",
    ]
    partial = json.loads((tmp_path / "PARTIAL.json").read_text())
    assert partial["updates"] == partial["last_validated"] == 100
    assert partial["checkpoint_sha256"] == runner.digest(last)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_update_restores_core_training_mode_after_rank_zero_evaluation(monkeypatch):
    subject: Any = object.__new__(runner.QuarterRunner)
    model = TinyModel()
    subject.pilot = SimpleNamespace(
        model=model, training=[Path(str(i)) for i in range(243)]
    )
    subject.ddp = torch.nn.Sequential(model)
    subject.om4 = None
    subject.schedule = TaskSchedule(0, 6000, "scratch")
    subject.completed, subject.rank = 0, 0
    subject.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None)
    monkeypatch.setattr(runner.dist, "get_world_size", lambda: 4)
    # Match Pilot.evaluate: it changes the core, not the outer DDP wrapper.
    subject.pilot.model.eval()
    assert subject.ddp.training and not model.training

    class ForwardBoundaryReached(Exception):
        pass

    def inspect_mode(ddp, *args, **kwargs):
        assert ddp.training and model.training and model.layer.training
        raise ForwardBoundaryReached

    monkeypatch.setattr(runner, "accumulate_task_gradients", inspect_mode)
    with pytest.raises(ForwardBoundaryReached):
        subject.update()


@pytest.mark.parametrize("corrupt_best", [False, True])
def test_terminal_marker_matches_selected_evaluation_contract(tmp_path, corrupt_best):
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.out, subject.rank = tmp_path, 0
    subject.args = SimpleNamespace(validate_every=100)
    subject.schedule = TaskSchedule(*runner.ARMS["u-global"])
    subject.completed = subject.last_validated = 6000
    subject.contract = {"fixture": True}
    subject.restore = lambda path: None
    last, selected = tmp_path / "quarter-last.pt", tmp_path / "best.pt"
    last.write_bytes(b"terminal weights and optimizer")
    selected.write_bytes(b"selected weights")
    best = {
        "global_step": 5700,
        "score": 0.8,
        "checkpoint_sha256": runner.digest(selected),
    }
    (tmp_path / "best.json").write_text(json.dumps(best))
    (tmp_path / "PARTIAL.json").write_text("{}")
    if corrupt_best:
        selected.write_bytes(b"changed selected weights")
        with pytest.raises(ValueError, match="Selected checkpoint hash differs"):
            subject.train(SimpleNamespace())
        assert not (tmp_path / "TRAIN_COMPLETE.json").exists()
        return
    subject.train(SimpleNamespace())
    complete = json.loads((tmp_path / "TRAIN_COMPLETE.json").read_text())
    assert complete["global_step"] == complete["updates"] == 6000
    assert complete["task_counts"] == {"om4": 3000, "observation": 3000}
    assert complete["best"] == best
    assert complete["best_score"] == best["score"]
    assert complete["best_checkpoint_sha256"] == runner.digest(selected)
    assert complete["checkpoint_sha256"] == runner.digest(last)
    assert complete["quarter_contract"] == subject.contract
    assert complete["completed_utc"]
    assert not (tmp_path / "PARTIAL.json").exists()
    assert not (tmp_path / "COMPLETE.json").exists()


@pytest.mark.parametrize("bad", [None, "nan", "origins"])
def test_qualification_measures_actual_retention_once(monkeypatch, bad):
    subject: Any = object.__new__(runner.QuarterRunner)
    calls = []
    metrics = dict(origins=12, ts_mse=1.0, reconstruction_ts_mse=2.0, full_mse=3.0)
    if bad == "nan":
        metrics["full_mse"] = math.nan
    if bad == "origins":
        metrics["origins"] = 11
    subject.device = "cpu"

    def retention():
        calls.append("retention")
        return metrics

    subject.om4 = SimpleNamespace(retention=retention)
    monkeypatch.setattr(
        runner.torch.cuda, "synchronize", lambda device: calls.append("sync")
    )
    times = iter([10.0, 12.5])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    if bad:
        with pytest.raises(ValueError, match="12-origin OM4 retention"):
            subject.qualify_retention()
    else:
        assert subject.qualify_retention() == {
            "passed": True,
            "metrics": metrics,
            "seconds": 2.5,
        }
    assert calls == ["sync", "retention", "sync"]
    subject.om4 = None
    assert subject.qualify_retention() is None


def pending_fixture(tmp_path, tasks, arm=None):
    contract = {"arm": arm or ("u-global" if "om4" in tasks else "observation-only")}
    hashes = ["a" * 64] * 4
    inputs: dict[str, Any] = {}
    for task in tasks:
        inputs[task] = {}
        for kind, prefix in (("checkpoint", "resume"), ("expected", "expected")):
            path = tmp_path / f"qualification-{prefix}-{task}.pt"
            path.write_bytes(f"diagnostic {task} {kind}".encode())
            inputs[task][kind] = {"file": path.name, "sha256": runner.digest(path)}
    evidence = {
        "fit": {
            "passed": True,
            "losses": {t: list(range(11, 0, -1)) for t in tasks},
            "gradients": {
                f"{t}/{c}": True
                for t in tasks
                for c in ("initializer", "evolution", "completion_head", "adapter")
            },
        },
        "ddp": {
            "passed": True,
            "world_size": 4,
            "model_hashes": hashes,
            "task_model_hashes": {t: hashes for t in tasks},
        },
        "resume": {
            "passed": False,
            "fresh_process": False,
            "tasks": tasks,
            "producer_processes": [f"producer{i}" for i in range(4)],
            "inputs": inputs,
        },
        "io": {"passed": True, "observation_samples": 2, "om4_windows": 2},
        "throughput": {
            "passed": True,
            "seconds_per_update": 2,
            "task_seconds_per_update": {t: 2 for t in tasks},
        },
        "retention": {
            "passed": True,
            "seconds": 2,
            "metrics": dict(origins=12, ts_mse=1, reconstruction_ts_mse=2, full_mse=3),
        },
    }
    if contract["arm"] in runner.EARLY_ARMS:
        evidence["io"]["early_window_audit"] = valid_early_audit()
        if runner.EARLY_ARMS[contract["arm"]]:
            evidence["fit"]["gradients"].update(
                {
                    f"{task}/latent_{name}_rows": True
                    for task in tasks
                    for name in ("initializer", "evolution")
                }
            )
    pending = dict(
        diagnostic_only=True, quarter_contract=contract, quarter_evidence=evidence
    )
    runner.atomic_json(pending, tmp_path / "QUALIFICATION_PENDING.json")
    return pending


def replay_receipt(tmp_path, pending, task):
    return dict(
        diagnostic_only=True,
        passed=True,
        task=task,
        quarter_contract=pending["quarter_contract"],
        pending_sha256=runner.digest(tmp_path / "QUALIFICATION_PENDING.json"),
        inputs=pending["quarter_evidence"]["resume"]["inputs"][task],
        processes=[f"{task}-restart{i}" for i in range(4)],
    )


@pytest.mark.parametrize(
    "drift",
    [
        None,
        "producer",
        "same_replay",
        "contract",
        "pending",
        "checkpoint",
        "expected",
        "receipt_inputs",
    ],
)
def test_final_gate_requires_separate_hashed_task_restarts(tmp_path, drift):
    tasks = ["om4", "observation"]
    pending = pending_fixture(tmp_path, tasks)
    contract = pending["quarter_contract"]
    om4 = replay_receipt(tmp_path, pending, "om4")
    runner.atomic_json(om4, tmp_path / "qualification-replay-om4.json")
    assert not runner.finalize_qualification(tmp_path, contract)
    assert not (tmp_path / "QUALIFIED.json").exists()
    observation = replay_receipt(tmp_path, pending, "observation")
    if drift == "producer":
        observation["processes"] = pending["quarter_evidence"]["resume"][
            "producer_processes"
        ]
    elif drift == "same_replay":
        observation["processes"] = om4["processes"]
    elif drift == "contract":
        observation["quarter_contract"] = {"arm": "observation-only"}
    elif drift == "pending":
        observation["pending_sha256"] = "changed"
    elif drift == "receipt_inputs":
        observation["inputs"] = {}
    elif drift in ("checkpoint", "expected"):
        artifact = observation["inputs"][drift]["file"]
        (tmp_path / artifact).write_bytes(b"changed")
    runner.atomic_json(observation, tmp_path / "qualification-replay-observation.json")
    if drift:
        with pytest.raises(ValueError, match="receipt|hash/path"):
            runner.finalize_qualification(tmp_path, contract)
        assert not (tmp_path / "QUALIFIED.json").exists()
        return
    assert runner.finalize_qualification(tmp_path, contract)
    qualified = json.loads((tmp_path / "QUALIFIED.json").read_text())
    runner.check_qualification(qualified, contract)
    resume = qualified["quarter_evidence"]["resume"]
    assert resume["passed"] and resume["fresh_process"]
    for task in tasks:
        assert resume["receipts"][task]["sha256"] == runner.digest(
            tmp_path / f"qualification-replay-{task}.json"
        )


@pytest.mark.parametrize(
    "failure", [None, "same_process", "wrong_task", "changed_state"]
)
def test_fresh_replay_compares_one_update_before_publishing(
    tmp_path, monkeypatch, failure
):
    pending = pending_fixture(tmp_path, ["observation"])
    task = "observation"
    expected = {
        "diagnostic_only": True,
        "quarter_contract": pending["quarter_contract"],
        "model": {"weight": torch.tensor(2.0)},
        "completed": 1,
        "optimizer": {"step": torch.tensor(1)},
        "counts": {"om4": 0, "observation": 1},
        "rng": {"ranks": [torch.tensor(i) for i in range(4)]},
    }
    expected_path = tmp_path / "qualification-expected-observation.pt"
    runner.atomic_torch(expected, expected_path)
    pending["quarter_evidence"]["resume"]["inputs"][task]["expected"]["sha256"] = (
        runner.digest(expected_path)
    )
    runner.atomic_json(pending, tmp_path / "QUALIFICATION_PENDING.json")
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.out, subject.rank, subject.contract = (
        tmp_path,
        0,
        pending["quarter_contract"],
    )
    subject.completed = 0
    subject.schedule = (
        TaskSchedule(3000, 3000, "mixed")
        if failure == "wrong_task"
        else TaskSchedule(0, 6000, "scratch")
    )
    events: list[Any] = []
    subject.restore = lambda path: events.append(("restore", path.name))
    subject.update = lambda: events.append(("update", task))
    actual = dict(expected)
    if failure == "changed_state":
        actual["model"] = {"weight": torch.tensor(3.0)}
    subject.replay_state = lambda: actual
    tokens = (
        [f"producer{i}" for i in range(4)]
        if failure == "same_process"
        else [f"replay{i}" for i in range(4)]
    )
    monkeypatch.setattr(runner, "process_tokens", lambda: tokens)
    monkeypatch.setattr(runner.dist, "barrier", lambda: events.append(("barrier",)))
    if failure:
        with pytest.raises((ValueError, AssertionError)):
            subject.qualify_replay(task)
        assert not (tmp_path / "QUALIFIED.json").exists()
    else:
        subject.qualify_replay(task)
        assert events[:2] == [
            ("restore", "qualification-resume-observation.pt"),
            ("update", task),
        ]
        assert (tmp_path / "QUALIFIED.json").exists()


def test_production_rejects_diagnostic_weights(tmp_path):
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.args = SimpleNamespace(mode="train")
    path = tmp_path / "quarter-last.pt"
    runner.atomic_torch({"diagnostic_only": True}, path)
    with pytest.raises(ValueError, match="cannot initialize production"):
        subject.restore(path)


@pytest.mark.parametrize("arm", list(runner.ARMS))
def test_qualify_writes_diagnostic_expectations_but_no_final_gate(
    tmp_path, monkeypatch, arm
):
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.out, subject.rank, subject.device = tmp_path, 0, "cpu"
    subject.completed, subject.io_samples, subject.io_seconds = 0, 1, 1.0
    subject.schedule = TaskSchedule(*runner.ARMS[arm])
    subject.contract = dict(
        arm=arm,
        code_commit="code",
        runtime_extension_sha256="runtime",
        data={"SHA256SUMS": "data"},
        selection_reference_sha256="reference",
    )
    state = {"step": 0}
    subject.pilot = SimpleNamespace(
        model=SimpleNamespace(
            state_dict=lambda: {"weight": torch.tensor(state["step"])}
        ),
        args=runner.pilot_arguments(args(tmp_path, arm=arm)),
        validation=[],
        evaluate=lambda paths: {},
        control={},
    )
    subject.optimizer = SimpleNamespace(state_dict=lambda: dict(state))
    retention_calls = []

    def retention():
        retention_calls.append(True)
        return dict(origins=12, ts_mse=1.0, reconstruction_ts_mse=2.0, full_mse=3.0)

    subject.om4 = (
        SimpleNamespace(
            reads=1,
            read_seconds=1.0,
            retention=retention,
            audit_early=valid_early_audit,
        )
        if arm != "observation-only"
        else None
    )

    def checkpoint(path):
        runner.atomic_torch(
            dict(state=state.copy(), completed=subject.completed, diagnostic_only=True),
            path,
        )

    def restore(path):
        saved = torch.load(path, weights_only=False)
        state.update(saved["state"])
        subject.completed = saved["completed"]

    def update(task=None, fixed=False):
        state["step"] += 1
        if not fixed:
            subject.completed += 1
        return 1 / state["step"], {
            c: True
            for c in (
                "initializer",
                "evolution",
                "adapter",
                "completion_head",
                "latent_initializer_rows",
                "latent_evolution_rows",
            )
        }

    subject.checkpoint, subject.restore, subject.update = checkpoint, restore, update
    monkeypatch.setattr(
        runner, "process_tokens", lambda: [f"producer{i}" for i in range(4)]
    )
    monkeypatch.setattr(
        runner, "gather_rank_rng", lambda **kwargs: {"ranks": list(range(4))}
    )
    monkeypatch.setattr(runner, "selection_score", lambda *a, **k: 1.0)
    monkeypatch.setattr(runner.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(runner.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        runner.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value] * 4),
    )
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda: 100)
    subject.qualify()
    pending = json.loads((tmp_path / "QUALIFICATION_PENDING.json").read_text())
    assert not (tmp_path / "QUALIFIED.json").exists()
    assert retention_calls == ([True] if arm != "observation-only" else [])
    assert pending["diagnostic_only"]
    resume = pending["quarter_evidence"]["resume"]
    assert resume["tasks"] == runner.training_paths(arm)
    assert set(pending["quarter_evidence"]["fit"]["losses"]) == set(resume["tasks"])
    assert (
        resume["in_process_passed"]
        and not resume["passed"]
        and not resume["fresh_process"]
    )
    for task in resume["tasks"]:
        inputs = runner.replay_inputs(tmp_path, pending, task)
        expected = torch.load(tmp_path / inputs["expected"]["file"], weights_only=False)
        assert (
            expected["diagnostic_only"]
            and expected["quarter_contract"] == subject.contract
        )
    with pytest.raises(ValueError, match="resume"):
        runner.check_qualification(pending, subject.contract)


@pytest.mark.parametrize(
    "arm,latent,name",
    [
        ("u-multitask-early", 0, "U-early-quarter"),
        ("u-multitask-early-latent", 10, "U-early-latent-quarter"),
    ],
)
def test_early_arm_science_and_exact_exposures(tmp_path, arm, latent, name):
    supplied = args(tmp_path, arm=arm)
    supplied.name = name
    actual = runner.pilot_arguments(supplied)
    baseline = runner.pilot_arguments(args(tmp_path, arm="u-global"))
    assert actual.name == name and actual.latent_channels == latent
    assert (
        actual.evolution_architecture
        == baseline.evolution_architecture
        == "extent-unet"
    )
    differing = {
        key for key in vars(actual) if getattr(actual, key) != getattr(baseline, key)
    }
    assert differing <= {"name", "latent_channels"}
    schedule = TaskSchedule(*runner.ARMS[arm])
    paths = [runner.scheduled_path(arm, schedule, step) for step in range(6000)]
    assert {path: paths.count(path) for path in set(paths)} == dict(
        recent=1500, early=1500, observation=3000
    )
    assert [path for path in paths if path != "observation"] == [
        "recent",
        "early",
    ] * 1500
    assert runner.exposure_counts(arm, schedule, 6000) == dict(
        om4=3000, observation=3000, om4_recent=1500, om4_early=1500
    )


def test_early_sampling_preserves_counter_seeds_and_global_eight():
    for completed, path in (
        (0, "recent"),
        (1, "early"),
        (98, "recent"),
        (2999, "early"),
    ):
        all_samples = []
        for rank in range(4):
            actual = runner.path_samples(
                path,
                recent_size=2800,
                observation_size=243,
                completed_om4=completed,
                completed_observation=20,
                rank=rank,
                world_size=4,
            )
            expected = runner.rank_samples(
                task="om4",
                dataset_size=2800,
                observation_size=243,
                seed=1729,
                completed_task_updates=completed,
                rank=rank,
                world_size=4,
            )
            for sample, original in zip(actual, expected, strict=True):
                assert sample.mask_seed == original.mask_seed
                assert sample.coverage_index == original.coverage_index
                assert sample.sample_index == (
                    int(np.random.default_rng(original.mask_seed).integers(1217))
                    if path == "early"
                    else original.sample_index
                )
            all_samples.extend(actual)
        assert (
            len(all_samples) == len({sample.mask_seed for sample in all_samples}) == 8
        )


@pytest.mark.parametrize("missing", [None, "initializer", "evolution"])
def test_every_latent_output_row_must_receive_gradient(missing):
    model = SimpleNamespace(latent_channels=10, physical_channels=77)
    for name, count in (("initializer", 174), ("evolution", 87)):
        head = torch.nn.Conv2d(2, count, 1)
        head.weight.grad = torch.ones_like(head.weight)
        if name == missing:
            head.weight.grad[86] = 0
        setattr(model, name, SimpleNamespace(net=[head]))
    if missing:
        with pytest.raises(ValueError, match="output-row gradients"):
            runner.latent_row_gradients(model)
    else:
        assert runner.latent_row_gradients(model) == dict(
            latent_initializer_rows=True, latent_evolution_rows=True
        )
    assert runner.latent_row_gradients(SimpleNamespace()) == {}


def test_early_objective_routes_only_dataset_and_preserves_recent(monkeypatch):
    from samudra.experiments import observation_joint

    subject: Any = object.__new__(runner.QuarterRunner)
    source = SimpleNamespace(
        trainset="recent",
        earlyset="early",
        model_sample="reader",
        mask="mask",
        names="names",
        weights="weights",
        lat="lat",
    )
    subject.om4 = source
    calls = []
    monkeypatch.setattr(
        observation_joint, "om4_objective", lambda *values: calls.append(values)
    )
    micro = SimpleNamespace(sample_index=19, mask_seed=123)
    subject.om4_objective("model", (micro, "coverage"))
    subject.om4_objective("model", (micro, "coverage", "early"))
    assert calls[0][1] is source and calls[1][1].trainset == "early"
    assert source.trainset == "recent"
    assert calls[0][2:] == calls[1][2:] == ([19], 0.1, 123, "coverage", 0.1)


@pytest.fixture
def early_audit_fixture():
    import datetime

    import cftime
    import xarray as xr

    from samudra.experiments.quarter_data import QuarterOm4Windows
    from samudra.utils.data import ChannelStatistics, Masks

    times = [cftime.DatetimeJulian(1960, 9, 29)]
    # Calendar points match the published December 28 -> January 3 leap gap.
    while times[-1] < cftime.DatetimeJulian(1961, 5, 1):
        date = times[-1]
        times.append(
            date
            + datetime.timedelta(days=6 if (date.month, date.day) == (12, 28) else 5)
        )
    names = ("thetao_0", "so_0", "zos")
    channels = (*names, "tauuo", "tauvo", "hfds")
    values = np.arange(len(times) * 6 * 6, dtype="f4").reshape(len(times), 6, 2, 3) / 17
    values[:, :, 0, 0] = np.nan
    mask = torch.ones(3, 2, 3, dtype=torch.bool)
    mask[:, 0, 0] = False

    def statistics(channels):
        return ChannelStatistics(
            np.arange(len(channels), dtype="f4") + 0.25,
            np.arange(len(channels), dtype="f4") + 1.5,
        )

    source = SimpleNamespace(
        data_layout=SimpleNamespace(prognostic_var_names=names),
        time=SimpleNamespace(values=np.array(times)),
        resolution=(torch.tensor([-45.0, 45.0]), torch.tensor([0.0, 120.0, 240.0])),
        masks=Masks(mask, mask[0]),
        statistics=statistics,
    )

    class Reader:
        def __init__(self):
            self.calls = []

        def physical_indices(self, indices):
            return indices

        def read_unique(self, indices, *, channels):
            self.calls.append((indices.copy(), channels))
            return torch.from_numpy(
                values[indices][
                    :, [(*names, "tauuo", "tauvo", "hfds").index(c) for c in channels]
                ].copy()
            )

    reader = Reader()
    window = QuarterOm4Windows(
        source,  # type: ignore[arg-type]
        reader,
        observation_statistics=ChannelStatistics(
            np.array([3.0, 2.0, 1.0], dtype="f4"), np.array([2.0, 3.0, 4.0], dtype="f4")
        ),
    )
    data = {
        name: (("time", "lev", "y", "x"), values[:, i : i + 1])
        if "_" in name
        else (("time", "y", "x"), values[:, i])
        for i, name in enumerate(channels)
    }
    data = {name.split("_")[0]: value for name, value in data.items()}
    data["wetmask"] = (("lev", "y", "x"), mask[:1].numpy())
    raw = xr.Dataset(
        data, coords=dict(time=times, lev=[0], y=[-45.0, 45.0], x=[0.0, 120.0, 240.0])
    )
    return window, reader, raw


def test_actual_early_audit_reads_first_last_leap_bounded(early_audit_fixture):
    window, reader, raw = early_audit_fixture
    result = runner.audit_early_windows(window, window, raw)
    assert result["passed"] and result["shared_masks_statistics"]
    assert result["windows"]["first"]["origin_index"] == 0
    assert result["windows"]["last"]["origin_index"] == len(window) - 1
    assert len(reader.calls) == 9 and window.reader is reader
    assert [len(indices) for indices, channels in reader.calls] == [19, 8, 24] * 3
    for case in result["windows"].values():
        assert case["normalization_checked"] and case["reads"] == 3
        assert case["maximum_native_request_bytes"] <= 24 * 3 * 2 * 3 * 4


@pytest.mark.parametrize("corrupt", ["value", "mask", "time", "normalization"])
def test_early_audit_rejects_drift_and_restores_reader(early_audit_fixture, corrupt):
    window, reader, raw = early_audit_fixture
    if corrupt == "value":
        raw["thetao"][0, 0, 1, 1] += 1
        # Raw shares fixture backing; force reader to return a discrepant plane.
        read = reader.read_unique

        def changed(indices, *, channels):
            result = read(indices, channels=channels)
            result[0, 0, 1, 1] += 1
            return result

        reader.read_unique = changed
    elif corrupt == "mask":
        raw["wetmask"][0, 1, 1] = False
        window.mask = window.mask.clone()
        window.mask[:, 1, 1] = True
    elif corrupt == "time":
        raw = raw.assign_coords(time=raw.time.values[::-1])
    else:
        sample = window.model_sample

        def changed_sample(ids):
            result = list(sample(ids))
            result[0] = result[0] + 1
            return tuple(result)

        window.model_sample = changed_sample
    with pytest.raises(AssertionError):
        runner.audit_early_windows(window, window, raw)
    assert window.reader is reader


def valid_early_audit():
    return dict(
        passed=True,
        frames=1241,
        origins=1217,
        shared_masks_statistics=True,
        windows={
            name: dict(
                equal_xarray=True,
                six_tensors_finite=True,
                reads=3,
                normalization_checked=True,
            )
            for name in ("first", "last", "leap")
        },
    )


@pytest.mark.parametrize("arm", list(runner.EARLY_ARMS))
@pytest.mark.parametrize(
    "missing", [None, "early_audit", "normalization", "latent_rows"]
)
def test_early_final_gate_requires_three_restarts_and_real_audit(
    tmp_path, arm, missing
):
    tasks = runner.training_paths(arm)
    pending = pending_fixture(tmp_path, tasks, arm=arm)
    evidence = pending["quarter_evidence"]
    if missing == "early_audit":
        del evidence["io"]["early_window_audit"]
    elif missing == "normalization":
        evidence["io"]["early_window_audit"]["windows"]["leap"][
            "normalization_checked"
        ] = False
    elif missing == "latent_rows":
        evidence["fit"]["gradients"].pop("early/latent_initializer_rows", None)
    runner.atomic_json(pending, tmp_path / "QUALIFICATION_PENDING.json")
    for task in tasks[:-1]:
        runner.atomic_json(
            replay_receipt(tmp_path, pending, task),
            tmp_path / f"qualification-replay-{task}.json",
        )
        assert not runner.finalize_qualification(tmp_path, pending["quarter_contract"])
    task = tasks[-1]
    runner.atomic_json(
        replay_receipt(tmp_path, pending, task),
        tmp_path / f"qualification-replay-{task}.json",
    )
    failure = missing in {"early_audit", "normalization"} or (
        missing == "latent_rows" and runner.EARLY_ARMS[arm]
    )
    if failure:
        with pytest.raises(ValueError, match="early|gradients"):
            runner.finalize_qualification(tmp_path, pending["quarter_contract"])
        assert not (tmp_path / "QUALIFIED.json").exists()
    else:
        assert runner.finalize_qualification(tmp_path, pending["quarter_contract"])
        final = json.loads((tmp_path / "QUALIFIED.json").read_text())
        assert set(final["quarter_evidence"]["resume"]["receipts"]) == set(tasks)


@pytest.mark.parametrize("task", ["recent", "early", "observation"])
def test_early_fresh_replay_uses_path_and_subtype_counters(tmp_path, monkeypatch, task):
    arm = "u-multitask-early-latent"
    pending = pending_fixture(tmp_path, runner.training_paths(arm), arm=arm)
    schedule = TaskSchedule(*runner.ARMS[arm])
    completed = next(
        index
        for index in range(6000)
        if runner.scheduled_path(arm, schedule, index) == task
    )
    expected = dict(
        diagnostic_only=True,
        quarter_contract=pending["quarter_contract"],
        completed=completed + 1,
        counts=runner.exposure_counts(arm, schedule, completed + 1),
        model={"weights": torch.tensor(1.0)},
        optimizer={"step": 1},
        rng={"ranks": [torch.tensor(rank) for rank in range(4)]},
    )
    path = tmp_path / f"qualification-expected-{task}.pt"
    runner.atomic_torch(expected, path)
    pending["quarter_evidence"]["resume"]["inputs"][task]["expected"]["sha256"] = (
        runner.digest(path)
    )
    runner.atomic_json(pending, tmp_path / "QUALIFICATION_PENDING.json")
    subject: Any = object.__new__(runner.QuarterRunner)
    subject.contract, subject.schedule = pending["quarter_contract"], schedule
    subject.completed, subject.rank, subject.out = completed, 0, tmp_path
    updates = []
    subject.restore = lambda path: None
    subject.update = lambda: updates.append(subject.path())
    subject.replay_state = lambda: expected
    monkeypatch.setattr(
        runner, "process_tokens", lambda: [f"{task}{rank}" for rank in range(4)]
    )
    monkeypatch.setattr(runner.dist, "barrier", lambda: None)
    subject.qualify_replay(task)
    assert updates == [task]
    receipt = json.loads((tmp_path / f"qualification-replay-{task}.json").read_text())
    assert receipt["task"] == task and receipt["passed"] and receipt["diagnostic_only"]
    assert not (tmp_path / "QUALIFIED.json").exists()
