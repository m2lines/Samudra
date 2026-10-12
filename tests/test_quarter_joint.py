# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
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
        "resume": {"passed": True, "tasks": tasks},
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
