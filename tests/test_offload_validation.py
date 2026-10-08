import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ocean_emulators.offload_validate import (
    SnapshotScanner,
    SnapshotJob,
    ValidationResult,
    _default_state,
    _jsonable_config,
    _parse_offload_args,
    commit_result,
    strict_validation_score,
    validate_snapshot,
    validation_config_mismatches,
)
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope
from tests.test_face_training_e2e import _face_config, face_root


def _run_and_validator_configs(tmp_path):
    run = _jsonable_config(
        _face_config(
            tmp_path,
            **{
                "--face_parallel.blend_scope": "rank",
                "--validation_mode": "offload",
            },
        )
    )
    validator = copy.deepcopy(run)
    validator["face_parallel"]["blend_scope"] = "face"
    validator["face_parallel"]["tiles_per_chunk"] = 1
    validator["face_parallel"]["read_threads"] = 17
    validator["validation_mode"] = "inline"
    validator["model"]["checkpointing"] = "all"
    validator["data"]["data_location"] = "/a/separate/copy.zarr"
    validator["data"]["num_workers"] = 19
    validator["data"]["prefetch_factor"] = 7
    validator["replay"]["storage_dtype"] = "float32"
    validator["lr_warmup_steps"] = 999
    return run, validator


def test_config_guard_accepts_runtime_and_branch_only_differences(tmp_path):
    run, validator = _run_and_validator_configs(tmp_path)
    assert validation_config_mismatches(run, validator) == []


def test_config_guard_accepts_validator_local_workload_counts(tmp_path):
    run, validator = _run_and_validator_configs(tmp_path)
    validator["one_step_val_num"] = 25
    validator["short_autoregressive_val_num"] = 1
    validator["long_autoregressive_val_num"] = 0
    assert validation_config_mismatches(run, validator) == []


def test_config_guard_still_rejects_a_rollout_horizon_change(tmp_path):
    run, validator = _run_and_validator_configs(tmp_path)
    validator["short_autoregressive_val_length"] += 1
    mismatches = validation_config_mismatches(run, validator)
    assert any("short_autoregressive_val_length" in item for item in mismatches)


def test_offload_options_are_removed_before_config_parsing(tmp_path):
    options, remaining = _parse_offload_args(
        [
            "--offload.runs",
            f"{tmp_path / 'a'},{tmp_path / 'b'}",
            "--offload.once",
            "--offload.poll-seconds",
            "7",
            "config.yaml",
            "--epochs",
            "2",
        ]
    )

    assert options.runs == ((tmp_path / "a").resolve(), (tmp_path / "b").resolve())
    assert options.once is True
    assert options.poll_seconds == 7
    assert remaining == ["config.yaml", "--epochs", "2"]


def test_config_guard_rejects_a_validation_window_change(tmp_path):
    run, validator = _run_and_validator_configs(tmp_path)
    validator["val_time"]["end"] = "1976-01-01"
    mismatches = validation_config_mismatches(run, validator)
    assert any("val_time.end" in mismatch for mismatch in mismatches)


def test_scanner_takes_each_runs_lowest_epoch_then_uses_fifo_mtime(tmp_path):
    run_config, validator_config = _run_and_validator_configs(tmp_path)
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        (run / "saved_nets").mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(run_config))

    a1 = runs[0] / "saved_nets" / "ema_ckpt_ep0001.pt"
    a2 = runs[0] / "saved_nets" / "ema_ckpt_ep0002.pt"
    b1 = runs[1] / "saved_nets" / "ema_ckpt_ep0001.pt"
    for path in (a1, a2, b1):
        path.write_bytes(path.name.encode())
    # Run B's first unvalidated epoch is oldest, even though run A already has
    # a later epoch waiting. Epoch 2 in A can never jump ahead of epoch 1 in A.
    a1.touch()
    a2.touch()
    b1.touch()
    import os

    os.utime(b1, (1, 1))
    os.utime(a2, (2, 2))
    os.utime(a1, (3, 3))

    job = SnapshotScanner(runs, validator_config).next_job()
    assert job is not None
    assert job.run_dir == runs[1]
    assert job.epoch == 1


@pytest.mark.parametrize(
    ("one_step", "autoregressive", "expected_reason"),
    [
        (1.0, {"val/mean/short-autoregressive-loss": 2.0}, None),
        (float("nan"), {"val/mean/short-autoregressive-loss": 2.0}, "one-step"),
        (1.0, {"val/mean/short-autoregressive-loss": float("inf")}, "short"),
        (1.0, {}, "short"),
    ],
)
def test_strict_score_rejects_any_missing_or_nonfinite_active_term(
    one_step, autoregressive, expected_reason
):
    trainer = SimpleNamespace(
        due_autoregressive_val_specs=lambda epoch: [SimpleNamespace(label="short")]
    )
    score, reason = strict_validation_score(
        trainer, 1, one_step, autoregressive, combined_score=1.5
    )
    if expected_reason is None:
        assert score == 1.5
        assert reason is None
    else:
        assert score is None
        assert expected_reason in reason


def _result(epoch: int, score: float) -> ValidationResult:
    return ValidationResult(
        epoch=epoch,
        num_batches_seen=epoch * 10,
        metrics={"val/mean/one-step-loss": score},
        scalar_metrics={"val/mean/one-step-loss": score},
        curves={"short": {"loss_by_step": [score], "rmse_by_step": [score]}},
        combined_score=score,
        strict_score=score,
        rejection_reason=None,
        contributors=["one-step"],
        one_step_seconds=1.0,
        autoregressive_seconds=2.0,
        one_step_peak_gib=3.0,
        autoregressive_peak_gib=4.0,
    )


def test_each_run_keeps_independent_best_state_and_atomic_metrics(tmp_path):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run, score in zip(runs, (2.0, 1.0), strict=True):
        snapshot = run / "saved_nets" / "ema_ckpt_ep0001.pt"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(run.name.encode())
        job = SimpleNamespace(
            run_dir=run,
            snapshot=snapshot,
            epoch=1,
            state=_default_state(),
        )
        commit_result(job, _result(1, score), wandb_mode="disabled", cfg=None)

    states = [
        json.loads((run / "offload_val" / "state.json").read_text()) for run in runs
    ]
    assert [state["best_score"] for state in states] == [2.0, 1.0]
    assert [state["best_epoch"] for state in states] == [1, 1]
    for run in runs:
        best = run / "saved_nets" / "best_validation_ema_ckpt.pt"
        assert best.read_bytes() == run.name.encode()
        records = [
            json.loads(line)
            for line in (run / "offload_val" / "metrics.jsonl").read_text().splitlines()
        ]
        assert [record["epoch"] for record in records] == [1]


def test_nonfinite_curve_is_recorded_as_null_instead_of_losing_epoch(tmp_path):
    run = tmp_path / "run"
    snapshot = run / "saved_nets" / "ema_ckpt_ep0001.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"snapshot")
    result = _result(1, 2.0)
    result.curves["short"]["loss_by_step"] = [1.0, float("nan"), float("inf")]
    result.strict_score = None
    result.rejection_reason = "non-finite short rollout"
    job = SimpleNamespace(
        run_dir=run,
        snapshot=snapshot,
        epoch=1,
        state=_default_state(),
    )

    commit_result(job, result, wandb_mode="disabled", cfg=None)

    record = json.loads((run / "offload_val" / "metrics.jsonl").read_text())
    state = json.loads((run / "offload_val" / "state.json").read_text())
    assert record["curves"]["short"]["loss_by_step"] == [1.0, None, None]
    assert state["validated_epochs"] == [1]
    assert state["rejections"] == {"1": "non-finite short rollout"}
    assert not (run / "saved_nets" / "best_validation_ema_ckpt.pt").exists()


def _small_validation_config(face_root, name: str, mode: str):
    return _face_config(
        face_root,
        **{
            "--experiment.name": name,
            "--epochs": "2",
            "--validation_mode": mode,
            "--one_step_val_num": "2",
            "--short_autoregressive_val_length": "3",
            "--short_autoregressive_val_num": "1",
            "--long_autoregressive_val_length": "4",
            "--long_autoregressive_val_num": "1",
            "--long_autoregressive_val_start_epoch": "1",
            "--autoregressive_val_steps_forward": "2",
        },
    )


def _run_and_capture_inline(config):
    captured = {}
    with MultitonScope():
        trainer = Trainer(config)
        original_one_step = trainer.validate_one_epoch
        original_autoregressive = trainer.validate_autoregressive_one_epoch

        def one_step(epoch):
            result = original_one_step(epoch)
            captured.setdefault(epoch, {}).update(result)
            return result

        def autoregressive(epoch):
            result = original_autoregressive(epoch)
            captured.setdefault(epoch, {}).update(result)
            return result

        trainer.validate_one_epoch = one_step
        trainer.validate_autoregressive_one_epoch = autoregressive
        trainer.run()
        snapshots = sorted(trainer.ckpt_paths.checkpoint_dir.glob("ema_ckpt_ep*.pt"))
    return captured, snapshots


def test_offloaded_snapshot_matches_both_and_reproduces_inline_validation(face_root):
    both_config = _small_validation_config(face_root, "both", "both")
    inline, both_snapshots = _run_and_capture_inline(both_config)
    assert [path.name for path in both_snapshots] == [
        "ema_ckpt_ep0001.pt",
        "ema_ckpt_ep0002.pt",
    ]

    offload_config = _small_validation_config(face_root, "offload", "offload")
    _, offload_snapshots = _run_and_capture_inline(offload_config)
    both_checkpoint = torch.load(
        both_snapshots[-1], map_location="cpu", weights_only=False
    )
    offload_checkpoint = torch.load(
        offload_snapshots[-1], map_location="cpu", weights_only=False
    )
    assert both_checkpoint["model"].keys() == offload_checkpoint["model"].keys()
    for name in both_checkpoint["model"]:
        assert torch.equal(
            both_checkpoint["model"][name], offload_checkpoint["model"][name]
        ), name

    validator_config = _small_validation_config(face_root, "validator", "inline")
    with MultitonScope():
        validator = Trainer(validator_config)
        validator.test_using_ema = False
        stride = validator.get_current_temporal_stride(1)
        validator.temporal_stride = stride
        validator.init_data_loaders(
            max(validator_config.replay.max_lead_steps), stride
        )
        result = validate_snapshot(
            validator,
            SnapshotJob(
                run_index=0,
                run_dir=both_config.experiment.output_dir,
                snapshot=both_snapshots[-1],
                epoch=2,
                state=_default_state(),
            ),
        )

    for key, expected in inline[2].items():
        if isinstance(expected, (float, int)):
            assert result.scalar_metrics[key] == pytest.approx(expected, rel=0, abs=0)
