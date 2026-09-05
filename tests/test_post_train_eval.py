# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from samudra.config import PostTrainEvalConfig
from samudra.post_train_eval import (
    _resolve_workers,
    checkpoint_label,
    discover_checkpoints_from_directory,
    partition_checkpoint_work,
    run_checkpoint_sweep,
)
from samudra.utils.location import LocalLocation
from samudra.utils.train import CheckpointPaths
from samudra.viz import VizTemplate


def test_resolve_workers_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_workers("auto", num_checkpoints=3) == [torch.device("cpu")]


def test_resolve_workers_assigns_one_checkpoint_per_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert _resolve_workers("cuda", num_checkpoints=2) == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    for epoch in range(1, 8):
        torch.save({"epoch": epoch}, tmp_path / f"ckpt_{epoch}.pt")
    torch.save({"epoch": 7}, tmp_path / "ema_ckpt.pt")
    # Files the sweep should ignore.
    torch.save({"epoch": 7}, tmp_path / "ckpt.pt")
    torch.save({"epoch": 3}, tmp_path / "best_inference_ckpt.pt")
    return tmp_path


def test_discovers_all_periodic_plus_ema(checkpoint_dir: Path):
    targets = discover_checkpoints_from_directory(CheckpointPaths(checkpoint_dir))
    assert [target.name for target in targets] == [
        *[f"ckpt_{epoch}.pt" for epoch in range(1, 8)],
        "ema_ckpt.pt",
    ]
    assert checkpoint_label(targets[0]) == "epoch_0001"
    assert checkpoint_label(targets[-1]) == "ema_latest"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("ckpt_12.pt", 12),
        ("ckpt_latest.pt", None),
        ("ckpt_12.pth", None),
        ("checkpoint_12.pt", None),
    ],
)
def test_periodic_checkpoint_epoch(filename: str, expected: int | None):
    assert CheckpointPaths.periodic_checkpoint_epoch(Path(filename)) == expected


def test_last_n_counts_only_periodic_checkpoints(checkpoint_dir: Path):
    targets = discover_checkpoints_from_directory(
        CheckpointPaths(checkpoint_dir), last_n_checkpoints=3
    )
    # The EMA checkpoint is added on top of the last N periodic checkpoints,
    # not counted against them.
    assert [target.name for target in targets] == [
        "ckpt_5.pt",
        "ckpt_6.pt",
        "ckpt_7.pt",
        "ema_ckpt.pt",
    ]


def test_explicit_checkpoints_plus_ema(checkpoint_dir: Path):
    targets = discover_checkpoints_from_directory(
        CheckpointPaths(checkpoint_dir), checkpoints=[2, 5]
    )
    assert [target.name for target in targets] == [
        "ckpt_2.pt",
        "ckpt_5.pt",
        "ema_ckpt.pt",
    ]


def test_missing_explicit_checkpoint_raises(checkpoint_dir: Path):
    with pytest.raises(ValueError, match="not found"):
        discover_checkpoints_from_directory(
            CheckpointPaths(checkpoint_dir), checkpoints=[2, 99]
        )


def test_mutually_exclusive_selection(checkpoint_dir: Path):
    with pytest.raises(ValueError, match="only one"):
        discover_checkpoints_from_directory(
            CheckpointPaths(checkpoint_dir), last_n_checkpoints=2, checkpoints=[1]
        )


def test_last_n_must_be_positive(checkpoint_dir: Path):
    with pytest.raises(ValueError, match=">= 1"):
        discover_checkpoints_from_directory(
            CheckpointPaths(checkpoint_dir), last_n_checkpoints=0
        )


def test_no_ema_checkpoint(tmp_path: Path):
    torch.save({"epoch": 1}, tmp_path / "ckpt_1.pt")
    targets = discover_checkpoints_from_directory(CheckpointPaths(tmp_path))
    assert targets == [tmp_path / "ckpt_1.pt"]


def test_partition_checkpoint_work_covers_all_entries():
    entries = [Path(f"p{i}") for i in range(5)]
    shards = partition_checkpoint_work(entries, 2)
    assert len(shards) == 2
    assert sorted(path.name for shard in shards for path in shard) == [
        f"p{i}" for i in range(5)
    ]


def test_single_worker_uses_process_path(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt_1.pt"
    checkpoint_path.touch()
    result = {"checkpoint_path": str(checkpoint_path)}
    work_queue = Mock()
    work_queue.get.return_value = result
    process = Mock(exitcode=0)
    context = Mock()
    context.Queue.return_value = work_queue
    context.Process.return_value = process
    monkeypatch.setattr(
        "samudra.post_train_eval._load_eval_config",
        lambda *_: SimpleNamespace(backend="cpu"),
    )
    monkeypatch.setattr(
        "samudra.post_train_eval.multiprocessing.get_context", lambda _: context
    )

    results = run_checkpoint_sweep(
        eval_config_path=Path("eval.yaml"),
        checkpoint_paths=CheckpointPaths(tmp_path),
        data_root=LocalLocation(path=tmp_path.resolve()),
        sweep_output_dir=tmp_path / "evals",
    )

    assert results == [result]
    context.Process.assert_called_once()
    process.start.assert_called_once()


def test_sweep_config_builds_runtime_object(tmp_path: Path, monkeypatch):
    viz_loader = Mock()
    monkeypatch.setattr(
        "samudra.post_train_eval._load_eval_config",
        lambda *_: SimpleNamespace(save_zarr=True),
    )
    monkeypatch.setattr("samudra.viz.VizTemplateConfig.from_yaml", viz_loader)
    cfg = PostTrainEvalConfig(
        eval_config_path=Path("eval.yaml"),
        viz_config_path=Path("viz.yaml"),
        eval_dirname="checkpoint_evals",
        epochs=[2, 5],
    )
    data_root = LocalLocation(path=tmp_path.resolve())

    sweep = cfg.build(
        nets_dir=tmp_path / "saved_nets",
        output_dir=tmp_path,
        data_root=data_root,
    )

    assert sweep.eval_config_path == Path("eval.yaml")
    assert sweep.checkpoint_paths.checkpoint_dir == tmp_path / "saved_nets"
    assert sweep.sweep_output_dir == tmp_path / "checkpoint_evals"
    assert sweep.viz_config_path == Path("viz.yaml")
    assert sweep.checkpoints == [2, 5]
    assert sweep.data_root == data_root
    viz_loader.assert_called_once_with(Path("viz.yaml"))


def test_sweep_config_validates_viz_output_early(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "samudra.post_train_eval._load_eval_config",
        lambda *_: SimpleNamespace(save_zarr=False),
    )
    cfg = PostTrainEvalConfig(
        eval_config_path=Path("eval.yaml"),
        viz_config_path=Path("viz.yaml"),
    )

    with pytest.raises(ValueError, match="save_zarr = true"):
        cfg.build(
            nets_dir=tmp_path / "saved_nets",
            output_dir=tmp_path,
            data_root=LocalLocation(path=tmp_path.resolve()),
        )


def test_sweep_config_validates_eval_config_early(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "samudra.post_train_eval._load_eval_config",
        Mock(side_effect=ValueError("invalid eval config")),
    )
    cfg = PostTrainEvalConfig(eval_config_path=Path("eval.yaml"))

    with pytest.raises(ValueError, match="invalid eval config"):
        cfg.build(
            nets_dir=tmp_path / "saved_nets",
            output_dir=tmp_path,
            data_root=LocalLocation(path=tmp_path.resolve()),
        )


def test_sweep_config_validates_viz_config_early(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "samudra.post_train_eval._load_eval_config",
        lambda *_: SimpleNamespace(save_zarr=True),
    )
    monkeypatch.setattr(
        "samudra.viz.VizTemplateConfig.from_yaml",
        Mock(side_effect=ValueError("invalid viz config")),
    )
    cfg = PostTrainEvalConfig(
        eval_config_path=Path("eval.yaml"), viz_config_path=Path("viz.yaml")
    )

    with pytest.raises(ValueError, match="invalid viz config"):
        cfg.build(
            nets_dir=tmp_path / "saved_nets",
            output_dir=tmp_path,
            data_root=LocalLocation(path=tmp_path.resolve()),
        )


def test_viz_template_instantiate_run_owns_run_defaults(tmp_path: Path, monkeypatch):
    data = Mock()
    resolved_location = Mock()
    resolved_location.open.return_value = data
    data_root = Mock()
    data_root.resolve.return_value = resolved_location
    template = VizTemplate("dataset", data_root, ["thetao"], Mock())
    instantiate = Mock(return_value=Mock())
    monkeypatch.setattr(template, "instantiate", instantiate)
    location = LocalLocation(path=tmp_path.resolve())

    result = template.instantiate_run(tmp_path / "viz", "epoch_0001", location)

    run = instantiate.call_args.args[1][0]
    assert result is instantiate.return_value
    assert run.name == "epoch_0001"
    assert run.data is data
    assert run.variables == ["thetao"]
    data_root.resolve.assert_called_once_with(location)
