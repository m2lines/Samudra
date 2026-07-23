# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import torch

from samudra.config import (
    CheckpointSweepConfig,
    EvalConfig,
    StandaloneCheckpointSweepConfig,
)
from samudra.post_train_eval import (
    CheckpointEvalTarget,
    CheckpointSweep,
    _run_single_checkpoint_eval,
)
from samudra.utils.location import LocalLocation


def test_checkpoint_sweep_config_has_no_enabled_switch():
    assert "enabled" not in CheckpointSweepConfig.model_fields


def test_checkpoint_sweep_config_accepts_built_eval(tmp_path):
    data_root = LocalLocation(path=tmp_path.resolve())
    evaluator = MagicMock()
    config = CheckpointSweepConfig(
        eval=EvalConfig(),
        backend="cpu",
    )

    sweep = config.build(
        evaluator=evaluator,
        data_root=data_root,
        nets_dir=tmp_path / "saved_nets",
        output_dir=tmp_path,
    )

    assert not hasattr(sweep, "build")
    assert sweep.eval_worker.evaluator is evaluator
    assert sweep.data_root is data_root
    assert sweep.eval_worker.backend == "cpu"


def test_checkpoint_eval_builds_with_sweep_dependencies(tmp_path):
    evaluator = MagicMock()
    evaluator.standalone_inference.return_value = {
        "scalar_tensor": torch.tensor(2.0),
        "non_scalar_tensor": torch.tensor([1.0, 2.0]),
    }
    entry = CheckpointEvalTarget(
        epoch=3,
        kind="periodic",
        path="ckpt_3.pt",
        for_inference=False,
    )
    sweep_root = tmp_path / "sweep"
    sweep = MagicMock(spec=CheckpointSweep)
    sweep.sweep_root = sweep_root
    eval_worker = MagicMock(return_value=evaluator)
    sweep.eval_worker = eval_worker

    result = _run_single_checkpoint_eval(
        entry=entry,
        sweep=sweep,
        gpu_index=None,
    )

    eval_worker.assert_called_once_with(entry, gpu_index=None)
    evaluator.standalone_inference.assert_called_once_with(
        output_dir=sweep_root / "epoch_0003",
        model_path=Path(entry.path),
        save_zarr=True,
    )
    assert (sweep_root / "epoch_0003").is_dir()
    assert result["output_dir"] == str(sweep_root / "epoch_0003")
    assert result["metrics"] == {"scalar_tensor": 2.0}


def test_standalone_sweep_uses_standard_config_cli(tmp_path):
    config = StandaloneCheckpointSweepConfig.from_yaml_and_cli(
        [
            "configs/test/standalone_checkpoint_sweep.yaml",
            "--checkpoint_dir",
            str(tmp_path / "saved_nets"),
            "--checkpoint_sweep.last_n_checkpoints",
            "2",
            "--checkpoint_sweep.backend",
            "cpu",
        ]
    )

    assert config.checkpoint_dir == tmp_path / "saved_nets"
    assert config.checkpoint_sweep.last_n_checkpoints == 2
    assert config.checkpoint_sweep.backend == "cpu"
