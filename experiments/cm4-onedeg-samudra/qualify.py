# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify real CM4 native reads and a GPU update before full training."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import torch

from samudra.config import CpuDataLoadingConfig, TrainConfig
from samudra.native_reader import NativeOm4Reader
from samudra.train import Trainer


def qualify(config_path: Path, output: Path) -> None:
    cfg = TrainConfig.from_yaml(config_path)
    native = cfg.data.build(cfg.experiment.resolved_data_root, include_inference=True)
    cpu_data = cfg.data.model_copy(update={"loading": CpuDataLoadingConfig()})
    reference = cpu_data.build(
        cfg.experiment.resolved_data_root, include_inference=True
    )
    assert native.inference_source is not None
    assert reference.inference_source is not None
    channels = tuple(
        native.data_layout.prognostic_var_names + native.data_layout.boundary_var_names
    )
    assert len(channels) == 81
    report: dict[str, Any] = {"splits": {}, "channels": len(channels)}
    for label, source, baseline, expected in zip(
        ("train", "validation", "held_out"),
        (*native.train_sources, *native.val_sources, native.inference_source),
        (*reference.train_sources, *reference.val_sources, reference.inference_source),
        (5840, 1460, 7300),
        strict=True,
    ):
        assert source.time.size == expected
        assert all(isinstance(t, cftime.DatetimeNoLeap) for t in source.time.values)
        # Non-monotonic, repeated positions also exercise sliced physical indices.
        positions = np.array([expected - 1, 1, 0, 1], dtype=np.int64)
        # The training and validation sources must actually use native reads;
        # inference intentionally uses the existing Xarray/Dask rollout path.
        if label != "held_out":
            assert isinstance(source.reader, NativeOm4Reader)
            np.testing.assert_array_equal(
                source.read(positions, channels), baseline.read(positions, channels)
            )
            stats = source.statistics(channels)
            assert np.isfinite(stats.mean).all()
            assert np.isfinite(stats.std).all() and (stats.std > 0).all()
        report["splits"][label] = {
            "records": expected,
            "first": str(source.time.values[0]),
            "last": str(source.time.values[-1]),
            "checked_positions": positions.tolist() if label != "held_out" else [],
            "matches_xarray_exactly": True if label != "held_out" else None,
            "reader": type(source.reader).__name__,
        }
    del native, reference

    cfg.backend = "cuda"
    cfg.debug = True  # Limit ordinary validation to four processed batches.
    cfg.experiment.name += "-qualification"
    cfg.experiment.wandb.mode = "disabled"
    cfg.experiment.base_output_dir = str(
        Path(cfg.experiment.base_output_dir) / "qualification"
    )
    trainer = Trainer(cfg)
    assert isinstance(trainer.primary_source.reader, NativeOm4Reader)
    assert "wfo" in trainer.boundary_var_names
    trainer.probe_optimizer_step()
    assert trainer.train_progress.optimizer_steps == 1
    val_stats = trainer.validate_one_epoch(1)
    val_loss = val_stats["val/mean/loss"]
    rollout_stats = trainer.validate_rollout_one_epoch(1)
    assert math.isfinite(float(val_loss))
    assert rollout_stats, "The one-year no-leap validation rollout did not run"
    for key, value in (val_stats | rollout_stats).items():
        if isinstance(value, (float, int, torch.Tensor)):
            assert math.isfinite(float(value)), (key, value)
    report.update(
        optimizer_steps=trainer.train_progress.optimizer_steps,
        validation_loss=float(val_loss),
        rollout_metrics={
            key: float(value)
            for key, value in rollout_stats.items()
            if isinstance(value, (float, int, torch.Tensor))
        },
        cuda_device=torch.cuda.get_device_name(),
        source_commit=os.environ.get("SAMUDRA_CODE_COMMIT"),
        passed=True,
    )
    trainer.finish()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.config, args.output)
