# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the selected EMA state and reproduce its validation rollout score."""

import argparse
import gc
import hashlib
import json
import logging
import math
from pathlib import Path

import torch

from samudra.config import TrainConfig
from samudra.train import Trainer
from samudra.utils.device import set_device
from samudra.utils.ema import EMATracker


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(training_config: Path, source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    set_device(torch.device("cpu"))
    cfg = TrainConfig.from_yaml(training_config)
    layout = cfg.data.sources[0].build_layout()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    assert checkpoint["epoch"] == 40
    assert checkpoint["validation_checkpoint_identity"] == {
        "metric": "rollout_rmse",
        "horizon": {"days": 360},
    }
    model = cfg.model.build(
        prog_channels=cfg.data.input_steps * len(layout.prognostic_var_names),
        boundary_channels=cfg.data.input_steps * len(layout.boundary_var_names),
        out_channels=cfg.data.output_steps * len(layout.prognostic_var_names),
        input_steps=cfg.data.input_steps,
        grid_sizes=[(180, 360)],
    )
    model.load_state_dict(
        {
            name.removeprefix("module."): value
            for name, value in checkpoint["model"].items()
        }
    )
    ema = EMATracker.from_state(checkpoint["ema"], model)
    ema.copy_to(model)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        if parameter.requires_grad:
            key = name.removeprefix("module.").replace(".", "")
            assert torch.equal(parameter, checkpoint["ema"]["ema_params"][key]), name
    expected_score = float(checkpoint["best_validation_score"])
    exported = {
        "model": model.state_dict(),
        "epoch": checkpoint["epoch"],
        "weight_kind": "ema",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256(source),
        "best_validation_score": expected_score,
        "validation_checkpoint_identity": checkpoint["validation_checkpoint_identity"],
        "wandb_id": checkpoint["wandb_id"],
    }
    torch.save(exported, output)
    readback = torch.load(output, map_location="cpu", weights_only=True, mmap=True)
    assert all(
        torch.equal(value, readback["model"][key])
        for key, value in model.state_dict().items()
    )
    report = {key: value for key, value in exported.items() if key != "model"}
    report["exported_checkpoint_sha256"] = sha256(output)
    del model, checkpoint, ema, exported
    gc.collect()

    # Validate the actual exported weights, without replacing them with a new EMA
    # tracker. This is a score check only: no training updates are performed.
    cfg.backend = "cuda"
    cfg.resume_ckpt_path = None
    cfg.post_train_eval = None
    cfg.test_using_ema = False
    cfg.experiment.name = "epoch40-export-validation-recheck"
    cfg.experiment.base_output_dir = str(output.parent / "preflight")
    cfg.experiment.wandb.mode = "disabled"
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(readback["model"])
    scores = trainer.validate_rollout_one_epoch(40)
    observed_score = float(scores["rollout_val/360d/normalized_rmse/channel_mean"])
    assert math.isclose(observed_score, expected_score, rel_tol=1e-4, abs_tol=1e-6), (
        observed_score,
        expected_score,
    )
    report["rechecked_validation_score"] = observed_score
    report["passed"] = True
    trainer.finish()
    (output.parent / "ema-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_config", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.training_config, args.source, args.output)
