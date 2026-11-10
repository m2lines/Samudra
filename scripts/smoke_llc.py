# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Qualify compact LLC loading, training, checkpoint reload and saved evaluation."""

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from samudra.config import EvalConfig, TrainConfig
from samudra.eval import Eval
from samudra.train import Trainer
from samudra.utils.logging import handle_logging
from samudra.utils.multiton import MultitonScope


def model_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    values = TrainConfig.from_yaml("samudra_llc/train.yaml").model_dump(mode="json")
    values.update(
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        steps=[1],
        save_freq=1,
        backend=args.backend,
        rollout_validation={"model_steps": 2, "frequency": 1, "steps_forward": 1},
    )
    values["experiment"].update(
        name="train", base_output_dir=str(args.output), data_root=str(args.store.parent)
    )
    values["experiment"]["wandb"]["mode"] = "disabled"
    values["model"]["unet"].update(ch_width=[8, 16], dilation=[1, 1], n_layers=[1, 1])
    values["model"].update(use_bfloat16=False, checkpointing=None)
    values["data"].update(
        input_steps=2,
        output_steps=1,
        loading={"type": "cpu", "num_workers": 0},
        inference_loading={"num_workers": 0},
    )
    source = values["data"]["sources"][0]
    source.update(
        data_location=str(args.store),
        data_means_location=str(args.store),
        data_stds_location=str(args.store),
        train_time={"start": "2011-09-13T00:00:00Z", "end": "2011-09-13T05:00:00Z"},
        val_time={"start": "2012-09-14T00:00:00Z", "end": "2012-09-14T05:00:00Z"},
        inference_time={"start": "2012-09-14T00:00:00Z", "end": "2012-09-14T05:00:00Z"},
    )
    cfg = TrainConfig.model_validate(values)
    cfg.save_yaml(args.output / "train.yaml")
    cfg.prepare_output_dirs()
    handle_logging(False, cfg.experiment.output_dir)
    with MultitonScope():
        trainer = Trainer(cfg)
        initial_digest = model_digest(trainer.model.state_dict())
        parameter_count = sum(p.numel() for p in trainer.model.parameters())
        trainer.run()
        checkpoint_path = trainer.ckpt_paths.latest_checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert checkpoint["train_progress"]["optimizer_steps"] == 4
        assert math.isfinite(checkpoint["best_validation_score"])
        assert all(
            torch.isfinite(value).all() for value in checkpoint["model"].values()
        )
        final_digest = model_digest(checkpoint["model"])
        assert initial_digest != final_digest, (
            "Training did not update model parameters"
        )
        trainer.finish()

    evaluation = EvalConfig.model_validate(
        {
            "backend": args.backend,
            "save_zarr": True,
            "num_model_steps_forward": 2,
            "ckpt_path": str(checkpoint_path),
            "model": values["model"],
            "data": values["data"],
            "experiment": {**values["experiment"], "name": "eval"},
        }
    )
    evaluation.save_yaml(args.output / "eval.yaml")
    with MultitonScope():
        metrics = Eval(evaluation).run()
    scalars = {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, numbers.Real)
    }
    assert scalars and all(math.isfinite(value) for value in scalars.values())
    assert any("rmse" in name for name in scalars), "Evaluation did not report RMSE"
    prediction_path = evaluation.experiment.output_dir / "predictions.zarr"
    with (
        xr.open_zarr(prediction_path) as predictions,
        xr.open_zarr(args.store) as physical,
    ):
        assert predictions.sizes["time"] == 4
        assert {"U", "V", "Theta", "Salt", "Eta"}.issubset(predictions.data_vars)
        for name in ("U", "V", "Theta", "Salt", "Eta"):
            mask = physical.wet_area.to_numpy() > 0
            if name == "Eta":
                mask = mask[0]
            array = predictions[name].to_numpy()
            assert np.isfinite(array[:, mask]).all(), name
            assert np.isnan(array[:, ~mask]).all(), name
        # Writer emits 2-D lat/lon on y/x axes for analysis-ready outputs.
        np.testing.assert_allclose(predictions.lat.isel(x=0), physical.lat)
        np.testing.assert_allclose(predictions.lon.isel(y=0), physical.lon)
        np.testing.assert_array_equal(predictions.areacello, physical.rA)
        np.testing.assert_array_equal(
            predictions.time,
            physical.time.sel(time=slice("2012-09-14T02:00:00", "2012-09-14T05:00:00")),
        )
    receipt = {
        "commit": args.commit,
        "store": str(args.store),
        "metadata_sha256": hashlib.sha256(
            (args.store / ".zmetadata").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "backend": args.backend,
        "parameter_count": parameter_count,
        "prognostic_channels": len(trainer.prognostic_var_names),
        "boundary_channels": len(trainer.boundary_var_names),
        "optimizer_steps": checkpoint["train_progress"]["optimizer_steps"],
        "initial_model_sha256": initial_digest,
        "trained_model_sha256": final_digest,
        "checkpoint": str(checkpoint_path),
        "validation_score": checkpoint["best_validation_score"],
        "predictions": str(prediction_path),
        "prediction_steps": 4,
        "metrics": scalars,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
