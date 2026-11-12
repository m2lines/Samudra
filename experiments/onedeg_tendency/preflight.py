# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the frozen training config and record every tendency weight."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from prepare_stats import open_canonical

from samudra.config import TrainConfig, build_loss_fn
from samudra.constants import build_om4_layout
from samudra.utils.location import LocalLocation
from samudra.utils.loss import TendencyLoss


def main():
    cfg = TrainConfig.from_yaml(sys.argv[1])
    bundle = Path(sys.argv[2])
    channels = tuple(build_om4_layout().prognostic_var_names)
    root = cfg.experiment.resolved_data_root
    assert isinstance(root, LocalLocation)
    # Locations are Pydantic values; resolved_data_root handles local paths.
    truth, _, stds_ds, _ = open_canonical(cfg)
    stds = np.array([float(stds_ds[c]) for c in channels])
    loss = build_loss_fn(
        cfg.loss,
        torch.device("cpu"),
        len(channels),
        cfg.model.pad,
        data_root=cfg.experiment.resolved_data_root,
        channels=channels,
        state_stds=stds,
    )
    assert isinstance(loss, TendencyLoss)
    weights = loss.loss_scale_per_channel().numpy()
    stats = xr.open_zarr(bundle / "stats/persistence_rms.zarr")
    physical = np.array([float(stats[c]) for c in channels])
    frame = pd.DataFrame(
        {
            "channel": channels,
            "state_std": stds,
            "physical_change_rms": physical,
            "normalized_change_rms": physical / stds,
            "loss_weight": weights,
        }
    )
    scale_summary = {}
    if cfg.model.residual_scale_location is not None:
        layout = cfg.data.sources[0].build_layout()
        model = cfg.model.build(
            prog_channels=cfg.data.input_steps * len(channels),
            boundary_channels=cfg.data.input_steps * len(layout.boundary_var_names),
            out_channels=cfg.data.output_steps * len(channels),
            input_steps=cfg.data.input_steps,
            grid_sizes=[(180, 360)],
        )
        cfg.model.configure_residual_scaling(
            model,
            data_root=root,
            channels=channels,
            state_stds=stds,
        )
        assert model.residual_scale is not None
        expected = (
            torch.from_numpy(physical / stds).float().repeat(cfg.data.output_steps)
        )
        torch.testing.assert_close(model.residual_scale, expected)
        prediction = model._assemble_prediction(
            torch.zeros(1, cfg.data.input_steps * len(channels), 1, 1),
            torch.ones(1, model.out_channels, 1, 1),
        )
        torch.testing.assert_close(prediction.flatten(), expected)
        frame["residual_output_scale"] = physical / stds
        scale_summary = {
            "residual_scale_min": float(expected.min()),
            "residual_scale_max": float(expected.max()),
            "residual_scale_checkpoint_key": "residual_scale",
        }
    frame.to_csv(bundle / "stats/weights.csv", index=False)
    assert cfg.model.pred_residuals
    assert cfg.data.input_steps == 2
    assert cfg.data.output_steps in (1, 2)
    assert cfg.steps[0] * cfg.data.output_steps == 8
    assert len(channels) == 77
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert truth.sizes["lat"] == 180 and truth.sizes["lon"] == 360
    summary = {
        "data_root": str(root.path),
        "data_attrs": truth.attrs,
        "calendar": truth.time.dt.calendar,
        "shape": dict(truth.sizes),
        "training_stats": dict(stats.attrs),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_ratio": float(weights.max() / weights.min()),
        "config": cfg.model_dump(mode="json"),
        **scale_summary,
    }
    (bundle / f"stats/provenance_{Path(sys.argv[1]).stem}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(frame.to_string(index=False))
    print("Preflight passed; maximum weight ratio:", summary["weight_ratio"])


if __name__ == "__main__":
    main()
