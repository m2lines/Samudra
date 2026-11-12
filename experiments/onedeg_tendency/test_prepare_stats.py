# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Prove the compact-store path gives the branch's original flat-store RMS."""

import sys
from pathlib import Path

import imprint_metrics
import numpy as np
import prepare_stats
import xarray as xr

from samudra.config import TrainConfig
from samudra.constants import build_om4_layout
from samudra.utils.location import UnresolvedLocation
from scripts.compute_tendency_stats import persistence_rms


def test_compact_statistics_equal_flat_statistics(tmp_path, monkeypatch):
    layout = build_om4_layout()
    rng = np.random.default_rng(2)
    time = xr.date_range(
        "1975-01-03", periods=10, freq="5D", calendar="julian", use_cftime=True
    )
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        v: (("lev", "time", "y", "x"), rng.normal(size=(19, 10, 2, 3)))
        for v in ("uo", "vo", "thetao", "so")
    }
    variables["zos"] = (("time", "y", "x"), rng.normal(size=(10, 2, 3)))
    for level in range(19):
        wet = np.ones((2, 3), dtype=bool)
        wet[0, 0] = False
        variables[f"mask_{level}"] = (("y", "x"), wet)
    data = xr.Dataset(
        variables,
        coords={
            "lev": list(layout.depth_levels),
            "time": time,
            "y": [-30.0, 30.0],
            "x": [0.0, 120.0, 240.0],
        },
    )
    data.to_zarr(tmp_path / "OM4.zarr")
    data.mean("time").to_zarr(tmp_path / "OM4_means.zarr")
    data.std("time").to_zarr(tmp_path / "OM4_stds.zarr")
    cfg = TrainConfig.from_yaml(
        Path(__file__).parents[2]
        / "src/samudra/configs/experiments/onedeg_tendency/train.yaml"
    )
    cfg.experiment.data_root = UnresolvedLocation(path=str(tmp_path))
    cfg.save_yaml(tmp_path / "train.yaml")
    (tmp_path / "stats").mkdir()
    monkeypatch.setattr(
        sys, "argv", ["prepare_stats", str(tmp_path / "train.yaml"), str(tmp_path)]
    )
    prepare_stats.main()
    canonical, _, _, _ = prepare_stats.open_canonical(cfg)
    reference = persistence_rms(canonical, list(layout.prognostic_var_names), 1)
    result = xr.open_zarr(tmp_path / "stats/persistence_rms.zarr")
    for name in reference:
        np.testing.assert_allclose(result[name], reference[name], rtol=1e-12)
    opened = imprint_metrics.open_data(tmp_path / "OM4.zarr")
    assert "wetmask" in opened and "mask_0" not in opened
