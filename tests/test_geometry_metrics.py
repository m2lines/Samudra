# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from samudra.config import TrainConfig


def test_latitude_metrics_physical_error_and_finite_counts(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    exp = repo / "experiments/geometry-film-1deg"
    cfg = TrainConfig.from_yaml(exp / "control-train.yaml")
    layout = cfg.data.sources[0].build_layout()
    lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(4)[0]))
    lon = np.arange(8) * 45 + 22.5
    times = xr.date_range(
        "2014-10-20", periods=598, freq="5D", calendar="noleap", use_cftime=True
    )
    shape = (598, 4, 8)
    data = xr.Dataset(
        {
            name: (("time", "y", "x"), np.zeros(shape, dtype="float32"))
            for name in layout.prognostic_var_names
        },
        coords={"time": times, "y": lat, "x": lon},
    )
    for i in range(19):
        data[f"mask_{i}"] = (("y", "x"), np.ones((4, 8), dtype=bool))
    for name in layout.boundary_var_names:
        data[name] = (("time", "y", "x"), np.zeros(shape, dtype="float32"))
    data.to_zarr(tmp_path / "OM4.zarr")
    stats = xr.Dataset(
        {
            name: xr.DataArray(1.0)
            for name in layout.prognostic_var_names + layout.boundary_var_names
        }
    )
    stats.to_zarr(tmp_path / "OM4_means.zarr")
    stats.to_zarr(tmp_path / "OM4_stds.zarr")
    pred = xr.Dataset(
        {
            name: (("time", "lev", "y", "x"), np.ones((598, 19, 4, 8), dtype="float32"))
            for name in ["uo", "vo", "thetao", "so"]
        },
        coords={
            "time": times,
            "lev": list(layout.depth_levels),
            "y": lat,
            "x": lon,
            "lat": ("y", lat),
        },
    )
    pred["zos"] = (("time", "y", "x"), np.ones(shape, dtype="float32"))
    for label in ["epoch_0070", "ema_latest"]:
        destination = tmp_path / "output" / cfg.experiment.name / "evals" / label
        destination.mkdir(parents=True)
        pred.to_zarr(destination / "predictions.zarr")
    cfgdict = cfg.model_dump(mode="json")
    cfgdict["experiment"]["data_root"] = str(tmp_path)
    cfgdict["experiment"]["base_output_dir"] = str(tmp_path / "output")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfgdict))
    monkeypatch.setattr("sys.argv", ["latitude_metrics.py", str(config)])
    runpy.run_path(str(exp / "latitude_metrics.py"), run_name="__main__")
    for table in (tmp_path / "output").rglob("latitude-errors.csv"):
        metrics = pd.read_csv(table)
        np.testing.assert_allclose(metrics.rmse, 1)
        np.testing.assert_allclose(metrics.bias, 1)
        assert metrics.nonfinite_wet_cells.sum() == 0
        assert metrics.lead_days.min() == 5
