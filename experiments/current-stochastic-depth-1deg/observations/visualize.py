# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from samudra.config import CpuDataLoadingConfig, EvalConfig, ObsMetricsConfig
from samudra.metrics import kernels, observations
from samudra.metrics.run import analysis_ready
from samudra.utils.data import stack_levels
from samudra.utils.location import LocalLocation
from samudra.viz import observations as figures

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
dask.config.set(scheduler="threads", num_workers=8)
root = Path(
    "/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006"  # pragma: allowlist secret
)
out = root / "observations" / "viz"
out.mkdir(parents=True, exist_ok=True)
cfg = EvalConfig.from_yaml(root / "control/eval.yaml")
cfg.data.loading = CpuDataLoadingConfig()
cfg.experiment.data_root = LocalLocation(
    path=Path("/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg")
)
bundle = cfg.data.build(cfg.experiment.resolved_data_root)
assert bundle.inference_source is not None
obs = ObsMetricsConfig.model_validate(
    EvalConfig._load_yaml(root / "source/src/samudra/configs/data/obs.yaml")
)
rollouts = {}
paths = {}
for variant, job in [("control", 210664), ("sd01", 210665)]:
    for checkpoint in ["epoch_0070", "ema_latest"]:
        label = f"{variant}_{checkpoint}"
        path = (
            root
            / variant
            / "output"
            / f"current-1deg-{variant}-ga2-{job}"
            / "evals"
            / checkpoint
        )
        paths[label] = path
        prediction = xr.open_zarr(path / "predictions.zarr", chunks={})
        wet = stack_levels(
            xr.Dataset(
                {
                    name: (
                        ("y", "x"),
                        bundle.inference_source.masks.prognostic[i]
                        .numpy()
                        .astype(bool),
                    )
                    for i, name in enumerate(bundle.data_layout.prognostic_var_names)
                },
                coords={"y": prediction.y, "x": prediction.x},
            ),
            bundle.data_layout,
        )
        for field in wet.data_vars:
            prediction[field] = prediction[field].where(wet[field])
        rollouts[label] = prediction

start = time.perf_counter()
reference_times = next(iter(rollouts.values())).time
for prediction in rollouts.values():
    np.testing.assert_array_equal(prediction.time.values, reference_times.values)
rollouts["om4"] = bundle.inference_source.to_xarray_dataset().sel(time=reference_times)
prepared = {
    label: observations.model_on_latlon_grid(
        analysis_ready(ds, bundle.data_layout), bundle.data_layout
    )
    for label, ds in rollouts.items()
}
products = observations.open_products(obs, cfg.experiment.resolved_data_root)
frame = pd.read_csv(root / "observations/observation_metrics.csv")
assert set(frame.model) == set(prepared)
frame.to_csv(out / "observation_metrics.csv", index=False)
# Six series including the observed reference; stock palette has only five.
figures.SERIES_COLOURS = (  # type: ignore[assignment]
    "#000000",
    "#0072b2",
    "#56b4e9",
    "#d55e00",
    "#cc79a7",
    "#009e73",
)
original_save = figures.save
files = []


def save(figure, directory, name):
    figure.savefig(Path(directory) / f"{name}.png", bbox_inches="tight", dpi=140)
    path = original_save(figure, directory, name)
    files.append(path)
    return path


figures.save = save
spectral_rows, curve_rows, band_rows, series_rows, stats_rows = [], [], [], [], []
original_panel = figures.spectra_panel


def spectral_panel(curves, title, xlabel, ylabel, scores=None):
    for model, regions in (scores or {}).items():
        for region, value in regions.items():
            spectral_rows.append(
                dict(
                    diagnostic=title,
                    model=model,
                    region=region,
                    log10_rmse_dex=value,
                    available=bool(np.isfinite(value)),
                )
            )
    for model, regions in curves.items():
        for region, (x, y) in regions.items():
            for a, b in zip(x, y, strict=True):
                curve_rows.append(
                    dict(
                        diagnostic=title,
                        model=model,
                        region=region,
                        x=a,
                        power=b,
                        x_units=xlabel,
                    )
                )
    pd.DataFrame(spectral_rows).to_csv(out / "spectral_scores.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "spectral_curves.csv", index=False)
    return original_panel(curves, title, xlabel, ylabel, scores)


figures.spectra_panel = spectral_panel
original_bands = figures.interannual_spectra_panel


def bands_panel(bands, title, xlabel, ylabel):
    for model, regions in bands.items():
        for region, (x, mean, lower, upper, years) in regions.items():
            for a, b, c, d in zip(x, mean, lower, upper, strict=True):
                band_rows.append(
                    dict(
                        diagnostic=title,
                        model=model,
                        region=region,
                        x=a,
                        mean=b,
                        lower=c,
                        upper=d,
                        years=years,
                    )
                )
    pd.DataFrame(band_rows).to_csv(out / "spectral_interannual.csv", index=False)
    return original_bands(bands, title, xlabel, ylabel)


figures.interannual_spectra_panel = bands_panel
original_series = figures.series_panel


def series_panel(series, title, ylabel, annotations=None):
    for model, values in series.items():
        for date, value in values.items():
            series_rows.append(
                dict(
                    diagnostic=title,
                    model=model,
                    date=str(date),
                    value=value,
                    units=ylabel,
                )
            )
        if annotations is not None:
            stats_rows.append(
                dict(
                    diagnostic=title,
                    model=model,
                    trend_per_year=kernels.series_linear_trend_per_year(values),
                    residual_variance=kernels.series_residual_variance(values),
                    units=ylabel,
                )
            )
    pd.DataFrame(series_rows).to_csv(out / "timeseries.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out / "timeseries_statistics.csv", index=False)
    return original_series(series, title, ylabel, annotations)


figures.series_panel = series_panel
# Run spectral comparisons first; then the rest of the observation figures.
figures.spectra_figures(prepared, products, str(out), obs.velocity_kind)
figures.timeseries_figures(prepared, products, str(out), obs.velocity_kind)
figures.rmse_map_figures(
    prepared, products, frame, obs.window, str(out), obs.velocity_kind
)
figures.variance_map_figures(prepared, products, frame, str(out))
save(
    figures.annual_rmse_panel(frame, "SD comparison: annual RMSE vs observations"),
    str(out),
    "annual_total_rmse_vs_observations",
)
assert len(files) == 20, files
(out / "COMPLETE.json").write_text(
    json.dumps(
        dict(
            figures=files,
            elapsed_seconds=time.perf_counter() - start,
            models=list(prepared),
            masking="Original per-channel wet masks; stored predictions unchanged",
            headline_metrics="Reused verified masked scoring CSV",
            spectral_scores=len(spectral_rows),
        ),
        indent=2,
    )
    + "\n"
)
print("COMPLETE", len(files), "figures", flush=True)
