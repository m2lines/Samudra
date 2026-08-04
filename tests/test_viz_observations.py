# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation figures in `samudra.viz`.

The point of routing both jobs through `samudra.metrics` is that a number
printed on a figure is the number logged to W&B. These check that, and that
each figure builder produces files on data shaped like the real thing.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from samudra.constants import build_om4_spec  # noqa: E402
from samudra.metrics import observations, report  # noqa: E402
from samudra.viz import observations as figures  # noqa: E402

OM4_SPEC = build_om4_spec(prognostic_vars_key="thermo_dynamic_all")
WINDOW = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"))


@pytest.fixture(scope="module")
def synthetic():
    """A rollout and three observation products on realistic, differing grids."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2021-01-01", "2022-12-31", freq="5D")
    months = pd.date_range("2021-01-01", "2022-12-01", freq="MS")

    model_lat = np.linspace(-78.0, 78.0, 40)
    model_lon = np.arange(2.5, 360.0, 5.0)
    obs_lat = np.linspace(-79.0, 79.0, 60)
    obs_lon = np.arange(1.5, 360.0, 3.0)
    levels = np.array(OM4_SPEC.depth_levels)

    rollout = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                10
                + rng.normal(
                    size=(len(time), levels.size, model_lat.size, model_lon.size)
                ),
            ),
            "zos": (
                ("time", "y", "x"),
                rng.normal(scale=0.1, size=(len(time), model_lat.size, model_lon.size)),
            ),
        },
        coords={
            "y": model_lat,
            "x": model_lon,
            "lat": (
                ("y", "x"),
                np.broadcast_to(
                    model_lat[:, None], (model_lat.size, model_lon.size)
                ).copy(),
            ),
            "lon": (
                ("y", "x"),
                np.broadcast_to(
                    model_lon[None, :], (model_lat.size, model_lon.size)
                ).copy(),
            ),
            "lev": levels,
            "areacello": (("y", "x"), np.full((model_lat.size, model_lon.size), 1e10)),
            "time": time,
        },
    )

    def product(data_vars, stamps, extra=None):
        coords = {"lat": obs_lat, "lon": obs_lon, "time": stamps}
        coords.update(extra or {})
        return observations.with_cell_area(
            observations.standardize(xr.Dataset(data_vars, coords=coords))
        )

    shape = (len(time), obs_lat.size, obs_lon.size)
    duacs = product(
        {
            "ugos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
            "vgos": (("time", "lat", "lon"), rng.normal(scale=0.2, size=shape)),
        },
        time,
    )
    oisst = product(
        {"sst": (("time", "lat", "lon"), 10 + rng.normal(size=shape))}, time
    )
    depths = np.array([5.0, 50.0, 150.0, 400.0, 800.0, 1400.0, 1900.0])
    argo = product(
        {
            "temp": (
                ("time", "depth", "lat", "lon"),
                10
                + rng.normal(
                    size=(len(months), depths.size, obs_lat.size, obs_lon.size)
                ),
            )
        },
        months,
        {"depth": depths},
    )

    prepared = observations.model_on_latlon_grid(rollout, OM4_SPEC)
    rollouts = {"model": prepared}
    products = {"duacs": duacs, "oisst": oisst, "argo": argo}
    frame = report.compute_observation_metrics(
        rollouts,
        duacs=duacs,
        oisst=oisst,
        argo=argo,
        model_dz={"model": observations.model_depth_thickness(prepared, OM4_SPEC)},
        window=WINDOW,
        bootstrap_samples=0,
    )
    return rollouts, products, frame


def test_a_figure_reports_the_number_that_was_logged(synthetic, tmp_path):
    """A value annotated on a map must equal the scalar sent to W&B.

    This is the whole reason the figures reduce through `samudra.metrics`
    rather than carrying their own implementation: a second one would drift.
    """
    rollouts, products, frame = synthetic
    scalars = report.to_wandb(frame, "model")

    written = figures.rmse_map_figures(rollouts, products, frame, WINDOW, str(tmp_path))
    assert written, "no RMSE maps were produced"

    # The map the figure draws and the scalar W&B receives come from one
    # reduction, so re-deriving the total from the frame must match exactly.
    row = frame[
        (frame["metric"] == "surface_sst_total_rmse")
        & (frame["period_kind"] == "primary_complete_years")
    ].iloc[0]
    logged = scalars["obs/sst/total_rmse"]
    assert isinstance(logged, float), "W&B scalars must be plain floats"
    assert float(row["value"]) == pytest.approx(logged)

    # And the annotation renders that same number rather than recomputing it.
    assert f"{float(row['value']):.4g}" in figures._annotate(
        float(row["value"]), "degC", row
    )


def test_every_figure_builder_writes_files(synthetic, tmp_path):
    """Each builder produces PDFs on data shaped like the real products.

    A smoke test rather than an appearance check: it catches the failures that
    actually happen here -- a grid mismatch, a missing variable, an empty
    region -- which only surface once a figure is drawn end to end.
    """
    rollouts, products, frame = synthetic

    produced = []
    produced += figures.rmse_map_figures(
        rollouts, products, frame, WINDOW, str(tmp_path)
    )
    produced += figures.variance_map_figures(rollouts, products, frame, str(tmp_path))
    produced += figures.timeseries_figures(rollouts, products, str(tmp_path))
    produced.append(
        figures.save(
            figures.annual_rmse_panel(frame, "annual"), str(tmp_path), "annual"
        )
    )

    assert len(produced) >= 8
    for path in produced:
        assert path.endswith(".pdf")
        assert (tmp_path / path.rsplit("/", 1)[-1]).stat().st_size > 0


def test_spectra_figures_survive_regions_too_small_to_transform(synthetic, tmp_path):
    """A region the grid cannot resolve is skipped, not fatal.

    The coarse grids here make several named regions too small for a
    transform, which is exactly what a low-resolution rollout does in
    production.
    """
    rollouts, products, _ = synthetic
    produced = figures.spectra_figures(rollouts, products, str(tmp_path))
    assert produced
    for path in produced:
        assert path.endswith(".pdf")


def test_curve_scores_are_zero_against_the_reference_itself():
    """A run identical to the observations must score 0 dex, not merely 'small'."""
    wavenumber = np.logspace(-3, -1, 20)
    power = wavenumber**-2
    curves = {
        "DUACS": {"Gulf Stream": (wavenumber, power)},
        "model": {"Gulf Stream": (wavenumber, power)},
    }
    scores = figures._curve_scores(curves, "DUACS")
    assert scores["model"]["Gulf Stream"] == pytest.approx(0.0, abs=1e-12)
    assert "DUACS" not in scores, "the reference is not scored against itself"
