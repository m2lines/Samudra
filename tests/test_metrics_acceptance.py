# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test: reproduce the published OM4 observation metrics.

Scores the quarter-degree OM4 data *as the model* and compares against the OM4
column of the metrics document. Because those numbers were computed
independently -- different code, different infrastructure, a separately produced
observation archive -- agreement is strong evidence that this port is faithful
rather than merely self-consistent.

Needs no checkpoint, no GPU and no rollout: it is data in, numbers out. It does
need the quarter-degree OM4 store locally, so it is marked `manual` and skips
unless `SAMUDRA_OM4_QUARTERDEG` points at one:

    SAMUDRA_OM4_QUARTERDEG=/scratch/$USER/data/om4_quarterdeg_v2/OM4.zarr \\
        uv run pytest -m manual tests/test_metrics_acceptance.py

The observation products are read anonymously from the public bucket, so no
credentials are required.

Every value below was verified to 0.11% or better; four real bugs were found by
running it, each invisible to the synthetic unit tests: a cftime calendar,
mismatched product end dates, the IAP `depth_std` coordinate name, and computing
residual variance after regridding rather than before.
"""

import os

import pandas as pd
import pytest
import xarray as xr

from samudra.constants import build_om4_spec
from samudra.metrics import observations, report
from samudra.metrics.run import analysis_ready

OBS_ROOT = "s3://m2lines-pubs/Samudra/v2026-07/obs"
ANON = {"anon": True}

# The rollout span the published diagnostics cover. The window-sensitive
# residual-variance metrics characterise "the whole rollout", so OM4's full
# 1958- record has to be cut to a rollout-like span to be comparable.
ROLLOUT = (pd.Timestamp("2014-10-20"), pd.Timestamp("2022-12-24 23:59:59"))
PRIMARY = (pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"))

# The OM4 column of the metrics document. Tolerance is 1%: the observed spread
# is at most 0.11%, so this catches a regression without being brittle.
PUBLISHED = [
    ("surface_geostrophic_velocity_vector_total_rmse", None, 0.2090),
    ("instantaneous_surface_eke_total_rmse", None, 0.04399),
    ("surface_sst_total_rmse", None, 0.8052),
    ("ohc_per_area_total_rmse", "Surface (0-700m)", 3.010e9),
    ("ohc_per_area_total_rmse", "Intermediate (700-2000m)", 2.187e9),
    ("surface_sst_residual_variance_map_rmse", None, 0.3336),
    ("surface_sst_residual_variance_pattern_corr", None, 0.6861),
    ("ohc_upper700_per_area_residual_variance_map_rmse", None, 1.341e18),
    ("ohc_upper700_per_area_residual_variance_pattern_corr", None, 0.6662),
]


def _obs(name: str) -> xr.Dataset:
    ds = xr.open_zarr(f"{OBS_ROOT}/{name}", chunks={}, storage_options=ANON)
    return observations.with_cell_area(observations.standardize(ds))


@pytest.mark.manual
def test_om4_reproduces_published_observation_metrics():
    om4_path = os.environ.get("SAMUDRA_OM4_QUARTERDEG")
    if not om4_path:
        pytest.skip("set SAMUDRA_OM4_QUARTERDEG to the quarter-degree OM4 store")

    spec = build_om4_spec(prognostic_vars_key="thermo_dynamic_all")
    om4 = observations.model_on_latlon_grid(
        analysis_ready(xr.open_zarr(om4_path, chunks={}), spec), spec
    )
    om4 = om4.sel(time=slice(*ROLLOUT))

    frame = report.compute_observation_metrics(
        {"OM4": om4},
        duacs=_obs("duacs.zarr"),
        oisst=_obs("oisst.zarr"),
        argo=_obs("argo-iap.zarr"),
        model_dz={"OM4": observations.model_depth_thickness(om4, spec)},
        window=PRIMARY,
    )

    scored = frame[frame["period_kind"].isin(["primary_complete_years", "full_overlap"])]
    for metric, depth, expected in PUBLISHED:
        rows = scored[
            (scored["metric"] == metric)
            & (scored["depth"].isna() if depth is None else scored["depth"] == depth)
        ]
        assert not rows.empty, f"{metric} [{depth}] was not computed"
        got = float(rows.iloc[0]["value"])
        assert got == pytest.approx(expected, rel=0.01), (
            f"{metric} [{depth}]: got {got:.6g}, published {expected:.6g}"
        )
