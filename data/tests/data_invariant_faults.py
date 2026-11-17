# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plausible processing bugs that the ocean-physics properties must detect.

Each fault leaves the data looking reasonable field by field; only a physical
relation (a rotation, a window, a mask, a column depth) exposes it.
`python -m samudra_verification.gate` finds this file by its `_faults.py` suffix.
The regridding properties need xESMF, which the gate's environment leaves out,
so no fault here relies on them.
"""

from pathlib import Path

from samudra_verification.faults import Catalog, Fault

DATA = Path(__file__).resolve().parents[1]
PREPROCESSING = "ocean_preprocessing.preprocessing"
OM4 = "ocean_preprocessing.simulation_preprocessing.gfdl_om4"


def detectors(module: str, *tests: str) -> tuple[str, ...]:
    return tuple(f"tests/{module}.py::{test}" for test in tests)


FAULTS = (
    Fault(
        "rotation_matrix_not_orthogonal",
        PREPROCESSING,
        'xr.concat([np.cos(theta), np.sin(theta)], dim="dim_out")',
        'xr.concat([np.cos(theta), -np.sin(theta)], dim="dim_out")',
        detectors("test_rotation_properties", "test_rotation_preserves_speed"),
    ),
    Fault(
        # Speed and composition survive a clockwise rotation; only orientation catches it.
        "rotation_clockwise",
        PREPROCESSING,
        "theta = np.deg2rad(angle)",
        "theta = -np.deg2rad(angle)",
        detectors(
            "test_rotation_properties",
            "test_quarter_turn_counterclockwise_maps_x_to_north",
        ),
    ),
    Fault(
        "partial_depth_unclipped",
        PREPROCESSING,
        "depth_diff = depth_diff.where(depth_diff > 0, 0)",
        "depth_diff = depth_diff.where(depth_diff > 0, depth_diff)",
        detectors(
            "test_vertical_properties",
            "test_partial_depths_clip_columns_to_the_sea_floor",
        ),
    ),
    Fault(
        "flattened_mask_from_surface",
        PREPROCESSING,
        'ds[f"mask_{i}"] = ds["wetmask"].isel({"lev": i})',
        'ds[f"mask_{i}"] = ds["wetmask"].isel({"lev": 0})',
        detectors(
            "test_vertical_properties",
            "test_flattening_splits_levels_without_changing_them",
        ),
    ),
    Fault(
        "surface_filter_uses_bottom_mask",
        PREPROCESSING,
        "depth_indexer = {depth_dim: 0}",
        "depth_indexer = {depth_dim: -1}",
        detectors(
            "test_filter_properties", "test_surface_fields_filter_like_the_top_level"
        ),
    ),
    Fault(
        # The donor's stamps would then label each window's start: wfo one step early.
        "wfo_aligned_to_window_start",
        OM4,
        "interval_ends = bounds.isel(nv=1, drop=True)",
        "interval_ends = bounds.isel(nv=0, drop=True)",
        detectors(
            "test_time_properties",
            "test_transplanted_wfo_keeps_each_window_value",
            "test_transplant_refuses_wfo_one_window_off",
        ),
    ),
    Fault(
        "supergrid_centres_on_corners",
        OM4,
        "h_indicies = dict(nyp=slice(1, None, 2), nxp=slice(1, None, 2))",
        "h_indicies = dict(nyp=slice(0, -1, 2), nxp=slice(1, None, 2))",
        detectors(
            "test_native_grid_properties",
            "test_supergrid_splits_into_centres_and_corners",
        ),
    ),
    Fault(
        "observation_mean_trails_the_stamp",
        "ocean_preprocessing.obs_preprocessing.prepare",
        "rolled = ds.rolling(time=5, center=True, min_periods=5).mean()",
        "rolled = ds.rolling(time=5, center=False, min_periods=5).mean()",
        detectors(
            "test_time_properties",
            "test_observation_means_cover_exactly_the_om4_window",
        ),
    ),
    Fault(
        "face_velocity_left_uncentred",
        "ocean_preprocessing.simulation_preprocessing.interpolate",
        "if set([xh, yh]).issubset(da.dims):",
        "if xh in da.dims:",
        detectors(
            "test_native_grid_properties",
            "test_centring_averages_the_faces_around_each_cell",
        ),
    ),
)

CATALOG = Catalog(project=DATA, faults=FAULTS)
