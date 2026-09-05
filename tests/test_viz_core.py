# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Grid-geometry handling in the visualization path.

`samudra viz` was written for rectilinear ("gaussian") grids: it rebuilt cell
areas from the 1-D axes, dropped the source's 2-D lat/lon, aligned basin masks
by position, and plotted against the index axes. All four are wrong on a
curvilinear ("tripolar") grid, where lat/lon vary along both horizontal dims.
These tests pin the curvilinear behaviour while leaving the Gaussian path as it
was.
"""

import types

import numpy as np
import pytest
import xarray as xr

from samudra.constants import GridType, build_om4_layout
from samudra.viz.core import Viz, preserve_2d_coords, process_mask

NY, NX = 4, 6


def _tripolar_coords(ny: int = NY, nx: int = NX):
    """A nonseparable grid: both coordinates vary along both dims.

    This is the property that matters. Broadcasting the 1-D axes cannot
    reproduce it, so anything that reconstructs geometry that way is caught.
    """
    y = np.linspace(-60.0, 60.0, ny)
    x = np.linspace(0.0, 270.0, nx)
    lat2d = y[:, None] + 0.1 * x[None, :]
    lon2d = x[None, :] + 0.1 * y[:, None]
    return y, x, lat2d, lon2d


def _source(ny: int = NY, nx: int = NX, *, areacello=None) -> xr.Dataset:
    """A minimal source dataset in the layout viz receives from eval."""
    y, x, lat2d, lon2d = _tripolar_coords(ny, nx)
    coords = {
        "y": y,
        "x": x,
        "lat": (("y", "x"), lat2d),
        "lon": (("y", "x"), lon2d),
    }
    data_vars = {"tos": (("y", "x"), np.ones((ny, nx)))}
    if areacello is not None:
        data_vars["areacello"] = (("y", "x"), areacello)
    return xr.Dataset(data_vars, coords=coords)


def _viz(grid_type: GridType):
    """A stand-in exposing just what the geometry helpers read off `self`.

    Constructing a real `Viz` needs a full rollout, predictions and an output
    tree; these helpers only touch `data_layout`, so binding them to a stub
    keeps the tests about geometry.
    """
    stub = types.SimpleNamespace(data_layout=build_om4_layout(grid_type=grid_type))
    # Bind the real helpers so methods that call them through `self` work too.
    for name in ("_with_cell_areas", "_map_coords", "_reject_on_curvilinear"):
        setattr(stub, name, types.MethodType(getattr(Viz, name), stub))
    return stub


# --- 2-D coordinates survive preprocessing ------------------------------------


def test_preserve_2d_coords_keeps_nonseparable_geography():
    """The real cell centers survive the y/x -> lat/lon rename, exactly."""
    _, _, lat2d, lon2d = _tripolar_coords()

    out = preserve_2d_coords(_source())

    assert out["lat_2d"].dims == ("lat", "lon")
    assert out["lon_2d"].dims == ("lat", "lon")
    np.testing.assert_array_equal(out["lat_2d"].values, lat2d)
    np.testing.assert_array_equal(out["lon_2d"].values, lon2d)


def test_preserve_2d_coords_geography_is_not_recoverable_by_broadcasting():
    """Guards the fixture: broadcasting the axes must not reproduce the truth.

    Without this the test above would still pass on a rectilinear grid and
    would not be testing anything.
    """
    y, x, lat2d, lon2d = _tripolar_coords()

    assert not np.allclose(np.broadcast_to(y[:, None], lat2d.shape), lat2d)
    assert not np.allclose(np.broadcast_to(x[None, :], lon2d.shape), lon2d)


def test_preserve_2d_coords_is_idempotent():
    """A source that has already been through this must not crash.

    Renaming `lat` onto an occupied `lat_2d` is an xarray error, and rollouts
    can arrive already carrying the preserved names.
    """
    once = preserve_2d_coords(_source())
    # Put it back on y/x, keeping the preserved names, as a re-fed rollout would.
    again = once.rename({"lat": "y", "lon": "x"})

    out = preserve_2d_coords(again)

    _, _, lat2d, lon2d = _tripolar_coords()
    np.testing.assert_array_equal(out["lat_2d"].values, lat2d)
    np.testing.assert_array_equal(out["lon_2d"].values, lon2d)
    assert out["tos"].dims == ("lat", "lon")


def test_preserve_2d_coords_renames_axes_to_lat_lon():
    """The index axes still land on the names the rest of viz works in."""
    out = preserve_2d_coords(_source())

    assert out["tos"].dims == ("lat", "lon")
    assert "y" not in out.dims and "x" not in out.dims


# --- cell areas ----------------------------------------------------------------


def test_tripolar_uses_source_areacello_not_cosine_latitude():
    """Weighted means must follow the real areas, not cos(lat).

    On a tripolar grid the two disagree: cell area collapses towards the
    Arctic fold while cos(lat) does not know the grid has folded.
    """
    area = np.linspace(1.0, 50.0, NY * NX).reshape(NY, NX)
    data = preserve_2d_coords(_source(areacello=area))

    out = Viz._with_cell_areas(_viz("tripolar"), data)

    # `areacello` is the weighting field, normalized; `areacello_spherical` is
    # the physical area in m^2. Both must come from the source.
    np.testing.assert_allclose(out["areacello_spherical"].values, area)
    np.testing.assert_allclose(out["areacello"].values, area / area.sum())

    field = xr.DataArray(
        np.linspace(0.0, 1.0, NY * NX).reshape(NY, NX), dims=("lat", "lon")
    )
    weighted = float(field.weighted(out["areacello"]).mean().values)

    lat2d = data["lat_2d"].values
    cosine = np.cos(np.deg2rad(lat2d))
    cosine_weighted = float((field.values * cosine).sum() / cosine.sum())

    assert not np.isclose(weighted, cosine_weighted, rtol=1e-3), (
        "areacello weighting is indistinguishable from cosine-latitude "
        "weighting, so this test cannot detect the bug it exists for"
    )


def test_tripolar_without_areacello_fails_loudly():
    """A wrong figure is worse than an error, so refuse to invent areas."""
    data = preserve_2d_coords(_source())

    with pytest.raises(ValueError, match="carries no 'areacello'"):
        Viz._with_cell_areas(_viz("tripolar"), data)


def test_tripolar_rejects_areacello_on_the_wrong_grid():
    """Areas given on a different grid than the data are refused."""
    data = preserve_2d_coords(_source())
    # A distinct dim name, so xarray keeps the mismatch instead of aligning it.
    data["areacello"] = (("lat", "lon_other"), np.ones((NY, NX - 1)))

    with pytest.raises(ValueError, match="expected"):
        Viz._with_cell_areas(_viz("tripolar"), data)


def test_tripolar_rejects_unusable_areacello():
    """Areas that are all NaN cannot weight anything."""
    data = preserve_2d_coords(_source())
    data["areacello"] = (("lat", "lon"), np.full((NY, NX), np.nan))

    with pytest.raises(ValueError, match="no positive finite values"):
        Viz._with_cell_areas(_viz("tripolar"), data)


def test_gaussian_area_path_is_unchanged():
    """The rectilinear path still derives areas from the axes."""
    ny, nx = 5, 8
    data = xr.Dataset(
        {"tos": (("lat", "lon"), np.ones((ny, nx)))},
        coords={
            "lat": np.linspace(-80.0, 80.0, ny),
            "lon": np.linspace(0.0, 350.0, nx),
        },
    )

    out = Viz._with_cell_areas(_viz("gaussian"), data)

    assert "areacello" in out and "areacello_spherical" in out
    # Cosine weights, normalized, as before.
    expected = np.cos(np.deg2rad(data["lat"].values))
    expected = np.repeat(expected[:, None], nx, axis=1)
    expected = expected / expected.sum()
    np.testing.assert_allclose(out["areacello"].values, expected, rtol=1e-6)


# --- maps ----------------------------------------------------------------------


def test_map_coords_are_2d_geographic_on_tripolar():
    """pcolormesh must receive degrees, not cell indices."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})

    map_x, map_y = Viz._map_coords(_viz("tripolar"), data)

    assert map_x.name == "lon_2d" and map_y.name == "lat_2d"
    assert map_x.dims == ("y", "x") and map_y.dims == ("y", "x")
    _, _, lat2d, lon2d = _tripolar_coords()
    np.testing.assert_array_equal(map_y.values, lat2d)
    np.testing.assert_array_equal(map_x.values, lon2d)


def test_map_coords_stay_on_index_axes_for_gaussian():
    """The fast rectilinear path is untouched."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})

    map_x, map_y = Viz._map_coords(_viz("gaussian"), data)

    assert map_x.name == "x" and map_y.name == "y"


def test_map_coords_fail_loudly_when_geography_was_lost():
    data = xr.Dataset(
        {"tos": (("y", "x"), np.ones((NY, NX)))},
        coords={"y": np.arange(NY), "x": np.arange(NX)},
    )

    with pytest.raises(ValueError, match="lat_2d"):
        Viz._map_coords(_viz("tripolar"), data)


def test_map_helpers_pass_2d_coords_and_plate_carree(monkeypatch):
    """The surface-map helper hands pcolormesh 2-D coords and a transform."""
    import cartopy.crs as ccrs  # type: ignore[import-untyped]

    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})["tos"]
    captured = {}

    class _Axis:
        def pcolormesh(self, x, y, values, **kwargs):
            captured["x"] = x
            captured["y"] = y
            captured["transform"] = kwargs["transform"]
            return "image"

        def add_feature(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass

        def gridlines(self, *args, **kwargs):
            return types.SimpleNamespace(
                top_labels=True,
                right_labels=True,
                left_labels=True,
                xlabel_style={},
                ylabel_style={},
                xlocator=None,
            )

    viz = _viz("tripolar")
    Viz.plot_surface_map(viz, _Axis(), data, "title", 0)

    assert captured["x"].name == "lon_2d"
    assert captured["y"].name == "lat_2d"
    assert isinstance(captured["transform"], ccrs.PlateCarree)


# --- basin masks ---------------------------------------------------------------


def _mask(ny: int, nx: int, *, with_coords: xr.Dataset | None = None):
    mask = xr.DataArray(
        np.ones((ny, nx)),
        dims=("lat", "lon"),
        coords={"lat": np.arange(ny), "lon": np.arange(nx)},
    )
    if with_coords is not None:
        mask = mask.assign_coords(
            lat_2d=(("lat", "lon"), with_coords["lat_2d"].values),
            lon_2d=(("lat", "lon"), with_coords["lon_2d"].values),
        )
    return mask


def test_mismatched_basin_dimensions_fail_loudly():
    """The published Gaussian mask cannot be stretched onto a native grid."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})

    with pytest.raises(ValueError, match="same grid as the data"):
        process_mask(data, _mask(NY + 1, NX), "tripolar")


def test_basin_mask_without_2d_coords_fails_loudly():
    """Matching shapes do not imply matching geography off a rectilinear grid."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})

    with pytest.raises(ValueError, match="no 2-D"):
        process_mask(data, _mask(NY, NX), "tripolar")


def test_basin_mask_on_a_different_grid_fails_loudly():
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})
    other = preserve_2d_coords(_source())
    other["lat_2d"] = other["lat_2d"] + 5.0
    mask = _mask(NY, NX, with_coords=other)

    with pytest.raises(ValueError, match="does not match"):
        process_mask(data, mask, "tripolar")


def test_coordinate_matched_basin_mask_aligns():
    """A mask carrying this grid's own cell centers is accepted."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})
    mask = _mask(NY, NX, with_coords=preserve_2d_coords(_source()))

    out = process_mask(data, mask, "tripolar")

    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out["y"].values, data["y"].values)
    np.testing.assert_array_equal(out["x"].values, data["x"].values)


def test_coordinate_matched_mask_gives_the_expected_basin_weighted_result():
    """A basin mean must reduce over exactly the cells the mask selects."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})
    mask = _mask(NY, NX, with_coords=preserve_2d_coords(_source()))
    # Select a single column, so the expected answer is arithmetic.
    mask = mask.where(mask["lon"] == 2, 0)

    aligned = process_mask(data, mask, "tripolar")

    field = xr.DataArray(
        np.arange(NY * NX, dtype=float).reshape(NY, NX), dims=("y", "x")
    )
    got = float((field * aligned).mean().values)
    expected = float(field.isel(x=2).mean().values)

    assert np.isclose(got, expected)


def test_gaussian_mask_path_is_unchanged():
    """Positional relabeling still happens on a rectilinear grid."""
    data = preserve_2d_coords(_source()).rename({"lat": "y", "lon": "x"})

    out = process_mask(data, _mask(NY, NX), "gaussian")

    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out["y"].values, data["y"].values)


# --- refusing steps that only hold on a rectilinear grid ------------------------


def test_rectilinear_only_step_is_rejected_on_tripolar():
    with pytest.raises(NotImplementedError, match="thetao_mae_metrics"):
        Viz._reject_on_curvilinear(_viz("tripolar"), "thetao_mae_metrics", "reason.")


@pytest.mark.parametrize(
    "step",
    [
        "thetao_mae_metrics",
        "enso_plots",
        "movies",
        "ocean_temperature_profile_plots",
    ],
)
def test_every_rectilinear_only_step_is_guarded(step):
    """These steps average over 'x' or select by index range.

    Neither is meaningful once rows of the grid stop following lines of
    constant latitude, so each must refuse rather than draw a wrong figure.
    """
    viz = _viz("tripolar")
    with pytest.raises(NotImplementedError, match=step):
        getattr(Viz, f"step_{step}")(viz)


def test_rectilinear_only_step_runs_on_gaussian():
    """No exception is the whole assertion here."""
    Viz._reject_on_curvilinear(_viz("gaussian"), "thetao_mae_metrics", "reason.")
