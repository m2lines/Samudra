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
    for name in (
        "_with_cell_areas",
        "_map_coords",
        "_map_plot_kwargs",
        "_reject_on_curvilinear",
    ):
        setattr(stub, name, types.MethodType(getattr(Viz, name), stub))
    return stub


# The builders above stay functions because they take arguments and call each
# other. These fixtures cover the defaults, which is what almost every test
# below wants.


@pytest.fixture
def tripolar_coords():
    """The nonseparable axes and 2-D centers the other fixtures are built on."""
    return _tripolar_coords()


@pytest.fixture
def source():
    """A source dataset as it arrives from eval, on y/x with 2-D lat/lon."""
    return _source()


@pytest.fixture
def preserved(source):
    """`source` after preprocessing: y/x renamed, true geography kept as *_2d."""
    return preserve_2d_coords(source)


@pytest.fixture
def curvilinear_data(preserved):
    """`preserved` back on the y/x axis names the plotting helpers read."""
    return preserved.rename({"lat": "y", "lon": "x"})


@pytest.fixture
def curvilinear_field(curvilinear_data):
    """A single variable off `curvilinear_data`, for the `.plot()` helpers."""
    return curvilinear_data["tos"]


@pytest.fixture
def gaussian_viz():
    return _viz("gaussian")


@pytest.fixture
def tripolar_viz():
    return _viz("tripolar")


@pytest.fixture
def llc_viz():
    return _viz("llc")


# --- 2-D coordinates survive preprocessing ------------------------------------


def test_preserve_2d_coords_keeps_nonseparable_geography(source, tripolar_coords):
    """The real cell centers survive the y/x -> lat/lon rename, exactly."""
    _, _, lat2d, lon2d = tripolar_coords

    out = preserve_2d_coords(source)

    assert out["lat_2d"].dims == ("lat", "lon")
    assert out["lon_2d"].dims == ("lat", "lon")
    np.testing.assert_array_equal(out["lat_2d"].values, lat2d)
    np.testing.assert_array_equal(out["lon_2d"].values, lon2d)


def test_preserve_2d_coords_geography_is_not_recoverable_by_broadcasting(
    tripolar_coords,
):
    """Guards the fixture: broadcasting the axes must not reproduce the truth.

    Without this the test above would still pass on a rectilinear grid and
    would not be testing anything.
    """
    y, x, lat2d, lon2d = tripolar_coords

    assert not np.allclose(np.broadcast_to(y[:, None], lat2d.shape), lat2d)
    assert not np.allclose(np.broadcast_to(x[None, :], lon2d.shape), lon2d)


def test_preserve_2d_coords_is_idempotent(source, tripolar_coords):
    """A source that has already been through this must not crash.

    Renaming `lat` onto an occupied `lat_2d` is an xarray error, and rollouts
    can arrive already carrying the preserved names.
    """
    once = preserve_2d_coords(source)
    # Put it back on y/x, keeping the preserved names, as a re-fed rollout would.
    again = once.rename({"lat": "y", "lon": "x"})

    out = preserve_2d_coords(again)

    _, _, lat2d, lon2d = tripolar_coords
    np.testing.assert_array_equal(out["lat_2d"].values, lat2d)
    np.testing.assert_array_equal(out["lon_2d"].values, lon2d)
    assert out["tos"].dims == ("lat", "lon")


def test_preserve_2d_coords_renames_axes_to_lat_lon(source):
    """The index axes still land on the names the rest of viz works in."""
    out = preserve_2d_coords(source)

    assert out["tos"].dims == ("lat", "lon")
    assert "y" not in out.dims and "x" not in out.dims


# --- cell areas ----------------------------------------------------------------


def test_tripolar_uses_source_areacello_not_cosine_latitude(tripolar_viz):
    """Weighted means must follow the real areas, not cos(lat).

    On a tripolar grid the two disagree: cell area collapses towards the
    Arctic fold while cos(lat) does not know the grid has folded.
    """
    area = np.linspace(1.0, 50.0, NY * NX).reshape(NY, NX)
    data = preserve_2d_coords(_source(areacello=area))

    out = Viz._with_cell_areas(tripolar_viz, data)

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


def test_tripolar_without_areacello_fails_loudly(tripolar_viz, preserved):
    """A wrong figure is worse than an error, so refuse to invent areas."""
    with pytest.raises(ValueError, match="carries no 'areacello'"):
        Viz._with_cell_areas(tripolar_viz, preserved)


def test_tripolar_rejects_areacello_on_the_wrong_grid(tripolar_viz, preserved):
    """Areas given on a different grid than the data are refused."""
    # A distinct dim name, so xarray keeps the mismatch instead of aligning it.
    preserved["areacello"] = (("lat", "lon_other"), np.ones((NY, NX - 1)))

    with pytest.raises(ValueError, match="expected"):
        Viz._with_cell_areas(tripolar_viz, preserved)


def test_tripolar_rejects_unusable_areacello(tripolar_viz, preserved):
    """Areas that are all NaN cannot weight anything."""
    preserved["areacello"] = (("lat", "lon"), np.full((NY, NX), np.nan))

    with pytest.raises(ValueError, match="no positive finite values"):
        Viz._with_cell_areas(tripolar_viz, preserved)


def test_gaussian_area_path_is_unchanged(gaussian_viz):
    """The rectilinear path still derives areas from the axes."""
    ny, nx = 5, 8
    data = xr.Dataset(
        {"tos": (("lat", "lon"), np.ones((ny, nx)))},
        coords={
            "lat": np.linspace(-80.0, 80.0, ny),
            "lon": np.linspace(0.0, 350.0, nx),
        },
    )

    out = Viz._with_cell_areas(gaussian_viz, data)

    assert "areacello" in out and "areacello_spherical" in out
    # Cosine weights, normalized, as before.
    expected = np.cos(np.deg2rad(data["lat"].values))
    expected = np.repeat(expected[:, None], nx, axis=1)
    expected = expected / expected.sum()
    np.testing.assert_allclose(out["areacello"].values, expected, rtol=1e-6)


# --- maps ----------------------------------------------------------------------


def test_map_coords_are_2d_geographic_on_tripolar(
    tripolar_viz, curvilinear_data, tripolar_coords
):
    """pcolormesh must receive degrees, not cell indices."""
    map_x, map_y = Viz._map_coords(tripolar_viz, curvilinear_data)

    assert map_x.name == "lon_2d" and map_y.name == "lat_2d"
    assert map_x.dims == ("y", "x") and map_y.dims == ("y", "x")
    _, _, lat2d, lon2d = tripolar_coords
    np.testing.assert_array_equal(map_y.values, lat2d)
    np.testing.assert_array_equal(map_x.values, lon2d)


def test_llc_takes_the_curvilinear_path_too(llc_viz, curvilinear_data):
    """Tripolar is not the only curvilinear grid `GridType` names.

    Every branch asks `is_curvilinear`, so lat-lon-cap has to get the 2-D
    coordinates rather than the rectilinear default it would fall into if the
    branches tested for a specific grid by name.
    """
    map_x, map_y = Viz._map_coords(llc_viz, curvilinear_data)

    assert map_x.name == "lon_2d" and map_y.name == "lat_2d"
    with pytest.raises(NotImplementedError, match="llc"):
        Viz._reject_on_curvilinear(llc_viz, "movies", "reason.")


def test_map_plot_kwargs_name_the_2d_coords_on_tripolar(
    tripolar_viz, curvilinear_field
):
    """`.plot()` would otherwise use the index dims as plotting axes."""
    import cartopy.crs as ccrs  # type: ignore[import-untyped]

    kwargs = Viz._map_plot_kwargs(tripolar_viz, curvilinear_field)

    assert kwargs["x"] == "lon_2d" and kwargs["y"] == "lat_2d"
    assert isinstance(kwargs["transform"], ccrs.PlateCarree)


def test_map_plot_kwargs_are_empty_for_gaussian(gaussian_viz, curvilinear_field):
    """The rectilinear path keeps letting xarray choose."""
    assert Viz._map_plot_kwargs(gaussian_viz, curvilinear_field) == {}


def test_map_coords_stay_on_index_axes_for_gaussian(gaussian_viz, curvilinear_data):
    """The fast rectilinear path is untouched."""
    map_x, map_y = Viz._map_coords(gaussian_viz, curvilinear_data)

    assert map_x.name == "x" and map_y.name == "y"


def test_map_coords_fail_loudly_when_geography_was_lost(tripolar_viz):
    data = xr.Dataset(
        {"tos": (("y", "x"), np.ones((NY, NX)))},
        coords={"y": np.arange(NY), "x": np.arange(NX)},
    )

    with pytest.raises(ValueError, match="lat_2d"):
        Viz._map_coords(tripolar_viz, data)


def test_map_helpers_pass_2d_coords_and_plate_carree(tripolar_viz, curvilinear_field):
    """The surface-map helper hands pcolormesh 2-D coords and a transform."""
    import cartopy.crs as ccrs  # type: ignore[import-untyped]

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

    Viz.plot_surface_map(tripolar_viz, _Axis(), curvilinear_field, "title", 0)

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


@pytest.fixture
def matching_mask(preserved):
    """A mask carrying this grid's own cell centers, so alignment succeeds."""
    return _mask(NY, NX, with_coords=preserved)


def test_mismatched_basin_dimensions_fail_loudly(curvilinear_data):
    """The published Gaussian mask cannot be stretched onto a native grid."""
    with pytest.raises(ValueError, match="same grid as the data"):
        process_mask(curvilinear_data, _mask(NY + 1, NX), "tripolar")


def test_basin_mask_without_2d_coords_fails_loudly(curvilinear_data):
    """Matching shapes do not imply matching geography off a rectilinear grid."""
    with pytest.raises(ValueError, match="no 2-D"):
        process_mask(curvilinear_data, _mask(NY, NX), "tripolar")


def test_basin_mask_on_a_different_grid_fails_loudly(curvilinear_data):
    other = preserve_2d_coords(_source())
    other["lat_2d"] = other["lat_2d"] + 5.0
    mask = _mask(NY, NX, with_coords=other)

    with pytest.raises(ValueError, match="does not match"):
        process_mask(curvilinear_data, mask, "tripolar")


def test_coordinate_matched_basin_mask_aligns(curvilinear_data, matching_mask):
    """A mask carrying this grid's own cell centers is accepted."""
    out = process_mask(curvilinear_data, matching_mask, "tripolar")

    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out["y"].values, curvilinear_data["y"].values)
    np.testing.assert_array_equal(out["x"].values, curvilinear_data["x"].values)


def test_coordinate_matched_mask_gives_the_expected_basin_weighted_result(
    curvilinear_data, matching_mask
):
    """A basin mean must reduce over exactly the cells the mask selects."""
    # Select a single column, so the expected answer is arithmetic.
    mask = matching_mask.where(matching_mask["lon"] == 2, 0)

    aligned = process_mask(curvilinear_data, mask, "tripolar")

    field = xr.DataArray(
        np.arange(NY * NX, dtype=float).reshape(NY, NX), dims=("y", "x")
    )
    got = float((field * aligned).mean().values)
    expected = float(field.isel(x=2).mean().values)

    assert np.isclose(got, expected)


def test_gaussian_mask_path_is_unchanged(curvilinear_data):
    """Positional relabeling still happens on a rectilinear grid."""
    out = process_mask(curvilinear_data, _mask(NY, NX), "gaussian")

    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out["y"].values, curvilinear_data["y"].values)


_BASIN_NAMES = (
    "basin_atlantic",
    "basin_pacific",
    "basin_indian",
    "basin_southern",
    "basin_arctic",
)


def _basin_set(preserved: xr.Dataset, assignment: np.ndarray) -> xr.Dataset:
    """A partitioning mask set on this grid, from one code per cell.

    Codes run 1-5 over `_BASIN_NAMES` in order; 0 leaves a cell unassigned, as
    the marginal seas and the land are.
    """
    return xr.Dataset(
        {
            name: (("lat", "lon"), (assignment == code).astype(float))
            for code, name in enumerate(_BASIN_NAMES, start=1)
        },
        coords={
            # Nominal 1-D axes, as both the published masks and OM4's native
            # ones carry. The true centers are the 2-D pair below.
            "lat": preserved["lat"].values,
            "lon": preserved["lon"].values,
            "lat_2d": (("lat", "lon"), preserved["lat_2d"].values),
            "lon_2d": (("lat", "lon"), preserved["lon_2d"].values),
        },
    )


def _aligned_basins(grid_type: GridType, data: xr.Dataset, basins: xr.Dataset):
    """`Viz.basin_masks` off a stub, bypassing the cached_property descriptor."""
    stub = _viz(grid_type)
    stub.data = data
    stub._basins = basins
    return Viz.basin_masks.func(stub)


def test_basin_masks_keep_every_cell_the_mask_set_assigns(preserved, curvilinear_data):
    """A cell in exactly one basin going in must be in exactly one coming out.

    This used to cut the Atlantic, Pacific and Indian at 32S. That was written
    for the published Gaussian masks, whose Southern Ocean runs north to 32.5S,
    so it removed nothing there. Masks built from OM4's own region codes put
    the boundary further south, and the cells between the two lines were
    deleted from the three basins without the Southern Ocean reaching up to
    claim them, dropping them out of the basin diagnostics entirely.
    """
    lat2d = preserved["lat_2d"].values
    # Southern Ocean stopping well south of 32S, Atlantic running down to meet
    # it, which is how OM4's region codes divide this boundary.
    assignment = np.where(lat2d < -45.0, 4, 1)
    basins = _basin_set(preserved, assignment)

    masks = _aligned_basins("tripolar", curvilinear_data, basins)

    # `process_mask` writes unassigned cells as NaN rather than zero.
    claimed = sum(np.nan_to_num(masks[name].values) for name in masks.data_vars)
    np.testing.assert_array_equal(claimed, np.ones_like(lat2d))


def test_the_basin_fixture_straddles_the_old_cut(preserved):
    """Guards the test above: it only bites if cells sit between the lines.

    Without cells that are south of 32S and north of the Southern Ocean, the
    old cut would have been a no-op here too and the test would pass either
    way.
    """
    lat2d = preserved["lat_2d"].values
    nominal = np.broadcast_to(preserved["lat"].values[:, None], lat2d.shape)

    at_risk = (nominal < -32.0) & (lat2d >= -45.0)

    assert at_risk.any()


# --- refusing steps that only hold on a rectilinear grid ------------------------


def test_rectilinear_only_step_is_rejected_on_tripolar(tripolar_viz):
    with pytest.raises(NotImplementedError, match="thetao_mae_metrics"):
        Viz._reject_on_curvilinear(tripolar_viz, "thetao_mae_metrics", "reason.")


@pytest.mark.parametrize(
    "step",
    [
        "thetao_mae_metrics",
        "enso_plots",
        "movies",
        "ocean_temperature_profile_plots",
    ],
)
def test_every_rectilinear_only_step_is_guarded(step, tripolar_viz):
    """These steps average over 'x' or select by index range.

    Neither is meaningful once rows of the grid stop following lines of
    constant latitude, so each must refuse rather than draw a wrong figure.
    """
    with pytest.raises(NotImplementedError, match=step):
        getattr(Viz, f"step_{step}")(tripolar_viz)


def test_rectilinear_only_step_runs_on_gaussian(gaussian_viz):
    """No exception is the whole assertion here."""
    Viz._reject_on_curvilinear(gaussian_viz, "thetao_mae_metrics", "reason.")
