"""Test tile geometry, the catalog, and the overlap blending operator."""

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.tiling import (
    SIDES,
    TileBlender,
    TileSpec,
    build_group_layout,
    build_tile_catalog,
    ramp_profile,
    resolve_face,
    tile_spec_from_coords,
    validate_tile_group,
)

WINDOWS = ["quintic", "kbd"]

# The live experiment's geometry, shrunk 16x so tests stay fast: four tiles of
# TILE cells tiling a CANONICAL square with OVERLAP cells shared on each seam.
TILE = 23
OVERLAP = 2
STRIDE = TILE - OVERLAP
CANONICAL = TILE + STRIDE


def make_tiles(*, tile: int = TILE, stride: int = STRIDE, face: int = 1) -> list[TileSpec]:
    """A 2x2 group of overlapping tiles, in catalog order."""
    specs = []
    for tile_id, (j0, i0) in enumerate(
        [(0, 0), (0, stride), (stride, 0), (stride, stride)]
    ):
        specs.append(
            TileSpec(
                tile_id=tile_id,
                dataset_index=tile_id,
                face=face,
                i_start=i0,
                i_end=i0 + tile,
                j_start=j0,
                j_end=j0 + tile,
                owned=(0, tile, 0, tile),
            )
        )
    return specs


def make_layout(**kwargs) -> "object":
    return build_group_layout(make_tiles(**kwargs))


def crop_tiles(field: torch.Tensor, layout) -> torch.Tensor:
    """Cut per-tile views out of one canonical field: [C, Hg, Wg] -> [T, C, H, W]."""
    return torch.stack(
        [
            field[:, tile.j_start : tile.j_end, tile.i_start : tile.i_end]
            for tile in layout.tiles
        ]
    )


# --------------------------------------------------------------------------
# Ramp profiles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("width", [1, 2, 3, 8, 16])
def test_ramp_is_monotone_and_spans_the_unit_interval(window, width) -> None:
    ramp = ramp_profile(window, width)
    assert ramp.shape == (width,)
    assert torch.all(ramp > 0) and torch.all(ramp < 1)
    assert torch.all(torch.diff(ramp) > 0)


@pytest.mark.parametrize("width", [1, 2, 4, 16])
def test_quintic_ramp_is_sum_complementary(width) -> None:
    """Opposing quintic weights sum to one: a true partition of unity."""
    ramp = ramp_profile("quintic", width)
    torch.testing.assert_close(ramp + ramp.flip(0), torch.ones(width, dtype=ramp.dtype))


@pytest.mark.parametrize("width", [2, 4, 16])
def test_kbd_ramp_is_power_complementary_not_sum_complementary(width) -> None:
    """KBD satisfies Princen-Bradley, so normalization is load-bearing for it."""
    ramp = ramp_profile("kbd", width)
    torch.testing.assert_close(
        ramp**2 + ramp.flip(0) ** 2, torch.ones(width, dtype=ramp.dtype)
    )
    assert not torch.allclose(ramp + ramp.flip(0), torch.ones(width, dtype=ramp.dtype))


def test_ramp_rejects_unknown_window_and_bad_width() -> None:
    with pytest.raises(ValueError, match="Unknown window kind"):
        ramp_profile("hann", 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width must be"):
        ramp_profile("quintic", 0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_layout_recovers_neighbours_overlaps_and_exterior_sides() -> None:
    layout = make_layout()

    assert layout.canonical_origin == (0, 0)
    assert layout.canonical_shape == (CANONICAL, CANONICAL)

    # Tile 0 sits at the low corner: interior on its high sides only.
    assert layout.overlaps[(0, "ihi")] == OVERLAP
    assert layout.overlaps[(0, "jhi")] == OVERLAP
    assert (0, "ilo") in layout.exterior_sides
    assert (0, "jlo") in layout.exterior_sides

    # Tile 3 is the mirror image at the high corner.
    assert layout.overlaps[(3, "ilo")] == OVERLAP
    assert layout.overlaps[(3, "jlo")] == OVERLAP
    assert (3, "ihi") in layout.exterior_sides
    assert (3, "jhi") in layout.exterior_sides

    # Every tile in a 2x2 has exactly two interior and two exterior sides.
    for tile_id in range(4):
        interior = [s for s in SIDES if (tile_id, s) in layout.overlaps]
        assert len(interior) == 2, tile_id


def test_single_tile_layout_is_entirely_exterior() -> None:
    """The 1x1 task: constant padding on every side, no overlap anywhere."""
    tile = TileSpec(
        tile_id=0, dataset_index=0, face=1,
        i_start=0, i_end=TILE, j_start=0, j_end=TILE, owned=(0, TILE, 0, TILE),
    )
    layout = build_group_layout([tile])
    assert layout.overlaps == {}
    assert layout.exterior_sides == frozenset((0, side) for side in SIDES)


def test_tiles_that_share_no_cells_are_mutually_exterior() -> None:
    far_apart = [
        TileSpec(
            tile_id=k, dataset_index=k, face=1,
            i_start=20 * k, i_end=20 * k + 8, j_start=20 * k, j_end=20 * k + 8,
            owned=(0, 8, 0, 8),
        )
        for k in range(2)
    ]
    layout = build_group_layout(far_apart)
    assert layout.overlaps == {}
    assert len(layout.exterior_sides) == 8


def test_diagonal_tiles_share_the_four_tile_corner_block() -> None:
    """Diagonal tiles are not edge neighbours, but they do share the corner.

    Reporting an overlap on the corresponding sides is correct: the separable
    window turns the two 1-D ramps into exactly the corner product, and on the
    rest of the strip the tile is the only contributor, so normalization
    returns its weight to one.
    """
    tiles = make_tiles()
    layout = build_group_layout([tiles[1], tiles[2]])
    assert layout.overlaps[(1, "ilo")] == OVERLAP
    assert layout.overlaps[(2, "ihi")] == OVERLAP
    assert (1, "ihi") in layout.exterior_sides


def test_layout_rejects_cross_face_groups_and_duplicate_ids() -> None:
    tiles = make_tiles()
    crossed = [tiles[0], TileSpec(**{**vars(tiles[1]), "face": 2})]
    with pytest.raises(ValueError, match="spans faces"):
        build_group_layout(crossed)

    duplicated = [tiles[0], TileSpec(**{**vars(tiles[1]), "tile_id": 0})]
    with pytest.raises(ValueError, match="duplicate tile_ids"):
        build_group_layout(duplicated)


def test_tile_spec_rejects_an_owned_region_outside_its_extent() -> None:
    with pytest.raises(ValueError, match="owned region"):
        TileSpec(
            tile_id=0, dataset_index=0, face=1,
            i_start=0, i_end=8, j_start=0, j_end=8, owned=(0, 8, 0, 9),
        )


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def test_catalog_recovers_extents_from_absolute_index_coordinates() -> None:
    spec = tile_spec_from_coords(
        tile_id=7,
        dataset_index=3,
        face=1,
        x=np.arange(2880, 3248),
        y=np.arange(1072, 1440),
    )
    assert (spec.i_start, spec.i_end) == (2880, 3248)
    assert (spec.j_start, spec.j_end) == (1072, 1440)
    assert spec.shape == (368, 368)
    assert spec.tile_id == 7 and spec.dataset_index == 3


def test_catalog_rejects_non_contiguous_index_coordinates() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        tile_spec_from_coords(
            tile_id=0, dataset_index=0, face=1,
            x=np.array([0, 1, 3]), y=np.arange(3),
        )


def test_face_resolution_prefers_the_attr_then_the_name() -> None:
    assert resolve_face(xr.Dataset(attrs={"llc_face": 5})) == 5
    assert resolve_face(xr.Dataset(), name_hint="LLC4320_face3_i0-8_j0-8.zarr") == 3
    with pytest.raises(ValueError, match="Cannot determine the LLC face"):
        resolve_face(xr.Dataset(), name_hint="no-face-here.zarr")


def test_build_tile_catalog_reads_x_and_y_coords() -> None:
    datasets = [
        xr.Dataset(coords={"x": np.arange(i0, i0 + TILE), "y": np.arange(j0, j0 + TILE)})
        for j0, i0 in [(0, 0), (0, STRIDE), (STRIDE, 0), (STRIDE, STRIDE)]
    ]
    catalog = build_tile_catalog(datasets, faces=[1] * 4)
    layout = build_group_layout(catalog)
    assert layout.canonical_shape == (CANONICAL, CANONICAL)
    assert len(layout.overlaps) == 8  # two interior sides per tile


def test_build_tile_catalog_requires_index_coords() -> None:
    with pytest.raises(ValueError, match="missing coordinate"):
        build_tile_catalog([xr.Dataset()], faces=[1])


# --------------------------------------------------------------------------
# Group validation
# --------------------------------------------------------------------------


def make_group_datasets(*, times: int = 4, corrupt_first_time: bool = False):
    """Four caches cut from one truth field, so overlaps agree by construction."""
    layout = make_layout()
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((times, 2, CANONICAL, CANONICAL))
    lon = rng.standard_normal((CANONICAL, CANONICAL))
    lat = rng.standard_normal((CANONICAL, CANONICAL))
    if corrupt_first_time:
        truth[0, :1] = np.nan

    datasets = []
    for tile in layout.tiles:
        j = slice(tile.j_start, tile.j_end)
        i = slice(tile.i_start, tile.i_end)
        # Copy: numpy slices are views, so without this a per-tile edit would
        # silently propagate to every other tile and mask a real disagreement.
        datasets.append(
            xr.Dataset(
                {
                    "prognostic": (
                        ("time", "prognostic_channel", "y", "x"),
                        truth[:, :, j, i].copy(),
                    ),
                    "XC": (("y", "x"), lon[j, i].copy()),
                    "YC": (("y", "x"), lat[j, i].copy()),
                    "prognostic_mean": (("prognostic_channel",), np.zeros(2)),
                },
                coords={
                    "time": np.arange(times),
                    "x": np.arange(tile.i_start, tile.i_end),
                    "y": np.arange(tile.j_start, tile.j_end),
                },
                attrs={"prognostic_channel_names_json": '["a", "b"]'},
            )
        )
    return layout, datasets


def test_validate_accepts_tiles_cut_from_one_truth_field() -> None:
    layout, datasets = make_group_datasets()
    report = validate_tile_group(layout, datasets, probe_vars=("prognostic",))
    assert report.is_clean


def test_validate_rejects_tiles_that_disagree_in_their_claimed_overlap() -> None:
    """The topology-independent gate: XC/YC must match exactly where tiles meet."""
    layout, datasets = make_group_datasets()
    datasets[1]["XC"][0, 0] += 1.0
    with pytest.raises(ValueError, match="disagree on XC"):
        validate_tile_group(layout, datasets, probe_vars=("prognostic",))


def test_validate_rejects_mismatched_times_channels_stats_and_shapes() -> None:
    layout, datasets = make_group_datasets()
    shifted = [d.copy() for d in datasets]
    shifted[2] = shifted[2].assign_coords(time=np.arange(4) + 100)
    with pytest.raises(ValueError, match="time coordinate does not match"):
        validate_tile_group(layout, shifted, probe_vars=("prognostic",))

    renamed = [d.copy() for d in datasets]
    renamed[1].attrs["prognostic_channel_names_json"] = '["a", "c"]'
    with pytest.raises(ValueError, match="channel_names_json differs"):
        validate_tile_group(layout, renamed, probe_vars=("prognostic",))

    restated = [d.copy() for d in datasets]
    restated[3]["prognostic_mean"] = restated[3]["prognostic_mean"] + 1.0
    with pytest.raises(ValueError, match="prognostic_mean differs"):
        validate_tile_group(layout, restated, probe_vars=("prognostic",))

    cropped = [d.copy() for d in datasets]
    cropped[0] = cropped[0].isel(x=slice(0, TILE - 1))
    with pytest.raises(ValueError, match="declares shape"):
        validate_tile_group(layout, cropped, probe_vars=("prognostic",))


def test_validate_reports_nonfinite_times_without_raising() -> None:
    """Every current cache has an all-NaN first forcing timestamp; report, don't crash."""
    layout, datasets = make_group_datasets(corrupt_first_time=True)
    report = validate_tile_group(layout, datasets, probe_vars=("prognostic",))
    assert not report.is_clean
    assert report.nonfinite_times == (0,)
    assert report.nonfinite_detail[(0, "prognostic")] > 0


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", WINDOWS)
def test_normalized_weights_sum_to_one_everywhere(window) -> None:
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    ones = torch.ones(layout.num_tiles, 1, TILE, TILE, dtype=torch.float64)
    torch.testing.assert_close(
        blender.to_canonical(ones), torch.ones(1, CANONICAL, CANONICAL, dtype=torch.float64)
    )
    assert bool((blender.denominator > 0).all())


def test_quintic_raw_weights_are_a_partition_of_unity_including_the_corner() -> None:
    """Separability gives the 4-tile corner for free: (1-fx+fx)(1-fy+fy) = 1."""
    layout = make_layout()
    blender = TileBlender(layout, window="quintic", dtype=torch.float64)
    torch.testing.assert_close(
        blender.denominator, torch.ones_like(blender.denominator)
    )
    corner = slice(STRIDE, TILE)
    assert blender.weights[:, :, corner, corner].shape[-2:] == (OVERLAP, OVERLAP)


def test_kbd_raw_weights_do_not_sum_to_one_so_normalization_matters() -> None:
    layout = make_layout()
    blender = TileBlender(layout, window="kbd", dtype=torch.float64)
    assert not torch.allclose(blender.denominator, torch.ones_like(blender.denominator))
    assert bool((blender.denominator > 0).all())


@pytest.mark.parametrize("window", WINDOWS)
def test_exterior_sides_are_untapered(window) -> None:
    """No neighbour means nothing to hand off to, so weight stays at 1."""
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    # Tile 0 is exterior on ilo and jlo, interior on ihi and jhi.
    weights = blender.weights[0, 0]
    torch.testing.assert_close(weights[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert weights[-1, -1] < 1.0


@pytest.mark.parametrize("window", WINDOWS)
def test_idempotence_tiles_cut_from_one_field_blend_back_to_it(window) -> None:
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    truth = torch.randn(3, CANONICAL, CANONICAL, dtype=torch.float64)
    tiles = crop_tiles(truth, layout)

    torch.testing.assert_close(blender.to_canonical(tiles), truth)
    torch.testing.assert_close(blender.blend(tiles), tiles)


@pytest.mark.parametrize("window", WINDOWS)
def test_constant_reproduction(window) -> None:
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    tiles = torch.full((layout.num_tiles, 2, TILE, TILE), 3.25, dtype=torch.float64)
    torch.testing.assert_close(blender.blend(tiles), tiles)


@pytest.mark.parametrize("window", WINDOWS)
def test_linear_reproduction_is_exact(window) -> None:
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    j = torch.arange(CANONICAL, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(CANONICAL, dtype=torch.float64).unsqueeze(0)
    ramp = (2.0 * j - 3.0 * i + 7.0).unsqueeze(0)
    tiles = crop_tiles(ramp, layout)
    torch.testing.assert_close(blender.to_canonical(tiles), ramp)


@pytest.mark.parametrize("window", WINDOWS)
def test_blend_is_invariant_to_tile_ordering(window) -> None:
    tiles_spec = make_tiles()
    forward = TileBlender(build_group_layout(tiles_spec), window=window, dtype=torch.float64)
    order = [3, 1, 0, 2]
    reversed_layout = build_group_layout([tiles_spec[k] for k in order])
    backward = TileBlender(reversed_layout, window=window, dtype=torch.float64)

    data = torch.randn(4, 2, TILE, TILE, dtype=torch.float64)
    torch.testing.assert_close(
        forward.blend(data), backward.blend(data[order])[
            [order.index(k) for k in range(4)]
        ]
    )


@pytest.mark.parametrize("window", WINDOWS)
def test_residual_blend_equals_full_field_blend_when_current_states_agree(window) -> None:
    """The invariant replay must preserve: identical overlap values before each step."""
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    current = torch.randn(2, CANONICAL, CANONICAL, dtype=torch.float64)
    current_tiles = crop_tiles(current, layout)
    residuals = torch.randn(4, 2, TILE, TILE, dtype=torch.float64)

    via_residuals = current_tiles + blender.blend(residuals)
    via_full_fields = blender.blend(current_tiles + residuals)
    torch.testing.assert_close(via_residuals, via_full_fields)


@pytest.mark.parametrize("window", WINDOWS)
def test_blend_supports_a_leading_batch_dimension(window) -> None:
    layout = make_layout()
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    batched = torch.randn(3, 4, 2, TILE, TILE, dtype=torch.float64)
    out = blender.blend(batched)
    assert out.shape == batched.shape
    for b in range(3):
        torch.testing.assert_close(out[b], blender.blend(batched[b]))


@pytest.mark.parametrize("window", WINDOWS)
def test_blend_gradcheck(window) -> None:
    layout = build_group_layout(make_tiles(tile=5, stride=4))
    blender = TileBlender(layout, window=window, dtype=torch.float64)
    data = torch.randn(4, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(blender.blend, (data,))


@pytest.mark.parametrize("window", WINDOWS)
def test_coverage_excludes_padding_cells_from_the_denominator(window) -> None:
    """Coverage is 'is this a real cell', kept distinct from wetness."""
    layout = make_layout()
    coverage = torch.ones(layout.num_tiles, 1, TILE, TILE, dtype=torch.float64)
    # Tile 1's first column sits inside its overlap with tile 0, so dropping it
    # leaves a real contributor behind. Dropping a uniquely-owned cell instead
    # would (correctly) leave the canonical grid uncovered.
    coverage[1, :, :, 0] = 0.0
    blender = TileBlender(layout, window=window, coverage=coverage, dtype=torch.float64)

    truth = torch.randn(2, CANONICAL, CANONICAL, dtype=torch.float64)
    tiles = crop_tiles(truth, layout)
    tiles[1, :, :, 0] = 12345.0  # garbage in the excluded column
    torch.testing.assert_close(blender.to_canonical(tiles), truth)


def test_blender_rejects_a_layout_it_cannot_cover() -> None:
    left = TileSpec(
        tile_id=0, dataset_index=0, face=1,
        i_start=0, i_end=4, j_start=0, j_end=4, owned=(0, 4, 0, 4),
    )
    right = TileSpec(
        tile_id=1, dataset_index=1, face=1,
        i_start=6, i_end=10, j_start=0, j_end=4, owned=(0, 4, 0, 4),
    )
    layout = build_group_layout([left, right])
    with pytest.raises(ValueError, match="zero total weight"):
        TileBlender(layout, dtype=torch.float64)


def test_blender_rejects_mismatched_input_shapes() -> None:
    blender = TileBlender(make_layout(), dtype=torch.float64)
    with pytest.raises(ValueError, match="Expected 4 tiles"):
        blender.blend(torch.zeros(3, 2, TILE, TILE, dtype=torch.float64))
    with pytest.raises(ValueError, match="Expected tiles shaped"):
        blender.blend(torch.zeros(4, 2, TILE + 1, TILE, dtype=torch.float64))
    with pytest.raises(ValueError, match=r"Expected \[T, C, H, W\]"):
        blender.blend(torch.zeros(4, TILE, TILE, dtype=torch.float64))


@pytest.mark.parametrize("window", WINDOWS)
def test_ramp_wider_than_the_overlap_still_reproduces_truth(window) -> None:
    """A whole-tile taper is STRATA's shape; normalization keeps it exact."""
    layout = make_layout()
    blender = TileBlender(
        layout, window=window, ramp_width=TILE // 2, dtype=torch.float64
    )
    assert bool((blender.denominator > 0).all())

    truth = torch.randn(2, CANONICAL, CANONICAL, dtype=torch.float64)
    tiles = crop_tiles(truth, layout)
    torch.testing.assert_close(blender.to_canonical(tiles), truth)

    # It really is a different operator: the seam weights are not the defaults.
    default = TileBlender(layout, window=window, dtype=torch.float64)
    assert not torch.allclose(blender.weights, default.weights)


def test_colliding_ramps_are_rejected() -> None:
    """Only a tile with neighbours on both ends of an axis can over-taper."""
    row = [
        TileSpec(
            tile_id=k, dataset_index=k, face=1,
            i_start=k * STRIDE, i_end=k * STRIDE + TILE,
            j_start=0, j_end=TILE, owned=(0, TILE, 0, TILE),
        )
        for k in range(3)
    ]
    layout = build_group_layout(row)
    assert layout.overlaps[(1, "ilo")] == OVERLAP
    assert layout.overlaps[(1, "ihi")] == OVERLAP

    TileBlender(layout, ramp_width=TILE // 2, dtype=torch.float64)  # fits
    with pytest.raises(ValueError, match="collide"):
        TileBlender(layout, ramp_width=TILE, dtype=torch.float64)
