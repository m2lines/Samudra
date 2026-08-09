"""Test the tiled rollout's step algebra, seam bookkeeping, and probes.

These cover the pieces that decide whether a blended inference run is
trustworthy, without standing up a checkpoint or a data pipeline.
"""

import numpy as np
import pytest
import torch

from ocean_emulators.tiled_eval import (
    _seam_pairs,
    advance_state,
    apply_perturbation,
    preblend_disagreement,
)
from ocean_emulators.tiling import TileBlender, TileSpec, build_group_layout

TILE = 23
OVERLAP = 2
STRIDE = TILE - OVERLAP
CANONICAL = TILE + STRIDE


def make_layout():
    tiles = [
        TileSpec(
            tile_id=tile_id, dataset_index=tile_id, face=1,
            i_start=i0, i_end=i0 + TILE, j_start=j0, j_end=j0 + TILE,
            owned=(0, TILE, 0, TILE),
        )
        for tile_id, (j0, i0) in enumerate(
            [(0, 0), (0, STRIDE), (STRIDE, 0), (STRIDE, STRIDE)]
        )
    ]
    return build_group_layout(tiles)


def crop_tiles(field, layout):
    return torch.stack(
        [
            field[:, tile.j_start : tile.j_end, tile.i_start : tile.i_end]
            for tile in layout.tiles
        ]
    )


# --------------------------------------------------------------------------
# Seam bookkeeping
# --------------------------------------------------------------------------


def test_seam_pairs_finds_the_four_edge_seams_and_skips_the_corner() -> None:
    """A 2x2 has four edge seams; the diagonal pair shares only the corner
    square, which the two edge seams already cover."""
    pairs = _seam_pairs(make_layout())
    assert len(pairs) == 4
    assert {pair.axis for pair in pairs} == {"i", "j"}
    assert sum(pair.axis == "i" for pair in pairs) == 2
    for pair in pairs:
        assert pair.width == OVERLAP
        assert pair.left_tile != pair.right_tile


def test_seam_slices_address_the_same_physical_cells_in_both_tiles() -> None:
    """If these disagreed, the disagreement metric would compare unrelated cells."""
    layout = make_layout()
    truth = torch.arange(float(CANONICAL * CANONICAL)).reshape(1, CANONICAL, CANONICAL)
    tiles = crop_tiles(truth, layout)
    for pair in _seam_pairs(layout):
        left = tiles[pair.left_tile][(slice(None), *pair.left_slice)]
        right = tiles[pair.right_tile][(slice(None), *pair.right_slice)]
        torch.testing.assert_close(left, right)


# --------------------------------------------------------------------------
# Pre-blend disagreement
# --------------------------------------------------------------------------


def test_identical_tiles_disagree_by_zero() -> None:
    layout = make_layout()
    pairs = _seam_pairs(layout)
    truth = torch.randn(1, CANONICAL, CANONICAL, dtype=torch.float64)
    summary, _ = preblend_disagreement(crop_tiles(truth, layout), pairs)
    assert summary.shape == (len(pairs), 1, OVERLAP)
    np.testing.assert_allclose(summary, 0.0, atol=1e-12)


def test_disagreement_recovers_a_known_offset() -> None:
    layout = make_layout()
    pairs = _seam_pairs(layout)
    residuals = torch.zeros(4, 1, TILE, TILE, dtype=torch.float64)
    residuals[1] += 0.25  # tile 1 predicts a constant offset from every neighbour

    summary, _ = preblend_disagreement(residuals, pairs)
    for index, pair in enumerate(pairs):
        involved = 1 in (pair.left_tile, pair.right_tile)
        expected = 0.25 if involved else 0.0
        np.testing.assert_allclose(summary[index], expected, atol=1e-12)


def test_full_mode_returns_raw_bands_keyed_by_seam() -> None:
    """Vertical and horizontal bands are transposed, so they cannot share one
    array; they come back per seam for the writer to group by orientation."""
    layout = make_layout()
    pairs = _seam_pairs(layout)
    residuals = torch.zeros(4, 2, TILE, TILE, dtype=torch.float64)
    _, bands = preblend_disagreement(residuals, pairs, include_bands=True)

    assert set(bands) == {pair.name for pair in pairs}
    shapes = {name: band.shape for name, band in bands.items()}
    assert {(2, TILE, OVERLAP), (2, OVERLAP, TILE)} == set(shapes.values())
    for pair in pairs:
        expected = (2, TILE, OVERLAP) if pair.axis == "i" else (2, OVERLAP, TILE)
        assert shapes[pair.name] == expected


def test_summary_stacks_uniformly_across_both_seam_orientations() -> None:
    """Reducing along the seam is what makes the two orientations commensurate."""
    layout = make_layout()
    pairs = _seam_pairs(layout)
    residuals = torch.randn(4, 3, TILE, TILE, dtype=torch.float64)
    summary, _ = preblend_disagreement(residuals, pairs)
    assert summary.shape == (len(pairs), 3, OVERLAP)


# --------------------------------------------------------------------------
# Step algebra
# --------------------------------------------------------------------------


def make_inputs_and_residuals(layout, *, num_out=2, extra=3):
    """Inputs shaped like a real model input: prognostic channels then the rest."""
    torch.manual_seed(0)
    current = torch.randn(num_out, CANONICAL, CANONICAL, dtype=torch.float64)
    prognostic = crop_tiles(current, layout)
    trailing = torch.randn(4, extra, TILE, TILE, dtype=torch.float64)
    inputs = torch.cat([prognostic, trailing], dim=1)
    residuals = torch.randn(4, num_out, TILE, TILE, dtype=torch.float64)
    return current, inputs, residuals


def test_advance_state_reconciles_overlaps_so_the_next_step_sees_one_value() -> None:
    """The invariant the whole scheme rests on: after a step, every overlap
    holds identical values, so residual blending stays equivalent to full-field
    blending on the step after."""
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    _, inputs, residuals = make_inputs_and_residuals(layout)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool)] * 4

    state = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=True
    )
    for pair in _seam_pairs(layout):
        left = state[pair.left_tile][(slice(None), *pair.left_slice)]
        right = state[pair.right_tile][(slice(None), *pair.right_slice)]
        torch.testing.assert_close(left, right)


def test_without_blending_overlaps_still_disagree() -> None:
    """The hard-crop control must genuinely differ, or rungs 2 and 3 of the
    ladder would be measuring the same thing."""
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    _, inputs, residuals = make_inputs_and_residuals(layout)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool)] * 4

    state = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=False
    )
    pair = _seam_pairs(layout)[0]
    left = state[pair.left_tile][(slice(None), *pair.left_slice)]
    right = state[pair.right_tile][(slice(None), *pair.right_slice)]
    assert not torch.allclose(left, right)


def test_consistent_predictions_pass_through_the_blend_untouched() -> None:
    """When tiles already agree there is nothing to reconcile, so blending must
    be a no-op -- otherwise it would be smoothing real structure."""
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    current, inputs, _ = make_inputs_and_residuals(layout)
    global_residual = torch.randn(2, CANONICAL, CANONICAL, dtype=torch.float64)
    residuals = crop_tiles(global_residual, layout)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool)] * 4

    state = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=True
    )
    torch.testing.assert_close(state, crop_tiles(current + global_residual, layout))
    torch.testing.assert_close(
        blender.to_canonical(state.unsqueeze(0))[0], current + global_residual
    )


def test_each_tile_is_remasked_with_its_own_wet_mask() -> None:
    """The live group's fourth tile has land the others do not; a shared mask
    would zero live cells in the wrong places."""
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    _, inputs, residuals = make_inputs_and_residuals(layout)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool) for _ in range(4)]
    wet[3][:, :4, :4] = False  # only tile 3 has land

    state = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=True
    )
    assert torch.all(state[3, :, :4, :4] == 0.0)
    assert not torch.all(state[0, :, :4, :4] == 0.0)


def test_single_tile_group_is_an_unblended_rollout() -> None:
    """A one-tile group has no overlap, so a blended step must equal a plain
    one. This is the regression guard that tiling has not changed 1x1 behaviour."""
    tile = TileSpec(
        tile_id=0, dataset_index=0, face=1,
        i_start=0, i_end=TILE, j_start=0, j_end=TILE, owned=(0, TILE, 0, TILE),
    )
    layout = build_group_layout([tile])
    blender = TileBlender(layout, dtype=torch.float64)

    torch.manual_seed(1)
    inputs = torch.randn(1, 5, TILE, TILE, dtype=torch.float64)
    residuals = torch.randn(1, 2, TILE, TILE, dtype=torch.float64)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool)]

    blended = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=True
    )
    plain = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=False
    )
    torch.testing.assert_close(blended, plain)
    torch.testing.assert_close(blended, inputs[:, :2] + residuals)


# --------------------------------------------------------------------------
# Perturbation probe
# --------------------------------------------------------------------------


def test_perturbation_touches_only_prognostic_channels_inside_the_box() -> None:
    inputs = torch.zeros(2, 5, TILE, TILE, dtype=torch.float64)
    perturbed = apply_perturbation(
        inputs, num_out=2, centre=(11, 11), box=4, amplitude=1.5
    )
    assert torch.all(perturbed[:, :2, 9:13, 9:13] == 1.5)
    perturbed[:, :2, 9:13, 9:13] = 0.0
    assert torch.all(perturbed == 0.0), "perturbation leaked outside the box"


def test_zero_amplitude_perturbation_is_a_no_op() -> None:
    """A zero response is the control the far-field test is measured against."""
    torch.manual_seed(2)
    inputs = torch.randn(2, 5, TILE, TILE, dtype=torch.float64)
    perturbed = apply_perturbation(
        inputs, num_out=2, centre=(11, 11), box=8, amplitude=0.0
    )
    torch.testing.assert_close(perturbed, inputs)


def test_perturbation_box_is_clipped_at_the_tile_edge() -> None:
    inputs = torch.zeros(1, 3, TILE, TILE, dtype=torch.float64)
    perturbed = apply_perturbation(
        inputs, num_out=1, centre=(0, 0), box=8, amplitude=1.0
    )
    assert torch.all(perturbed[:, 0, :4, :4] == 1.0)
    assert perturbed.shape == inputs.shape


@pytest.mark.parametrize("blend", [True, False])
def test_advance_state_preserves_shape(blend) -> None:
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    _, inputs, residuals = make_inputs_and_residuals(layout)
    wet = [torch.ones(2, TILE, TILE, dtype=torch.bool)] * 4
    state = advance_state(
        inputs, residuals, blender=blender, tile_wet=wet, num_out=2, blend=blend
    )
    assert state.shape == (4, 2, TILE, TILE)


def test_unnormalizing_per_tile_then_stitching_equals_stitching_then_unnormalizing() -> None:
    """Why the canonical write unnormalizes before it stitches.

    Normalize carries a single tile's wet mask, so it cannot be applied to a
    720x720 canonical frame at all. Doing it per tile first is valid because
    unnormalization is affine per channel and the blend is a weighted mean whose
    weights sum to one, so the two commute exactly.
    """
    layout = make_layout()
    blender = TileBlender(layout, dtype=torch.float64)
    torch.manual_seed(4)

    normalized = torch.randn(4, 3, TILE, TILE, dtype=torch.float64)
    mean = torch.randn(3, 1, 1, dtype=torch.float64)
    std = torch.rand(3, 1, 1, dtype=torch.float64) + 0.5

    def unnormalize(x):
        return x * std + mean

    per_tile_first = blender.to_canonical(unnormalize(normalized).unsqueeze(0))[0]
    stitch_first = unnormalize(blender.to_canonical(normalized.unsqueeze(0))[0])
    torch.testing.assert_close(per_tile_first, stitch_first)
