"""Test grouped replay: one cursor per cluster, overlaps reconciled on writeback.

The property that matters is the invariant the whole scheme rests on -- after a
step, every overlap holds identical values -- because that is what makes blending
residuals equivalent to blending full fields on the step after. The rest of these
guard the ways a group can silently come apart: tiles pairing with the wrong
gold target, a row outliving the layout that shaped it, or the single-cache path
quietly changing.
"""

import pytest
import torch

from ocean_emulators.datasets import ReplayBatchSlot, ReplayCursor
from ocean_emulators.replay import ReplayEntry
from ocean_emulators.tiling import ReplayGroup, TileSpec, build_replay_groups
from ocean_emulators.train import Trainer

TILE = 4
STRIDE = 3
OVERLAP = TILE - STRIDE
CANONICAL = TILE + STRIDE
ORIGINS = [(0, 0), (0, STRIDE), (STRIDE, 0), (STRIDE, STRIDE)]


def make_tiles(face: int = 1) -> list[TileSpec]:
    return [
        TileSpec(
            tile_id=index, dataset_index=index, face=face,
            i_start=i0, i_end=i0 + TILE, j_start=j0, j_end=j0 + TILE,
            owned=(0, TILE, 0, TILE),
        )
        for index, (j0, i0) in enumerate(ORIGINS)
    ]


# --------------------------------------------------------------------------
# Group layout
# --------------------------------------------------------------------------


def test_ungrouped_layout_is_the_identity_so_nothing_downstream_changes() -> None:
    """A group index and a dataset index must be the same number when grouping
    is off, or every existing cursor would start addressing the wrong data."""
    groups = build_replay_groups(num_sources=3, num_strides=2, grouped=False)
    assert len(groups) == 6
    for index, group in enumerate(groups):
        assert group.dataset_indices == (index,)
        assert not group.is_grouped
        assert group.blender is None


def test_a_single_cache_is_ungrouped_even_when_grouping_is_requested() -> None:
    """One tile per group is the identity, so there is nothing to reconcile."""
    groups = build_replay_groups(num_sources=1, num_strides=3, grouped=True)
    assert [group.dataset_indices for group in groups] == [(0,), (1,), (2,)]
    assert all(not group.is_grouped for group in groups)


def test_groups_are_taken_across_sources_at_a_fixed_stride() -> None:
    """train_datasets is [source x stride], and two strides are two clocks -- a
    group spanning strides could not share one cursor."""
    groups = build_replay_groups(
        num_sources=4, num_strides=2, grouped=True, tiles=make_tiles()
    )
    assert len(groups) == 2
    assert groups[0].dataset_indices == (0, 2, 4, 6)  # stride slot 0
    assert groups[1].dataset_indices == (1, 3, 5, 7)  # stride slot 1
    for group in groups:
        assert group.is_grouped
        assert group.blender is not None
        assert group.layout.canonical_shape == (CANONICAL, CANONICAL)


def test_grouping_requires_one_tile_spec_per_source() -> None:
    with pytest.raises(ValueError, match="one TileSpec per source"):
        build_replay_groups(
            num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()[:2]
        )


# --------------------------------------------------------------------------
# Row packing
# --------------------------------------------------------------------------


def make_trainer(groups: list[ReplayGroup]) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.replay_groups = groups
    trainer.tile_wet_masks = None
    trainer.data_stride = [1]
    return trainer


def test_a_one_tile_row_keeps_its_bare_shape() -> None:
    """Sidecars store these tensors, so an ungrouped row must stay exactly the
    shape it was before grouping existed."""
    trainer = make_trainer(build_replay_groups(num_sources=1, num_strides=1, grouped=False))
    state = torch.randn(5, 4, 4)
    assert trainer._stack_tile_states([state]).shape == (5, 4, 4)


def test_a_multi_tile_row_gains_a_tile_axis() -> None:
    trainer = make_trainer(
        build_replay_groups(num_sources=4, num_strides=1, grouped=True, tiles=make_tiles())
    )
    states = [torch.randn(5, TILE, TILE) for _ in range(4)]
    assert trainer._stack_tile_states(states).shape == (4, 5, TILE, TILE)


def test_entry_states_accepts_both_row_shapes() -> None:
    """[C,H,W] is a pre-grouping sidecar; [T,C,H,W] is a grouped row."""
    trainer = make_trainer(build_replay_groups(num_sources=1, num_strides=1, grouped=False))
    cursor = ReplayCursor(0, 0, 0, 1, 1)

    legacy = ReplayEntry(state=torch.randn(5, TILE, TILE), cursor=cursor)
    assert [s.shape for s in trainer._entry_states(legacy, 1)] == [(1, 5, TILE, TILE)]

    grouped = ReplayEntry(state=torch.randn(4, 5, TILE, TILE), cursor=cursor)
    assert len(trainer._entry_states(grouped, 4)) == 4


def test_a_row_from_a_different_layout_is_rejected_not_mis_paired() -> None:
    """Resuming across a layout change must fail loudly; pairing four tile states
    against a one-tile group would train on silently shifted fields."""
    trainer = make_trainer(
        build_replay_groups(num_sources=4, num_strides=1, grouped=True, tiles=make_tiles())
    )
    entry = ReplayEntry(
        state=torch.randn(2, 5, TILE, TILE), cursor=ReplayCursor(0, 0, 0, 1, 1)
    )
    with pytest.raises(ValueError, match="diverged"):
        trainer._entry_states(entry, 4)


# --------------------------------------------------------------------------
# Transition ordering
# --------------------------------------------------------------------------


class _FakeTransition:
    def __init__(self, tile_index: int):
        self.tile_index = tile_index


def test_transitions_are_re_chunked_per_slot_in_tile_order() -> None:
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    slots = [
        ReplayBatchSlot(replay_index=index, cursor=ReplayCursor(0, index, 0, 1, 1))
        for index in range(2)
    ]
    transitions = [_FakeTransition(t) for _ in range(2) for t in range(4)]

    chunks = trainer._transitions_per_slot(slots, transitions)
    assert [len(chunk) for _, chunk in chunks] == [4, 4]
    assert [t.tile_index for _, chunk in chunks for t in chunk] == [0, 1, 2, 3] * 2


def test_mis_ordered_transitions_are_rejected() -> None:
    """A shuffled load would pair a tile's state with a neighbour's gold target,
    which trains on a shifted field and looks like a physics problem."""
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    slots = [ReplayBatchSlot(replay_index=0, cursor=ReplayCursor(0, 0, 0, 1, 1))]
    with pytest.raises(RuntimeError, match="out of tile order"):
        trainer._transitions_per_slot(
            slots, [_FakeTransition(t) for t in (0, 2, 1, 3)]
        )


def test_a_short_transition_list_is_rejected() -> None:
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    slots = [ReplayBatchSlot(replay_index=0, cursor=ReplayCursor(0, 0, 0, 1, 1))]
    with pytest.raises(RuntimeError, match="expected 4 transition"):
        trainer._transitions_per_slot(slots, [_FakeTransition(0), _FakeTransition(1)])


# --------------------------------------------------------------------------
# The invariant: overlaps agree after a step
# --------------------------------------------------------------------------


def crop_tiles(field: torch.Tensor, groups) -> torch.Tensor:
    layout = groups[0].layout
    return torch.stack(
        [
            field[:, tile.j_start : tile.j_end, tile.i_start : tile.i_end]
            for tile in layout.tiles
        ]
    )


def test_writeback_leaves_every_overlap_holding_one_value() -> None:
    """The invariant the scheme rests on. Without it, residual blending stops
    being equivalent to full-field blending on the very next step."""
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles(),
        dtype=torch.float64,
    )
    layout = groups[0].layout
    blender = groups[0].blender

    torch.manual_seed(0)
    current = crop_tiles(torch.randn(3, CANONICAL, CANONICAL, dtype=torch.float64), groups)
    # Tiles disagree in the overlap, as they always do out of the model.
    raw_prediction = current + torch.randn_like(current)

    residual = raw_prediction - current
    reconciled = current + blender.blend(residual)

    for a_index, tile in enumerate(layout.tiles):
        for b_index, other in enumerate(layout.tiles):
            if b_index <= a_index:
                continue
            j0 = max(tile.j_start, other.j_start)
            j1 = min(tile.j_end, other.j_end)
            i0 = max(tile.i_start, other.i_start)
            i1 = min(tile.i_end, other.i_end)
            if j1 <= j0 or i1 <= i0:
                continue
            left = reconciled[a_index][
                :, j0 - tile.j_start : j1 - tile.j_start,
                i0 - tile.i_start : i1 - tile.i_start,
            ]
            right = reconciled[b_index][
                :, j0 - other.j_start : j1 - other.j_start,
                i0 - other.i_start : i1 - other.i_start,
            ]
            torch.testing.assert_close(left, right)

    # And the unreconciled prediction genuinely did NOT agree, or the test above
    # would pass for the wrong reason.
    tile, other = layout.tiles[0], layout.tiles[1]
    j0 = max(tile.j_start, other.j_start)
    j1 = min(tile.j_end, other.j_end)
    i0 = max(tile.i_start, other.i_start)
    i1 = min(tile.i_end, other.i_end)
    assert not torch.allclose(
        raw_prediction[0][:, j0 - tile.j_start : j1 - tile.j_start,
                          i0 - tile.i_start : i1 - tile.i_start],
        raw_prediction[1][:, j0 - other.j_start : j1 - other.j_start,
                          i0 - other.i_start : i1 - other.i_start],
    )


def test_reconciling_agreeing_tiles_changes_nothing() -> None:
    """When the tiles already agree there is nothing to reconcile, so the blend
    must be a no-op rather than smoothing real structure."""
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles(),
        dtype=torch.float64,
    )
    torch.manual_seed(1)
    truth = torch.randn(3, CANONICAL, CANONICAL, dtype=torch.float64)
    current = crop_tiles(truth, groups)
    consistent_residual = crop_tiles(
        torch.randn(3, CANONICAL, CANONICAL, dtype=torch.float64), groups
    )

    reconciled = current + groups[0].blender.blend(consistent_residual)
    torch.testing.assert_close(reconciled, current + consistent_residual)


# --------------------------------------------------------------------------
# Per-tile wet weighting
# --------------------------------------------------------------------------


def test_no_per_sample_weight_when_every_tile_shares_a_mask() -> None:
    """The common path must allocate nothing and leave the loss untouched."""
    trainer = make_trainer(
        build_replay_groups(num_sources=4, num_strides=1, grouped=True, tiles=make_tiles())
    )
    slots = [ReplayBatchSlot(replay_index=0, cursor=ReplayCursor(0, 0, 0, 1, 1))]
    assert trainer._batch_wet_weight(slots) is None


def test_per_sample_weight_gives_each_tile_its_own_mask() -> None:
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    masks = torch.ones(4, 2, TILE, TILE, dtype=torch.bool)
    masks[3, :, :2, :2] = False  # only the fourth tile has land
    trainer.tile_wet_masks = masks

    slots = [ReplayBatchSlot(replay_index=0, cursor=ReplayCursor(0, 0, 0, 1, 1))]
    weight = trainer._batch_wet_weight(slots)

    assert weight is not None
    assert weight.shape == (4, 2, TILE, TILE)
    assert weight[3, :, :2, :2].sum() == 0.0
    assert weight[0].sum() == 2 * TILE * TILE


def test_per_sample_weight_follows_the_source_not_the_stride() -> None:
    """dataset index is source-major over strides; a stride is not a new mask."""
    groups = build_replay_groups(
        num_sources=4, num_strides=2, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    trainer.data_stride = [1, 2]
    # Float, not bool, so each source stays distinguishable and a mis-mapping
    # to the wrong source cannot pass by collapsing to True.
    masks = torch.zeros(4, 1, TILE, TILE)
    for source in range(4):
        masks[source] = source + 1
    trainer.tile_wet_masks = masks

    # Group 1 is the stride-slot-1 group: datasets (1, 3, 5, 7), which are
    # sources 0..3 at stride slot 1. Dividing by num_strides recovers the source.
    slots = [ReplayBatchSlot(replay_index=0, cursor=ReplayCursor(1, 0, 0, 1, 1))]
    weight = trainer._batch_wet_weight(slots)
    assert [float(weight[t].max()) for t in range(4)] == [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------
# Grouped validation weighting
# --------------------------------------------------------------------------


def test_validation_weight_counts_every_cell_exactly_once() -> None:
    """The whole point of the ownership term: without it the shared cells are
    scored twice and every domain metric leans toward the seams."""
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    trainer.num_out = 3
    trainer.tile_wet_masks = None

    weight = trainer._grouped_val_weight(groups[0])
    assert weight.shape == (4, 3, TILE, TILE)
    # One channel's worth of weight must equal the canonical cell count.
    assert float(weight[:, 0].sum()) == float(CANONICAL * CANONICAL)


def test_validation_weight_also_carries_each_tile_s_own_land() -> None:
    """Land is per tile, so a shared mask would score one tile's ocean against
    another tile's coastline."""
    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    trainer.num_out = 2
    wet = torch.ones(4, 2, TILE, TILE, dtype=torch.bool)
    wet[3, :, -1, -1] = False  # a dry cell tile 3 owns
    trainer.tile_wet_masks = wet

    weight = trainer._grouped_val_weight(groups[0])
    assert float(weight[3, 0, -1, -1]) == 0.0
    # Exactly that one owned cell (per channel) is removed from the total.
    assert float(weight[:, 0].sum()) == float(CANONICAL * CANONICAL - 1)


def test_a_perfect_prediction_scores_zero_under_the_ownership_weight() -> None:
    """Sanity floor: the weighting must not invent error where there is none."""
    from ocean_emulators.utils.loss import decomposed_mse_mae

    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    trainer.num_out = 2
    trainer.tile_wet_masks = None
    weight = trainer._grouped_val_weight(groups[0])

    truth = torch.randn(4, 2, TILE, TILE)
    wet = torch.ones(2, TILE, TILE, dtype=torch.bool)
    loss = decomposed_mse_mae(truth, truth, wet=wet, sample_weight=weight)
    assert float(loss.abs().max()) == 0.0


def test_disowned_cells_cannot_affect_the_score() -> None:
    """Error placed only in cells another tile owns must not register, or the
    overlap would be contributing twice through the back door."""
    from ocean_emulators.utils.loss import decomposed_mse_mae

    groups = build_replay_groups(
        num_sources=4, num_strides=1, grouped=True, tiles=make_tiles()
    )
    trainer = make_trainer(groups)
    trainer.num_out = 2
    trainer.tile_wet_masks = None
    weight = trainer._grouped_val_weight(groups[0])

    truth = torch.zeros(4, 2, TILE, TILE)
    corrupted = truth.clone()
    # Tile 0 is exterior on its low sides, so its disowned band is the high edge.
    disowned = weight[0, 0] == 0
    corrupted[0, :, disowned] = 1000.0

    wet = torch.ones(2, TILE, TILE, dtype=torch.bool)
    loss = decomposed_mse_mae(corrupted, truth, wet=wet, sample_weight=weight)
    assert float(loss.abs().max()) == 0.0
