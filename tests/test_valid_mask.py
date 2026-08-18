import pytest
import torch

from ocean_emulators.config import DataConfig
from ocean_emulators.datasets import _append_static_channels, _static_input_channels
from ocean_emulators.models.modules.padding import apply_spatial_pad

GRID = (6, 6)


def _valid_only() -> torch.Tensor:
    stack = _static_input_channels(
        None, include_spatial=False, include_valid_mask=True, grid_shape=GRID
    )
    assert stack is not None
    return stack


def test_valid_mask_defaults_on():
    """New runs should be conditioned on it; existing jobs opt out in their .sh."""
    assert DataConfig.model_fields["valid_mask"].default is True


def test_static_channels_are_ordered_spatial_then_valid():
    """The order is baked into every checkpoint's input channel count, so it has
    to stay [spatial | valid]."""
    features = torch.randn(4, *GRID)
    stack = _static_input_channels(
        features, include_spatial=True, include_valid_mask=True, grid_shape=GRID
    )

    assert stack.shape == (5, *GRID)
    torch.testing.assert_close(stack[:4], features)
    torch.testing.assert_close(stack[4], torch.ones(GRID))


@pytest.mark.parametrize(
    ("spatial", "valid", "expected"),
    [(False, False, 0), (True, False, 4), (False, True, 1), (True, True, 5)],
)
def test_static_channel_counts(spatial, valid, expected):
    stack = _static_input_channels(
        torch.randn(4, *GRID),
        include_spatial=spatial,
        include_valid_mask=valid,
        grid_shape=GRID,
    )
    assert (0 if stack is None else stack.shape[0]) == expected


def test_valid_mask_works_without_spatial_features():
    """A cache with no XC/YC/rA must still be able to carry a valid mask -- that
    is the 1-tile case, where the geographic fields are often absent."""
    stack = _static_input_channels(
        None, include_spatial=False, include_valid_mask=True, grid_shape=GRID
    )
    torch.testing.assert_close(stack, torch.ones(1, *GRID))


def test_spatial_features_requested_but_absent_raises():
    with pytest.raises(ValueError, match="XC, YC, and rA"):
        _static_input_channels(
            None, include_spatial=True, include_valid_mask=False, grid_shape=GRID
        )


def test_nothing_is_appended_when_both_are_off():
    """Backward compatibility: with the mask off the input is untouched, so an
    existing checkpoint's channel count still matches."""
    data = torch.randn(2, 3, *GRID)
    stack = _static_input_channels(
        torch.randn(4, *GRID),
        include_spatial=False,
        include_valid_mask=False,
        grid_shape=GRID,
    )

    assert stack is None
    assert _append_static_channels(data, stack) is data


def test_appended_channels_broadcast_across_the_batch():
    data = torch.randn(3, 2, *GRID)
    out = _append_static_channels(data, _valid_only())

    assert out.shape == (3, 3, *GRID)
    for sample in range(3):
        torch.testing.assert_close(out[sample, 2], torch.ones(GRID))


def test_append_rejects_a_shape_mismatch():
    data = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="Static channel shape"):
        _append_static_channels(data, _valid_only())


def test_valid_mask_separates_padding_from_land():
    """The whole point. Land is filled with 0 and the network pads with 0, so in
    every other channel a padded cell and a land cell are the same number. Only
    the valid channel tells them apart.
    """
    data = torch.full((1, 1, *GRID), 0.7)
    data[0, 0, 2, 2] = 0.0  # a land cell: real data that happens to be zero

    padded = apply_spatial_pad(
        _append_static_channels(data, _valid_only()), 1, "constant"
    )

    land_data, land_valid = padded[0, 0, 3, 3], padded[0, 1, 3, 3]
    ring_data, ring_valid = padded[0, 0, 0, 0], padded[0, 1, 0, 0]

    # Indistinguishable in the data channel...
    assert float(land_data) == 0.0
    assert float(ring_data) == 0.0
    # ...and separated by the valid channel.
    assert float(land_valid) == 1.0
    assert float(ring_valid) == 0.0


def test_valid_channel_is_one_everywhere_inside_the_patch():
    """Every cell the patch holds is real, including a halo once halos land: the
    ring of zeros is created by padding, not by the data."""
    padded = apply_spatial_pad(
        _append_static_channels(torch.zeros(1, 1, *GRID), _valid_only()), 2, "constant"
    )
    valid = padded[0, 1]

    torch.testing.assert_close(valid[2:-2, 2:-2], torch.ones(GRID))
    assert float(valid[0].sum()) == 0.0
    assert float(valid[:, 0].sum()) == 0.0
