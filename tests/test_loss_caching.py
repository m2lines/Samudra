"""The loss caches constants. It must not change a single bit by doing so.

At face scale one loss call moved ~64 GB through the allocator, and a large
share of that was rebuilding tensors that are fixed for the whole run: the
bool->float casts of the wet mask and the channel weights, and gradient_z's
per-variable "which level pairs are wet" weight. Those are now cached.

Caching is only safe while the cached inputs really are constant, so this
file pins both halves of that: the results are bit-identical to the
expressions they replaced, and the cache is keyed tightly enough that a
different mask or a different channel weight cannot be served a stale one.
"""

import gc

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, build_loss_fn
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.loss import (
    _cast_constant,
    _channel_weight,
    _gradient_z_midpoint_weight,
    gradient_z_norms,
    weighted_channel_denominator,
)
from ocean_emulators.utils.multiton import MultitonScope

BATCH, SIZE = 3, 8


# --------------------------------------------------------------------------
# _cast_constant
# --------------------------------------------------------------------------


def test_a_cast_constant_is_the_same_tensor_every_time() -> None:
    mask = torch.ones(4, SIZE, SIZE, dtype=torch.bool)
    first = _cast_constant(mask, torch.float32)
    assert torch.equal(first, mask.to(dtype=torch.float32))
    assert _cast_constant(mask, torch.float32) is first


def test_each_dtype_gets_its_own_entry() -> None:
    mask = torch.ones(4, SIZE, SIZE, dtype=torch.bool)
    as_float = _cast_constant(mask, torch.float32)
    as_half = _cast_constant(mask, torch.float16)
    assert as_float.dtype == torch.float32
    assert as_half.dtype == torch.float16
    assert _cast_constant(mask, torch.float32) is as_float


def test_a_tensor_already_in_that_dtype_is_handed_straight_back() -> None:
    weight = torch.ones(4, SIZE, SIZE)
    assert _cast_constant(weight, torch.float32) is weight


def test_two_masks_with_equal_values_do_not_share_an_entry() -> None:
    """Keyed by identity, not by value -- `Tensor.__eq__` is elementwise."""
    one = torch.ones(4, SIZE, SIZE, dtype=torch.bool)
    other = torch.ones(4, SIZE, SIZE, dtype=torch.bool)
    assert _cast_constant(one, torch.float32) is not _cast_constant(
        other, torch.float32
    )


def test_the_cache_does_not_pin_a_mask_it_was_given() -> None:
    """The curriculum rebuilds the loss at every step change."""
    from ocean_emulators.utils.loss import _CAST_CACHE

    mask = torch.ones(4, SIZE, SIZE, dtype=torch.bool)
    _cast_constant(mask, torch.float32)
    before = len(_CAST_CACHE)
    assert before >= 1
    del mask
    gc.collect()
    assert len(_CAST_CACHE) < before


def test_channel_weight_still_matches_the_expression_it_replaced() -> None:
    wet = torch.randint(0, 2, (4, SIZE, SIZE), dtype=torch.bool)
    spatial = torch.rand(4, SIZE, SIZE)
    extra = torch.randint(0, 2, (BATCH, 4, SIZE, SIZE), dtype=torch.bool)
    dtype = torch.float32

    reference = wet.to(dtype=dtype).unsqueeze(0)
    reference = reference * spatial.to(dtype=dtype).unsqueeze(0)
    reference = reference * extra.to(dtype=dtype)

    for _ in range(2):  # second call is the one that reads the cache
        assert torch.equal(
            _channel_weight(dtype, wet=wet, spatial_weight=spatial, extra_weight=extra),
            reference,
        )


# --------------------------------------------------------------------------
# gradient_z's per-variable midpoint weight
# --------------------------------------------------------------------------


def _midpoint_reference(wet, spatial, *, lower, upper, num_times, num_vars, dtype):
    """Exactly the expression `_gradient_z_weight` used to inline."""
    wet_by_time = wet.reshape(num_times, num_vars, *wet.shape[-2:]).bool()
    midpoint = (wet_by_time[:, upper] & wet_by_time[:, lower]).to(dtype=dtype)
    if spatial is not None:
        spatial_by_time = spatial.reshape(num_times, num_vars, *spatial.shape[-2:])
        midpoint = midpoint * torch.minimum(
            spatial_by_time[:, upper], spatial_by_time[:, lower]
        ).to(dtype=dtype)
    return midpoint


@pytest.mark.parametrize("with_spatial", [True, False])
def test_the_midpoint_weight_matches_the_expression_it_replaced(
    with_spatial: bool,
) -> None:
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        num_vars = len(tensor_map.prognostic_var_names)
        generator = torch.Generator().manual_seed(0)
        wet = torch.rand(num_vars, SIZE, SIZE, generator=generator) > 0.3
        spatial = (
            torch.rand(num_vars, SIZE, SIZE, generator=generator)
            if with_spatial
            else None
        )
        variable = sorted(tensor_map.VAR_SET_3D)[0]
        indices = tensor_map.VAR_3D_IDX[variable].to(dtype=torch.long)
        lower, upper = indices[:-1], indices[1:]
        kwargs = dict(
            lower=lower,
            upper=upper,
            num_times=1,
            num_vars=num_vars,
            dtype=torch.float32,
        )
        reference = _midpoint_reference(wet, spatial, **kwargs)
        for _ in range(2):  # the second reads the cache
            assert torch.equal(
                _gradient_z_midpoint_weight(
                    wet=wet, spatial_weight=spatial, variable=variable, **kwargs
                ),
                reference,
            )


def test_a_different_channel_weight_is_not_served_the_cached_one() -> None:
    """The mask keys the cache; the channel weight is checked by identity."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        num_vars = len(tensor_map.prognostic_var_names)
        wet = torch.ones(num_vars, SIZE, SIZE, dtype=torch.bool)
        variable = sorted(tensor_map.VAR_SET_3D)[0]
        indices = tensor_map.VAR_3D_IDX[variable].to(dtype=torch.long)
        kwargs = dict(
            variable=variable,
            lower=indices[:-1],
            upper=indices[1:],
            num_times=1,
            num_vars=num_vars,
            dtype=torch.float32,
        )
        ones = _gradient_z_midpoint_weight(
            wet=wet, spatial_weight=torch.ones(num_vars, SIZE, SIZE), **kwargs
        )
        halves = _gradient_z_midpoint_weight(
            wet=wet, spatial_weight=torch.full((num_vars, SIZE, SIZE), 0.5), **kwargs
        )
        assert torch.equal(halves, ones * 0.5)


# --------------------------------------------------------------------------
# The whole stack
# --------------------------------------------------------------------------


def _face_like_loss(channels, wet, sample_weight, y_coord):
    config = GradientLossConfig(
        type=["gradient_h", "gradient_z"],
        metric="mse_mae",
        lambda_h=0.1,
        lambda_z=0.1,
        channel_weights={"Eta": 5.0},
    )
    return build_loss_fn(
        config,
        wet,
        y_coord,
        torch.device("cpu"),
        channels,
        "constant",
        denominator=weighted_channel_denominator(
            wet=wet, batch=BATCH, extra_weight=sample_weight
        ),
        gradient_z_norms=gradient_z_norms(
            wet=wet, batch=BATCH, sample_weight=sample_weight
        ),
    )


def test_a_bool_sample_weight_scores_exactly_as_a_float_one() -> None:
    """`GradientAugmentedLoss` now casts the per-sample weight once, up front,
    instead of letting all seven consumers cast it themselves. That is only a
    saving if it is also a no-op, so: same bits either way."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        channels = len(tensor_map.prognostic_var_names)
        generator = torch.Generator().manual_seed(0)
        pred = torch.randn(BATCH, channels, SIZE, SIZE, generator=generator)
        target = torch.randn(BATCH, channels, SIZE, SIZE, generator=generator)
        wet = torch.rand(channels, SIZE, SIZE, generator=generator) > 0.2
        sample_weight = (
            torch.rand(BATCH, channels, SIZE, SIZE, generator=generator) > 0.1
        )
        y_coord = xr.DataArray(np.linspace(-60.0, 60.0, SIZE), dims="lat")

        loss_fn = _face_like_loss(channels, wet, sample_weight, y_coord)
        as_bool = loss_fn(pred, target, sample_weight=sample_weight)
        as_float = loss_fn(
            pred, target, sample_weight=sample_weight.to(dtype=torch.float32)
        )
        assert torch.equal(as_bool, as_float)


def test_repeated_calls_give_the_same_answer() -> None:
    """A stale cache would show up here and nowhere else."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        channels = len(tensor_map.prognostic_var_names)
        generator = torch.Generator().manual_seed(1)
        pred = torch.randn(BATCH, channels, SIZE, SIZE, generator=generator)
        target = torch.randn(BATCH, channels, SIZE, SIZE, generator=generator)
        wet = torch.rand(channels, SIZE, SIZE, generator=generator) > 0.2
        sample_weight = (
            torch.rand(BATCH, channels, SIZE, SIZE, generator=generator) > 0.1
        )
        y_coord = xr.DataArray(np.linspace(-60.0, 60.0, SIZE), dims="lat")
        loss_fn = _face_like_loss(channels, wet, sample_weight, y_coord)
        first = loss_fn(pred, target, sample_weight=sample_weight)
        for _ in range(3):
            assert torch.equal(
                loss_fn(pred, target, sample_weight=sample_weight), first
            )
