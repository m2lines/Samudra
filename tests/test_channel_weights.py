"""Per-channel loss weighting, and the batch normalization it sits on top of.

Two separate things are pinned here because they interact. `GradientLossConfig`
-- the loss every LLC run uses -- applied no channel weighting at all, so a
surface field like Eta was outvoted 51-to-1 by each 3D variable. And
`_weighted_channel_mean` scaled its denominator by the batch size even when the
weight already carried a batch axis, which shrank the loss by the number of
tiles in flight for exactly the multi-tile runs that need the weighting.
"""

import pytest
import torch

from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.loss import (
    DEFAULT_CHANNEL_WEIGHTS,
    WeightedLoss,
    channel_weight_vector,
    decomposed_mae,
    decomposed_mse,
    decomposed_mse_mae,
    gradient_h_l1_loss,
)
from ocean_emulators.utils.multiton import MultitonScope

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Batch normalization
# --------------------------------------------------------------------------

METRICS = [decomposed_mse, decomposed_mae]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("batch", [1, 2, 4, 9, 36])
def test_a_uniform_error_scores_the_same_at_every_batch_size(metric, batch) -> None:
    """The loss is a mean, so a constant error of 1 must score 1 whether one
    tile or a whole face is in flight -- otherwise the effective learning rate
    depends on how many tiles a rank happens to hold."""
    channels, size = 2, 3
    pred = torch.zeros(batch, channels, size, size)
    target = torch.ones(batch, channels, size, size)
    wet = torch.ones(channels, size, size)
    sample_weight = torch.ones(batch, channels, size, size)
    scored = metric(pred, target, wet=wet, sample_weight=sample_weight)
    assert scored.tolist() == pytest.approx([1.0] * channels)


@pytest.mark.parametrize("metric", METRICS)
def test_an_all_wet_sample_weight_matches_no_sample_weight(metric) -> None:
    """Passing a mask that excludes nothing must not change the score."""
    batch, channels, size = 4, 2, 3
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(batch, channels, size, size, generator=generator)
    target = torch.randn(batch, channels, size, size, generator=generator)
    wet = torch.ones(channels, size, size)
    assert metric(
        pred, target, wet=wet, sample_weight=torch.ones(batch, channels, size, size)
    ).tolist() == pytest.approx(metric(pred, target, wet=wet).tolist())


def test_gradient_h_shares_the_fix() -> None:
    """gradient_h routes through the same mean, so it moved with it."""
    batch, channels, size = 4, 2, 5
    generator = torch.Generator().manual_seed(1)
    pred = torch.randn(batch, channels, size, size, generator=generator)
    target = torch.randn(batch, channels, size, size, generator=generator)
    wet = torch.ones(channels, size, size)
    with_weight = gradient_h_l1_loss(
        pred=pred,
        target=target,
        wet=wet,
        pad_mode="constant",
        sample_weight=torch.ones(batch, channels, size, size),
    )
    without = gradient_h_l1_loss(
        pred=pred, target=target, wet=wet, pad_mode="constant"
    )
    assert with_weight.tolist() == pytest.approx(without.tolist())


def test_a_per_sample_mask_still_excludes_the_cells_it_names() -> None:
    """The fix must not turn the mask into a no-op: a sample masked out
    entirely should not pull the mean toward its error."""
    channels, size = 1, 2
    pred = torch.zeros(2, channels, size, size)
    target = torch.zeros(2, channels, size, size)
    target[1] = 10.0  # the masked-out sample is wildly wrong
    wet = torch.ones(channels, size, size)
    sample_weight = torch.ones(2, channels, size, size)
    sample_weight[1] = 0.0
    assert decomposed_mse(
        pred, target, wet=wet, sample_weight=sample_weight
    ).tolist() == pytest.approx([0.0])


def test_an_all_dry_batch_scores_zero_rather_than_dividing_by_zero() -> None:
    """A face has tiles that are entirely land; they must contribute nothing
    and must not produce a NaN that poisons the whole step."""
    pred = torch.zeros(2, 1, 3, 3)
    target = torch.ones(2, 1, 3, 3)
    wet = torch.ones(1, 3, 3)
    scored = decomposed_mse_mae(
        pred, target, wet=wet, sample_weight=torch.zeros(2, 1, 3, 3)
    )
    assert torch.isfinite(scored).all()
    assert scored.tolist() == pytest.approx([0.0])


# --------------------------------------------------------------------------
# Channel weights
# --------------------------------------------------------------------------


def test_named_variables_are_weighted_and_the_rest_default_to_one() -> None:
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        names = tensor_map.prognostic_var_names
        weights = channel_weight_vector(
            {"Eta": 5.0}, device=CPU, num_channels=len(names)
        )
        assert weights[names.index("Eta")].item() == 5.0
        for channel in ("U_0", "V_50", "Theta_0", "Salt_25"):
            assert weights[names.index(channel)].item() == 1.0


def test_eta_is_one_channel_against_fifty_one_per_3d_variable() -> None:
    """The reason Eta needs weighting at all: state the imbalance explicitly so
    a change in the variable set shows up here."""
    with MultitonScope():
        names = TensorMap.init_instance("all", "all_fw_noeta").prognostic_var_names
        counts: dict[str, int] = {}
        for name in names:
            counts[name.split("_")[0]] = counts.get(name.split("_")[0], 0) + 1
        assert counts == {"U": 51, "V": 51, "Theta": 51, "Salt": 51, "Eta": 1}


def test_a_weight_naming_no_variable_is_rejected() -> None:
    """A silently ignored typo would leave the run unweighted and look fine."""
    with MultitonScope():
        names = TensorMap.init_instance("all", "all_fw_noeta").prognostic_var_names
        with pytest.raises(ValueError, match="not.*prognostic variables"):
            channel_weight_vector({"eta": 5.0}, device=CPU, num_channels=len(names))


def test_no_mapping_falls_back_to_the_historic_defaults() -> None:
    with MultitonScope():
        names = TensorMap.init_instance("all", "all_fw_noeta").prognostic_var_names
        weights = channel_weight_vector(None, device=CPU, num_channels=len(names))
        for channel, expected in (("U_0", 1.0), ("Theta_0", 1.5), ("Eta", 1.5)):
            assert weights[names.index(channel)].item() == expected
        assert DEFAULT_CHANNEL_WEIGHTS["Eta"] == 1.5


def test_the_wrapper_scales_the_loss_and_passes_the_sample_weight_through() -> None:
    """The grouped path always supplies a per-sample mask, so a wrapper that
    swallowed it would silently drop per-tile land masking."""
    with MultitonScope():
        names = TensorMap.init_instance("all", "all_fw_noeta").prognostic_var_names
        seen = {}

        def inner(pred, target, sample_weight=None):
            seen["sample_weight"] = sample_weight
            return torch.ones(len(names))

        weighted = WeightedLoss(
            loss_fn=inner, device=CPU, num_channels=len(names), weights={"Eta": 5.0}
        )
        mask = torch.ones(1, len(names), 2, 2)
        scored = weighted(torch.zeros(1, len(names), 2, 2), torch.zeros(1, len(names), 2, 2), sample_weight=mask)
        assert seen["sample_weight"] is mask
        assert scored[names.index("Eta")].item() == 5.0
        assert scored[names.index("U_0")].item() == 1.0
