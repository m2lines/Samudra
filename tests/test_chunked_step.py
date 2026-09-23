"""Splitting a microbatch into chunks must not quietly change the objective.

A face-sized replay row is more tiles than a GPU holds, so the step runs in
chunks and accumulates gradients. Chunk losses are recombined by sample-count
share, which is exact only while the tiles share a land mask.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, build_loss_fn
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.loss import (
    decomposed_mse,
    decomposed_mse_mae,
    gradient_z_norms,
    weighted_channel_denominator,
)
from ocean_emulators.utils.multiton import MultitonScope

BATCH, CHANNELS, SIZE = 9, 2, 4
SPANS = [(0, 3), (3, 6), (6, 9)]


def fields():
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(BATCH, CHANNELS, SIZE, SIZE, generator=generator),
        torch.randn(BATCH, CHANNELS, SIZE, SIZE, generator=generator),
        torch.ones(CHANNELS, SIZE, SIZE),
    )


def recombine(metric, pred, target, wet, sample_weight):
    """What `_replay_forward_backward` accumulates: sum of share x chunk mean."""
    total = torch.zeros(CHANNELS)
    for start, stop in SPANS:
        chunk = metric(
            pred[start:stop],
            target[start:stop],
            wet=wet,
            sample_weight=None if sample_weight is None else sample_weight[start:stop],
        )
        total = total + chunk * ((stop - start) / BATCH)
    return total


@pytest.mark.parametrize("metric", [decomposed_mse, decomposed_mse_mae])
def test_chunking_is_exact_when_the_tiles_share_a_mask(metric) -> None:
    """The single-tile and uniform-mask cases -- every run before this one."""
    pred, target, wet = fields()
    for sample_weight in (None, torch.ones(BATCH, CHANNELS, SIZE, SIZE)):
        whole = metric(pred, target, wet=wet, sample_weight=sample_weight)
        assert recombine(metric, pred, target, wet, sample_weight).tolist() == (
            pytest.approx(whole.tolist(), rel=1e-6)
        )


def test_an_uneven_split_is_still_exact_for_a_shared_mask() -> None:
    """Chunks need not be equal: 9 tiles at 4 per chunk splits 3/3/3, but 5 at
    2 splits 2/2/1."""
    pred, target, wet = fields()
    whole = decomposed_mse(pred, target, wet=wet)
    total = torch.zeros(CHANNELS)
    for start, stop in [(0, 4), (4, 7), (7, 9)]:
        chunk = decomposed_mse(pred[start:stop], target[start:stop], wet=wet)
        total = total + chunk * ((stop - start) / BATCH)
    assert total.tolist() == pytest.approx(whole.tolist(), rel=1e-6)


def test_a_fixed_denominator_makes_chunking_exact_with_different_land() -> None:
    """The face case: two of its 36 tiles are entirely land.

    A denominator derived from the batch makes each chunk a mean over its own
    cells, and means do not add. One derived from the masks up front makes each
    chunk a share of the same mean, and shares do.
    """
    pred, target, wet = fields()
    sample_weight = torch.ones(BATCH, CHANNELS, SIZE, SIZE)
    sample_weight[6:] = 0.0  # an all-land tile, as face 1 has two of
    sample_weight[3:6, :, :2] = 0.0  # and a half-land one

    whole = decomposed_mse(pred, target, wet=wet, sample_weight=sample_weight)
    denominator = weighted_channel_denominator(
        wet=wet, batch=BATCH, extra_weight=sample_weight
    )
    total = torch.zeros(CHANNELS)
    for start, stop in SPANS:
        total = total + decomposed_mse(
            pred[start:stop],
            target[start:stop],
            wet=wet,
            sample_weight=sample_weight[start:stop],
            denominator=denominator,
        )
    assert total.tolist() == pytest.approx(whole.tolist(), rel=1e-6)


def test_without_a_fixed_denominator_chunking_is_not_exact() -> None:
    """Why the denominator exists. If this ever starts passing, the recombine
    path has changed and the fixed denominator may no longer be needed."""
    pred, target, wet = fields()
    sample_weight = torch.ones(BATCH, CHANNELS, SIZE, SIZE)
    sample_weight[6:] = 0.0
    sample_weight[3:6, :, :2] = 0.0
    whole = decomposed_mse(pred, target, wet=wet, sample_weight=sample_weight)
    chunked = recombine(decomposed_mse, pred, target, wet, sample_weight)
    assert ((chunked - whole).abs() / whole).max().item() > 0.01


@pytest.mark.parametrize("spans", [SPANS, [(0, 4), (4, 7), (7, 9)], [(0, 9)]])
def test_the_whole_loss_stack_is_chunk_summable(spans) -> None:
    """Not just the base metric: mse_mae + gradient_h + gradient_z + channel
    weights, which is the configuration a face run actually trains with.
    `gradient_z` is the hard one -- it averages per-pair ratios, so both its
    pair denominators and its pair counts have to be held fixed."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        channels = len(tensor_map.prognostic_var_names)
        size = 8
        generator = torch.Generator().manual_seed(0)
        pred = torch.randn(BATCH, channels, size, size, generator=generator)
        target = torch.randn(BATCH, channels, size, size, generator=generator)
        wet = torch.ones(channels, size, size)
        sample_weight = torch.ones(BATCH, channels, size, size)
        sample_weight[6:] = 0.0
        sample_weight[3:6, :, :4] = 0.0
        y_coord = xr.DataArray(np.linspace(-60.0, 60.0, size), dims="lat")
        config = GradientLossConfig(
            type=["gradient_h", "gradient_z"],
            metric="mse_mae",
            lambda_h=0.1,
            lambda_z=0.1,
            channel_weights={"Eta": 5.0},
        )

        whole = build_loss_fn(
            config, wet, y_coord, torch.device("cpu"), channels, "constant"
        )(pred, target, sample_weight=sample_weight)

        piece = build_loss_fn(
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
        total = sum(
            piece(pred[a:b], target[a:b], sample_weight=sample_weight[a:b])
            for a, b in spans
        )
        assert ((total - whole).abs().max() / whole.abs().max()).item() < 1e-6


def test_a_pair_that_is_dry_in_one_chunk_is_still_counted() -> None:
    """`gradient_z` drops depth pairs with no wet cell. Deriving that per
    chunk would drop a pair from one chunk's average and keep it in another's,
    so the pair counts have to come from the whole domain too."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
        channels = len(tensor_map.prognostic_var_names)
        size = 4
        wet = torch.ones(channels, size, size)
        sample_weight = torch.ones(BATCH, channels, size, size)
        sample_weight[3:] = 0.0  # only the first three samples are wet at all

        whole = gradient_z_norms(wet=wet, batch=BATCH, sample_weight=sample_weight)
        starved = gradient_z_norms(
            wet=wet, batch=3, sample_weight=sample_weight[3:6]
        )
        assert float(whole.count_by_time.sum()) > 0
        assert float(starved.count_by_time.sum()) == 0
