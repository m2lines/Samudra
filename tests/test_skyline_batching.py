# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from samudra.skyline.batching import device_batches


@pytest.mark.parametrize(
    "physical,prefetch", [(32, False), (64, True), (128, False), (256, True)]
)
def test_grouping_preserves_exact_draw_stream_and_gradients(physical, prefetch):
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)
    draws = []

    def draw():
        # Coverage's pool and accepted draws must occur separately per logical microbatch.
        pool = rng.normal(size=8192)
        values = pool[rng.integers(0, len(pool), size=32)]
        draws.append(values)
        return {"target": torch.tensor(values), "importance": torch.ones(32)}

    expected = []
    for _ in range(8):
        pool = reference_rng.normal(size=8192)
        expected.extend(pool[reference_rng.integers(0, len(pool), size=32)])
    values = []
    parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    with device_batches(draw, 32, 8, physical, torch.device("cpu"), prefetch) as (
        get,
        count,
    ):
        for _ in range(count):
            batch = get()
            values.extend(batch["target"].tolist())
            ((parameter - batch["target"]).square().mean() / count).backward()
    assert np.array_equal(values, expected)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert len(draws) == 8
    torch.testing.assert_close(
        parameter.grad, 2 * (parameter.detach() - torch.tensor(expected).mean())
    )


def test_invalid_grouping_fails_before_sampling():
    with pytest.raises(ValueError):
        with device_batches(lambda: None, 32, 8, 96, torch.device("cpu")):
            pass
