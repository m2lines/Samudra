# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from samudra.experiments.physical_state_batching import (
    collate_examples,
    compatible_groups,
)


def test_group_preserves_every_example_without_padding():
    samples = [{"surface": torch.zeros(1, n, 2, 2, 2)} for n in (25, 26, 25, 26, 25)]
    groups = compatible_groups(samples, 2)
    assert groups == [[0, 2], [4], [1, 3]]
    assert sorted(sum(groups, [])) == list(range(5))


def test_collation_retains_distinct_calendar_weights():
    samples = [
        {
            "surface": torch.full((1, 2, 2, 2, 2), i),
            "month_weights": torch.tensor(w),
            "name": str(i),
        }
        for i, w in enumerate(([0.4, 0.6], [0.5, 0.5]))
    ]
    batch = collate_examples(samples)
    assert batch["surface"].shape[0] == 2
    assert batch["month_weights"].shape == (2, 2)
    assert not torch.equal(batch["month_weights"][0], batch["month_weights"][1])
    assert batch["name"] == ["0", "1"]


def test_different_native_masks_rejected():
    with pytest.raises(ValueError, match="masks differ"):
        collate_examples([{"mask": torch.zeros(2)}, {"mask": torch.ones(2)}])
