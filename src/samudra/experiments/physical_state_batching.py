# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Group real examples without padding or changing per-example objectives."""

from collections import defaultdict

import torch


def compatible_groups(samples, microbatch):
    """Return original indices grouped by trajectory length, at most microbatch."""
    if microbatch < 1:
        raise ValueError("Microbatch must be positive")
    buckets = defaultdict(list)
    for index, sample in enumerate(samples):
        key = "fields" if "fields" in sample else "surface"
        buckets[sample[key].shape[1]].append(index)
    return [
        indices[start : start + microbatch]
        for indices in buckets.values()
        for start in range(0, len(indices), microbatch)
    ]


def collate_examples(samples):
    """Stack per-example weights; retain common native masks without batch axis.

    Loss callers must score each example independently before averaging. In
    particular, monthly weights and missing-observation denominators differ.
    """
    if not samples or any(sample.keys() != samples[0].keys() for sample in samples):
        raise ValueError("Examples must be nonempty with matching keys")
    result = {}
    for key, first in samples[0].items():
        values = [sample[key] for sample in samples]
        if key == "mask":
            if any(not torch.equal(first, value) for value in values):
                raise ValueError("Native wet masks differ")
            result[key] = first
        elif isinstance(first, torch.Tensor):
            if key in ("month_weights", "weights"):
                result[key] = torch.stack(values)
            elif first.ndim and first.shape[0] == 1:
                result[key] = torch.cat(values, dim=0)
            else:
                raise ValueError(f"Unknown tensor batching convention: {key}")
        else:
            result[key] = values
    return result
