# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Resident prepared frames for the small, fixed OM4 wave-one experiment.

The native loader remains the sole reader/normalizer. This cache stores its
float32 outputs once and reuses the canonical window plans without rereading
and decompressing the same histories every epoch.
"""

import numpy as np
import torch


class PreparedFrameCache:
    def __init__(self, frames, channels, boundary_channels, shape, device):
        self.prognostic = torch.empty(
            (frames, channels, *shape), device=device, dtype=torch.float32
        )
        self.boundary = torch.empty(
            (frames, boundary_channels, *shape), device=device, dtype=torch.float32
        )
        self.prognostic_ready = np.zeros(frames, dtype=bool)
        self.boundary_ready = np.zeros(frames, dtype=bool)

    @staticmethod
    def _write(storage, ready, indices, values):
        indices = np.asarray(indices)
        b, times = indices.shape
        planes = values.reshape(b * times, *storage.shape[1:])
        if values.dtype != storage.dtype:
            raise ValueError("Cache must preserve prepared-frame dtype")
        storage.index_copy_(
            0, torch.tensor(indices.reshape(-1), device=storage.device), planes
        )
        ready[indices.reshape(-1)] = True

    @staticmethod
    def _read(storage, ready, indices):
        indices = np.asarray(indices)
        if not ready[indices].all():
            raise ValueError("Requested frames were not populated by the native loader")
        b, times = indices.shape
        selected = storage.index_select(
            0, torch.tensor(indices.reshape(-1), device=storage.device)
        )
        return selected.reshape(b, times * storage.shape[1], *storage.shape[-2:])

    def record(self, plan, batch):
        for step, values in zip(plan.steps, batch, strict=True):
            for use, value, storage, ready in (
                (step.input, values[0], self.prognostic, self.prognostic_ready),
                (step.boundary, values[1], self.boundary, self.boundary_ready),
                (step.label, values[2], self.prognostic, self.prognostic_ready),
            ):
                self._write(storage, ready, use.request.time_indices, value)

    def batch(self, dataset, indices):
        result = dataset.preparer.new_model_batch(self.prognostic.device)
        for step in dataset.shard.window_plan(indices).steps:
            result.append(
                self._read(
                    self.prognostic,
                    self.prognostic_ready,
                    step.input.request.time_indices,
                ),
                self._read(
                    self.boundary,
                    self.boundary_ready,
                    step.boundary.request.time_indices,
                ),
                self._read(
                    self.prognostic,
                    self.prognostic_ready,
                    step.label.request.time_indices,
                ),
            )
        return result
