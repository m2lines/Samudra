# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Differentiable projection from model cells to native IAP observation cells.

Keep IAP labels on their native horizontal grid. This operator averages physical
model predictions over each observation cell's wet support; it does not create
quarter-degree interior labels by interpolating the observations.
"""

import numpy as np
import torch

from samudra.experiments.observation_prepare import ConservativeRemap


def _sparse_tensor(matrix):
    matrix = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.stack((matrix.row, matrix.col)), dtype=torch.int64),
        torch.as_tensor(matrix.data, dtype=torch.float32),
        matrix.shape,
    ).coalesce()


class NativeIapOperator(torch.nn.Module):
    """Map ``[..., channel, model_y, model_x]`` to native observation cells.

    ``wet_mask`` has shape ``[channel, model_y, model_x]`` and must match the
    selected physical channels. The returned NaNs mark cells below the fixed
    wet-area threshold. Intersect this support with finite native labels in
    both losses and metrics; never reinterpret those NaNs as zero targets.

    Geometry follows the existing observation preparation's center-derived
    spherical cell bounds, including periodic longitude. Call on float32
    physical predictions after time averaging and before loss normalization.
    """

    wet_mask: torch.Tensor
    latitude_weights: torch.Tensor
    longitude_weights: torch.Tensor
    coverage: torch.Tensor
    supported: torch.Tensor
    area: torch.Tensor

    def __init__(
        self, model_lat, model_lon, iap_lat, iap_lon, wet_mask, min_coverage=0.9
    ):
        super().__init__()
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must lie in (0, 1]")
        remap = ConservativeRemap(model_lat, model_lon, iap_lat, iap_lon)
        mask = np.asarray(wet_mask)
        if mask.ndim != 3 or mask.shape[-2:] != remap.source_shape:
            raise ValueError("wet_mask must have shape [channel, model_y, model_x]")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("wet_mask must be boolean or binary")
        self.source_shape = remap.source_shape
        self.target_shape = remap.shape
        self.register_buffer("wet_mask", torch.as_tensor(mask, dtype=torch.bool))
        # Normalize each axis separately, keeping weights near unity.
        y = remap.y.multiply(1 / np.asarray(remap.y.sum(1)))
        x = remap.x.multiply(1 / np.asarray(remap.x.sum(1)))
        self.register_buffer("latitude_weights", _sparse_tensor(y))
        self.register_buffer("longitude_weights", _sparse_tensor(x))
        coverage = np.stack(
            [remap.integral(field.astype(np.float64)) / remap.area for field in mask]
        ).astype(np.float32)
        self.register_buffer("coverage", torch.from_numpy(coverage))
        self.register_buffer(
            "supported", torch.from_numpy(coverage >= min_coverage - 1e-7)
        )
        # Constant scale is irrelevant to normalized area-weighted losses.
        self.register_buffer("area", torch.from_numpy(remap.area.astype(np.float32)))

    def forward(self, physical):
        if physical.ndim < 3 or physical.shape[-3:] != self.wet_mask.shape:
            raise ValueError("Prediction channel/grid shape differs from wet_mask")
        if physical.dtype != torch.float32:
            raise ValueError("IAP projection requires physical float32 predictions")
        if not torch.isfinite(physical.masked_select(self.wet_mask)).all():
            raise ValueError("Nonfinite model prediction on wet support")
        leading = physical.shape[:-2]
        safe = torch.where(self.wet_mask, physical, 0).reshape(-1, *self.source_shape)
        count, _, width = safe.shape
        # Losses must retain float32 precision inside a bf16 training context.
        with torch.autocast(device_type=physical.device.type, enabled=False):
            latitude = torch.sparse.mm(
                self.latitude_weights,
                safe.permute(1, 0, 2).reshape(self.source_shape[0], -1),
            ).reshape(self.target_shape[0], count, width)
            longitude = torch.sparse.mm(
                self.longitude_weights,
                latitude.permute(2, 1, 0).reshape(width, -1),
            )
        integral = longitude.reshape(self.target_shape[1], count, self.target_shape[0])
        integral = integral.permute(1, 2, 0).reshape(*leading, *self.target_shape)
        mean = integral / self.coverage.clamp_min(1e-12)
        return torch.where(self.supported, mean, torch.nan)
