# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed, mask-aware fine-scale penalty added to the ordinary fixed MSE."""

import torch
from torch.nn import functional as F

from samudra.utils.loss import loss_fn_from_metric


def masked_highpass(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gaussian sigma1/radius3; periodic longitude, replicated latitude.

    Each variable/depth has its own wet support. Filter the signed error rather
    than matching power alone. All arithmetic is float32 even under autocast.
    """
    with torch.autocast(device_type=values.device.type, enabled=False):
        values = values.float()
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask[None]
        if mask.shape[1:] != values.shape[1:]:
            raise ValueError("Band mask/channel mismatch")
        support = mask.expand(values.shape[0], -1, -1, -1).float()
        wet = torch.where(support.bool(), values, 0.0)
        channels = values.shape[1]
        axis = torch.arange(-3, 4, device=values.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * axis.square())
        kernel = kernel / kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(
            channels, 1, 7, 7
        )

        def pad(x):
            columns = torch.arange(-3, x.shape[-1] + 3, device=x.device) % x.shape[-1]
            return F.pad(x.index_select(-1, columns), (0, 0, 3, 3), mode="replicate")

        denominator = F.conv2d(pad(support), kernel, groups=channels).clamp_min(1e-12)
        smooth = F.conv2d(pad(wet), kernel, groups=channels) / denominator
        return torch.where(support.bool(), wet - smooth, 0.0)


class BandLoss:
    """Ordinary normalized MSE plus a fixed training-calibrated band penalty.

    Full-grid means match the historical MSE convention. No online estimates,
    dynamic updates, reference-forecast weights, or trainable loss parameters.
    """

    def __init__(self, *, device, num_channels, band_scales, band_weight):
        self.loss_fn = loss_fn_from_metric("mse")
        scales = torch.as_tensor(band_scales, dtype=torch.float32, device=device)
        if (
            scales.shape != (num_channels,)
            or not torch.isfinite(scales).all()
            or not (scales > 0).all()
        ):
            raise ValueError(
                "Expected one positive finite band energy per prognostic channel"
            )
        if band_weight <= 0:
            raise ValueError("Band weight must be positive")
        self.band_scales = scales
        self.band_weight = float(band_weight)

    def __call__(self, pred, target, ctx):
        base = self.loss_fn(pred, target, ctx)
        highpass = masked_highpass(pred.float() - target.float(), ctx.label_mask)
        band = highpass.square().mean((0, 2, 3))
        levels = band.numel() // self.band_scales.numel()
        if levels * self.band_scales.numel() != band.numel():
            raise ValueError("Time/channel mismatch")
        return base + self.band_weight * band / self.band_scales.repeat(levels)

    def state_dict(self):
        return dict(
            band_scales=self.band_scales.detach().cpu(),
            band_weight=torch.tensor(self.band_weight),
        )

    def load_state_dict(self, state):
        if "band_scales" not in state or "band_weight" not in state:
            raise ValueError("Missing band-loss checkpoint state")
        if not torch.equal(state["band_scales"].cpu(), self.band_scales.cpu()):
            raise ValueError("Band calibration changed on resume")
        if abs(float(state["band_weight"]) - self.band_weight) > 1e-7:
            raise ValueError("Band coefficient changed on resume")
