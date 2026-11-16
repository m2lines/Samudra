# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Masked operators that treat land as absent rather than as zero-valued ocean.

With ``land: omit`` (see :class:`samudra.config.SamudraConfig`), every spatial
operator in the U-Net sees only ocean cells:

* convolutions are partial convolutions (Liu et al., 2018,
  https://arxiv.org/abs/1804.07723): ``conv(x·m) · k²C/conv₁(m) + b``, then
  ``·m``, where ``conv₁`` sums the mask over the kernel's footprint. Hidden
  layers use one (column) mask for all channels, so ``C`` cancels; the first
  layer gives each input channel its own depth's mask;
* batch norm takes its mean and variance over ocean cells only;
* average pooling averages ocean cells only, and a coarse cell is ocean if any of
  its fine cells is;
* bilinear upsampling interpolates from ocean cells only.

Every operator returns exact zeros on land, so land never mixes into an ocean
cell. Padding is unchanged: circular in longitude, zeros past the poles. The
zero-padded mask past the poles reads as "absent", like land.

Masks are float tensors of shape ``(1, 1, H, W)`` holding 0 (land) and 1
(ocean), in the dtype of the activations they multiply.

The operators reuse the parameters and buffers of the ordinary ``nn`` modules
(``nn.Conv2d``, ``nn.BatchNorm2d``), so a checkpoint loads with either setting.
"""

from collections.abc import Callable
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

LandMode = Literal["mask", "omit"]


def pad_field(x: torch.Tensor, n_pad: int, pad_mode: str) -> torch.Tensor:
    """Pad as the U-Net does: ``pad_mode`` in longitude, zeros in latitude."""
    x = F.pad(x, (n_pad, n_pad, 0, 0), mode=pad_mode)
    return F.pad(x, (0, 0, n_pad, n_pad), mode="constant")


def coarsen_mask(mask: torch.Tensor, pooling: int) -> torch.Tensor:
    """A coarse cell is ocean if any of its fine cells is."""
    return F.max_pool2d(mask, pooling)


def _safe_reciprocal_times(numerator: float, count: torch.Tensor) -> torch.Tensor:
    """``numerator / count`` where ``count > 0``, else 0."""
    return torch.where(
        count > 0, numerator / count.clamp_min(1), torch.zeros_like(count)
    )


def _partial_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dilation: tuple[int, int],
    mask: torch.Tensor,
    input_mask: torch.Tensor | None,
    n_pad: int,
    pad_mode: str,
) -> torch.Tensor:
    mask = mask.to(x.dtype)
    in_mask = mask if input_mask is None else input_mask.to(x.dtype)
    c_mask = in_mask.shape[1]
    if c_mask not in (1, x.shape[1]):
        raise ValueError(
            f"input mask has {c_mask} channels; expected 1 or {x.shape[1]}"
        )
    _, _, kh, kw = weight.shape
    y = F.conv2d(
        pad_field(x * in_mask, n_pad, pad_mode), weight, None, dilation=dilation
    )
    footprint = kh * kw * c_mask
    if footprint > 1:
        # Count in float32 outside autocast: bfloat16 holds integers exactly
        # only up to 256, and the first layer's count reaches k²C ≈ 1,500.
        with torch.autocast(device_type=x.device.type, enabled=False):
            m32 = in_mask.float()
            ones = torch.ones(1, c_mask, kh, kw, device=m32.device)
            count = F.conv2d(
                pad_field(m32, n_pad, pad_mode), ones, None, dilation=dilation
            )
            ratio = _safe_reciprocal_times(float(footprint), count)
        y = y * ratio.to(y.dtype)
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1).to(y.dtype)
    return y * mask


def partial_conv2d(
    x: torch.Tensor,
    conv: nn.Conv2d,
    mask: torch.Tensor,
    n_pad: int,
    pad_mode: str,
    input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``conv``'s weights as a partial convolution over ocean cells.

    ``x`` is unpadded; this pads ``x`` and the masks the same way. An output
    cell is the convolution over its ocean inputs, rescaled by the ratio of the
    kernel's footprint (``k²·C``) to the number of ocean inputs inside it, plus
    the bias. Output cells off ``mask`` (``(1, 1, H, W)``) are zero.

    ``input_mask`` (``(1, C_in, H, W)``) gives each input channel its own mask,
    as at the network's input, where a deep level has land where the surface is
    ocean. It defaults to ``mask``.
    """
    if conv.groups != 1 or conv.stride != (1, 1) or conv.padding != (0, 0):
        raise ValueError(
            "partial_conv2d supports only unpadded, stride-1, ungrouped convolutions"
        )
    return _partial_conv(
        x,
        conv.weight,
        conv.bias,
        (conv.dilation[0], conv.dilation[1]),
        mask,
        input_mask,
        n_pad,
        pad_mode,
    )


def partial_pointwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    """A 1×1 partial convolution with a per-channel ``input_mask``: mix only the
    channels that are ocean at each cell, rescaled by ``C / (ocean channels)``.

    ``weight`` is ``(C_out, C_in)`` or ``(C_out, C_in, 1, 1)``.
    """
    if weight.dim() == 2:
        weight = weight[:, :, None, None]
    return _partial_conv(x, weight, bias, (1, 1), mask, input_mask, 0, "constant")


def _all_reduce_sum(t: torch.Tensor, group) -> torch.Tensor:
    # The autograd-aware collective, so gradients flow through the statistics
    # on every rank, as they do in nn.SyncBatchNorm.
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(t, op=dist.ReduceOp.SUM, group=group)


def masked_batch_norm(
    x: torch.Tensor, bn: _BatchNorm, mask: torch.Tensor
) -> torch.Tensor:
    """Batch norm with statistics over ocean cells only, using ``bn``'s state.

    Follows ``nn.BatchNorm2d`` exactly, except that the reductions skip land:
    the biased batch variance normalizes, the unbiased one updates the running
    variance, and ``momentum=None`` means a cumulative average. For an
    ``nn.SyncBatchNorm`` in training, the sums are all-reduced over its process
    group, so the statistics cover the ocean cells of the global batch.
    """
    mask = mask.to(x.dtype)
    # Compute in at least float32, even under bfloat16 autocast, as torch's
    # batch norm does.
    compute_dtype = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(compute_dtype)
    use_batch_stats = bn.training or not bn.track_running_stats
    if use_batch_stats:
        mf = mask.to(compute_dtype)
        n = (mf.sum() * x.shape[0]).reshape(1)
        s1 = (xf * mf).sum(dim=(0, 2, 3))
        sync = (
            isinstance(bn, nn.SyncBatchNorm)
            and bn.training
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(bn.process_group) > 1
        )
        if sync:
            n = _all_reduce_sum(n, bn.process_group)
            s1 = _all_reduce_sum(s1, bn.process_group)
        # Clamp rather than check that n > 1: a check would sync the device on
        # every call. An all-land grid then normalizes to zeros.
        n = n.clamp_min(1)
        mean = s1 / n
        centered = (xf - mean.view(1, -1, 1, 1)) * mf
        s2 = (centered * centered).sum(dim=(0, 2, 3))
        if sync:
            s2 = _all_reduce_sum(s2, bn.process_group)
        var = s2 / n
        if bn.training and bn.track_running_stats:
            assert bn.num_batches_tracked is not None
            assert bn.running_mean is not None and bn.running_var is not None
            with torch.no_grad():
                bn.num_batches_tracked.add_(1)
                if bn.momentum is None:
                    factor = 1.0 / float(bn.num_batches_tracked)
                else:
                    factor = bn.momentum
                unbiased = var * (n / (n - 1).clamp_min(1))
                bn.running_mean.mul_(1 - factor).add_(mean.detach(), alpha=factor)
                bn.running_var.mul_(1 - factor).add_(unbiased.detach(), alpha=factor)
    else:
        assert bn.running_mean is not None and bn.running_var is not None
        mean = bn.running_mean
        var = bn.running_var

    inv_std = torch.rsqrt(var + bn.eps)
    scale = inv_std if bn.weight is None else inv_std * bn.weight
    shift = -mean * scale
    if bn.bias is not None:
        shift = shift + bn.bias
    y = xf * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    return y.to(x.dtype) * mask


def masked_avg_pool2d(
    x: torch.Tensor, mask: torch.Tensor, pooling: int
) -> torch.Tensor:
    """Average over the ocean cells of each window; land windows are zero."""
    mask = mask.to(x.dtype)
    num = F.avg_pool2d(x * mask, pooling)
    den = F.avg_pool2d(mask, pooling)
    return torch.where(den > 0, num / den.clamp_min(torch.finfo(den.dtype).tiny), 0.0)


def masked_max_pool2d(
    x: torch.Tensor, mask: torch.Tensor, pooling: int
) -> torch.Tensor:
    """Maximum over the ocean cells of each window; land windows are zero."""
    pooled = F.max_pool2d(x.masked_fill(mask == 0, float("-inf")), pooling)
    coarse = coarsen_mask(mask, pooling)
    return torch.where(coarse > 0, pooled, 0.0)


def masked_upsample(
    x: torch.Tensor,
    mask: torch.Tensor,
    upsample: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Interpolate from coarse ocean cells only: ``up(x·m) / up(m)``.

    The caller multiplies the result by the fine mask, after any padding to
    the skip connection's shape.
    """
    mask = mask.to(x.dtype)
    num = upsample(x * mask)
    den = upsample(mask)
    return torch.where(den > 0, num / den.clamp_min(torch.finfo(den.dtype).tiny), 0.0)
