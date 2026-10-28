# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Masked losses and deterministic artificial gaps for surface initialization."""

import torch


def structured_visibility(validity, seed):
    """Persistent blocks plus an optional latitude band; no global RNG changes.

    validity is [batch, time, variable, y, x]. Artificial hiding changes only
    inputs. Keep the original validity for labels, including in naturally missing
    regions. Longitude blocks wrap, latitude bands do not.
    """
    if validity.ndim != 5 or validity.shape[2] != 2:
        raise ValueError("Expected two surface variables with batch/time dimensions")
    batch, _, channels, h, w = validity.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    hidden = torch.zeros((batch, 1, channels, h, w), dtype=torch.bool)
    for b in range(batch):
        for c in range(channels):
            for _ in range(3):
                height = max(
                    1, int(h * (0.08 + 0.12 * torch.rand((), generator=generator)))
                )
                width = max(
                    1, int(w * (0.12 + 0.20 * torch.rand((), generator=generator)))
                )
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                x = int(torch.randint(w, (), generator=generator))
                columns = (torch.arange(width) + x) % w
                hidden[b, 0, c, y : y + height, columns] = True
            if bool(torch.rand((), generator=generator) < 0.5):
                height = max(1, h // 12)
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                hidden[b, 0, c, y : y + height] = True
    return validity.bool() & ~hidden.to(validity.device)


def corrupt_sample(sample, seed):
    """Return separate inputs/targets; never mutate cached or supplied labels."""
    result = dict(sample)
    visible = structured_visibility(sample["validity"], seed)
    result["surface"] = torch.where(visible, sample["surface"], 0)
    result["validity"] = visible
    result["completion_target"] = sample["surface"][:, 17:19]
    result["completion_valid"] = (
        sample["validity"][:, 17:19].bool() & ~visible[:, 17:19]
    )
    return result


def completion_loss(prediction, target, valid, latitude):
    """Mean normalized MSE by field/frame, full-grid wet observed support only."""
    valid = valid.bool()
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Completion predictions, labels and support must align")
    if not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite completion label on declared valid support")
    area = torch.cos(torch.deg2rad(torch.as_tensor(latitude, device=prediction.device)))
    weight = valid * area[None, None, None, :, None]
    denominator = weight.sum((-2, -1))
    error = torch.where(valid, prediction.float() - target, 0).square()
    channel = (error * weight).sum((-2, -1)) / denominator.clamp_min(1e-12)
    present = denominator > 0
    if not present.any():
        return prediction.sum() * 0
    return channel[present].mean()


def channel_mse(prediction, target, weights):
    """Area-weighted errors per sample/lead/channel, excluding dry cells."""
    error = torch.where(weights > 0, prediction.float() - target.float(), 0).square()
    return (error * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-12)


def balanced_loss(prediction, target, weights, names, interior_only=False):
    per_channel = channel_mse(prediction, target, weights)
    groups = []
    for variable in ("thetao", "so", "uo", "vo", "zos"):
        indices = [
            i
            for i, name in enumerate(names)
            if (name == variable or name.startswith(variable + "_"))
            and not (interior_only and name in ("thetao_0", "zos"))
        ]
        if indices:
            groups.append(per_channel[..., indices].mean())
    if not groups:
        raise ValueError("No supported physical fields in the requested loss")
    return torch.stack(groups).mean()
