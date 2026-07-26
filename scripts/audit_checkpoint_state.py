# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit checkpoint metadata, frozen inverse tensors, and residual scales."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

FROZEN_PREFIXES = ("encoder.", "decoder.")
METADATA_KEYS = (
    "epoch",
    "num_batches_seen",
    "num_optimizer_updates",
    "num_samples_seen",
    "best_inf_loss",
    "best_val_loss",
    "wandb_id",
    "wandb_name",
    "num_batches",
    "num_optimizer_steps",
    "num_samples",
    "best_validation_loss",
    "best_validation_error",
    "wandb_run_id",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inverse_checkpoint", type=Path)
    parser.add_argument("--mean-channels", type=int, default=40)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping; got {type(checkpoint)}.")
    return checkpoint


def _model_state(checkpoint: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a mapping named `model`.")
    return OrderedDict(
        (name.removeprefix("module."), value.detach().cpu())
        for name, value in state.items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    )


def _parameter_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().float().cpu().flatten()
    absolute = flat.abs()
    return {
        "shape": list(value.shape),
        "count": flat.numel(),
        "mean": float(flat.mean()),
        "standard_deviation": float(flat.std(unbiased=False)),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean_absolute": float(absolute.mean()),
        "median_absolute": float(absolute.median()),
        "absolute_quantiles": {
            str(quantile): float(torch.quantile(absolute, quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "negative_fraction": float((flat < 0).float().mean()),
        "near_zero_fraction": float((absolute < 1e-6).float().mean()),
        "values": flat.tolist(),
    }


def _inverse_preservation(
    current: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    reference_keys = {key for key in reference if key.startswith(FROZEN_PREFIXES)}
    current_keys = {key for key in current if key.startswith(FROZEN_PREFIXES)}
    shared = sorted(reference_keys & current_keys)
    maximum = max(
        (float((reference[key] - current[key]).abs().max()) for key in shared),
        default=0.0,
    )
    return {
        "shared_tensors": len(shared),
        "shared_parameters": sum(reference[key].numel() for key in shared),
        "missing_from_dynamics": sorted(reference_keys - current_keys),
        "missing_from_inverse": sorted(current_keys - reference_keys),
        "maximum_absolute_difference": maximum,
        "exact": (
            reference_keys == current_keys
            and maximum == 0.0
            and all(torch.equal(reference[key], current[key]) for key in shared)
        ),
    }


def _metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in METADATA_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            selected[key] = value
    return {
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "selected": selected,
    }


def main() -> None:
    args = _parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    inverse_checkpoint = _load_checkpoint(args.inverse_checkpoint)
    current = _model_state(checkpoint)
    reference = _model_state(inverse_checkpoint)

    scale_keys = [key for key in current if key.endswith("processor_residual_scale")]
    if len(scale_keys) != 1:
        raise ValueError(
            f"Expected exactly one processor residual scale; found {scale_keys}."
        )
    scale = current[scale_keys[0]]
    flat_scale = scale.flatten()
    if not 0 < args.mean_channels < flat_scale.numel():
        raise ValueError(
            f"mean-channels must lie in (0, {flat_scale.numel()}); "
            f"got {args.mean_channels}."
        )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "inverse_checkpoint": str(args.inverse_checkpoint.resolve()),
        "metadata": _metadata(checkpoint),
        "inverse_preservation": _inverse_preservation(current, reference),
        "processor_residual_scale_key": scale_keys[0],
        "processor_residual_scale": _parameter_summary(scale),
        "processor_residual_scale_groups": {
            "resolved_mean": _parameter_summary(flat_scale[: args.mean_channels]),
            "subpatch_moment": _parameter_summary(flat_scale[args.mean_channels :]),
        },
    }
    print("CHECKPOINT_STATE_AUDIT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
