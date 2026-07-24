# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit held-out reconstruction structure for a coarse learned inverse.

This intentionally reports metrics that the scalar training loss cannot
distinguish: channel errors, physical-gradient power, coarse-patch boundary
error, and (for the hybrid decoder) correction magnitude. It reuses the exact
training loader and checkpoint configuration so normalization and masks match
the run being audited.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from samudra.config import TrainConfig
from samudra.models.modules import ContinuousResampleAttentionResidualDecoder
from samudra.models.samudra_multi import SamudraMulti
from samudra.train import Trainer


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def _weighted_sum(
    value: torch.Tensor,
    mask: torch.Tensor,
    latitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latitude_weight = torch.cos(
        torch.deg2rad(latitude.to(device=value.device, dtype=value.dtype))
    )
    weight = mask.to(value.dtype) * latitude_weight[None, None, :, None]
    return (value * weight).sum(dim=(0, 2, 3)), weight.sum(dim=(0, 2, 3))


def _edge_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitude: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> dict[str, float]:
    lat_error = (
        (prediction[:, :, 1:] - prediction[:, :, :-1])
        - (target[:, :, 1:] - target[:, :, :-1])
    ).square()
    lat_mask = mask[:, :, 1:] & mask[:, :, :-1]
    lat_weight = torch.cos(
        torch.deg2rad(latitude[1:].to(device=prediction.device, dtype=prediction.dtype))
    )[None, None, :, None]
    lat_weight = lat_weight * lat_mask
    lat_boundary = (
        torch.arange(1, prediction.shape[-2], device=prediction.device) % patch_h == 0
    )

    rolled_prediction = prediction.roll(-1, dims=-1)
    rolled_target = target.roll(-1, dims=-1)
    lon_error = ((rolled_prediction - prediction) - (rolled_target - target)).square()
    lon_mask = mask & mask.roll(-1, dims=-1)
    lon_weight = torch.cos(
        torch.deg2rad(latitude.to(device=prediction.device, dtype=prediction.dtype))
    )[None, None, :, None]
    lon_weight = lon_weight * lon_mask
    lon_boundary = (
        torch.arange(1, prediction.shape[-1] + 1, device=prediction.device) % patch_w
        == 0
    )

    return {
        "edge_error_sum": float(
            (lat_error * lat_weight).sum() + (lon_error * lon_weight).sum()
        ),
        "edge_weight_sum": float(lat_weight.sum() + lon_weight.sum()),
        "seam_error_sum": float(
            (lat_error[:, :, lat_boundary] * lat_weight[:, :, lat_boundary]).sum()
            + (
                lon_error[:, :, :, lon_boundary] * lon_weight[:, :, :, lon_boundary]
            ).sum()
        ),
        "seam_weight_sum": float(
            lat_weight[:, :, lat_boundary].sum()
            + lon_weight[:, :, :, lon_boundary].sum()
        ),
        "prediction_gradient_sum": float(
            ((prediction[:, :, 1:] - prediction[:, :, :-1]).square() * lat_weight).sum()
            + ((rolled_prediction - prediction).square() * lon_weight).sum()
        ),
        "target_gradient_sum": float(
            ((target[:, :, 1:] - target[:, :, :-1]).square() * lat_weight).sum()
            + ((rolled_target - target).square() * lon_weight).sum()
        ),
    }


def _channel_labels(names: list[str], channels: int) -> list[str]:
    if channels % len(names):
        return [f"channel_{index}" for index in range(channels)]
    histories = channels // len(names)
    return [
        f"history_{history}/{name}" for history in range(histories) for name in names
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="samudra-coarse-audit-") as output_root:
        cfg = TrainConfig.from_yaml_and_cli(
            [
                str(args.config),
                f"--resume_ckpt_path={args.checkpoint}",
                "--experiment.name=audit",
                f"--experiment.base_output_dir={output_root}",
                f"--experiment.data_root={args.data_root}",
                f"--backend={args.backend}",
            ]
        )
        cfg.prepare_output_dirs()
        trainer = Trainer(cfg)
        trainer.init_data_loaders(1)
        trainer.val_loader.set_epoch(0)
        model = _unwrap(trainer.model)
        if not isinstance(model, SamudraMulti):
            raise TypeError("The coarse inverse audit requires SamudraMulti.")
        model.eval()
        encoder = _unwrap(model.encoder)
        patch_extent = getattr(encoder, "patch_extent", None)
        if patch_extent is None:
            raise TypeError("The audited encoder must expose a physical patch extent.")

        route_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "batches": 0,
                "sse": None,
                "weight": None,
                "edge_error_sum": 0.0,
                "edge_weight_sum": 0.0,
                "seam_error_sum": 0.0,
                "seam_weight_sum": 0.0,
                "prediction_gradient_sum": 0.0,
                "target_gradient_sum": 0.0,
                "correction_sum": 0.0,
                "prediction_sum": 0.0,
                "correction_weight": 0.0,
            }
        )

        decoder = _unwrap(model.decoder)
        try:
            with torch.no_grad():
                for batch_index, data in enumerate(trainer.val_loader):
                    if args.max_batches is not None and batch_index >= args.max_batches:
                        break
                    prediction = model(data)[0]
                    target = data.get_label(0)
                    mask = data.ctx.label_mask[None].expand_as(prediction)
                    latitude = data.ctx.output_resolution_cpu[0]
                    patch_h = round(patch_extent[0] * prediction.shape[-2] / 180.0)
                    patch_w = round(patch_extent[1] * prediction.shape[-1] / 360.0)
                    route = (
                        f"{len(data.ctx.input_resolution_cpu[0])}x"
                        f"{len(data.ctx.input_resolution_cpu[1])}->"
                        f"{prediction.shape[-2]}x{prediction.shape[-1]}"
                    )
                    stats = route_stats[route]
                    error_sum, weight = _weighted_sum(
                        (prediction - target).square(),
                        mask,
                        latitude,
                    )
                    stats["sse"] = (
                        error_sum.cpu()
                        if stats["sse"] is None
                        else stats["sse"] + error_sum.cpu()
                    )
                    stats["weight"] = (
                        weight.cpu()
                        if stats["weight"] is None
                        else stats["weight"] + weight.cpu()
                    )
                    for name, value in _edge_sums(
                        prediction,
                        target,
                        mask,
                        latitude,
                        patch_h,
                        patch_w,
                    ).items():
                        stats[name] += value

                    if isinstance(decoder, ContinuousResampleAttentionResidualDecoder):
                        prognostic, boundary = data.get_initial_input()
                        latent, latent_resolution = model.encode(
                            prognostic, boundary, data.ctx
                        )
                        correction = decoder.correction(
                            latent,
                            latent_resolution,
                            data.ctx.output_resolution_cpu,
                        )
                        correction_sum, correction_weight = _weighted_sum(
                            correction.square(),
                            mask,
                            latitude,
                        )
                        prediction_sum, _ = _weighted_sum(
                            prediction.square(),
                            mask,
                            latitude,
                        )
                        stats["correction_sum"] += float(correction_sum.sum())
                        stats["prediction_sum"] += float(prediction_sum.sum())
                        stats["correction_weight"] += float(correction_weight.sum())
                    stats["batches"] += 1
        finally:
            trainer.train_loader.close()
            trainer.val_loader.close()
            trainer.finish()

        labels = _channel_labels(
            list(trainer.prognostic_var_names),
            model.out_channels,
        )
        result: dict[str, Any] = {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "routes": {},
        }
        for route, stats in sorted(route_stats.items()):
            channel_mse = stats["sse"] / stats["weight"].clamp_min(1)
            worst = sorted(
                zip(labels, channel_mse.tolist(), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
            edge_mse = stats["edge_error_sum"] / stats["edge_weight_sum"]
            seam_mse = stats["seam_error_sum"] / stats["seam_weight_sum"]
            route_result = {
                "batches": stats["batches"],
                "channel_mean_mse": float(channel_mse.mean()),
                "channel_median_mse": float(channel_mse.median()),
                "worst_channels": [
                    {"channel": name, "mse": value} for name, value in worst
                ],
                "edge_gradient_mse": edge_mse,
                "patch_seam_gradient_mse": seam_mse,
                "seam_to_all_gradient_error_ratio": seam_mse / edge_mse,
                "prediction_to_target_gradient_power_ratio": (
                    stats["prediction_gradient_sum"] / stats["target_gradient_sum"]
                ),
            }
            if stats["correction_weight"]:
                route_result["correction_to_prediction_rms_ratio"] = (
                    stats["correction_sum"] / stats["prediction_sum"]
                ) ** 0.5
            result["routes"][route] = route_result
        print("AUDIT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
