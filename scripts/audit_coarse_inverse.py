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
import torch.nn.functional as F

from samudra.config import TrainConfig
from samudra.datasets import TrainDataLoader
from samudra.models.modules import ContinuousResampleAttentionResidualDecoder
from samudra.models.samudra_multi import SamudraMulti
from samudra.rust_data import RustTrainDataLoader
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


def _coarse_valid_mask(
    mask: torch.Tensor,
    coarse_shape: tuple[int, int],
) -> torch.Tensor:
    """Mark coarse cells containing at least one wet physical cell."""
    surface_wet = mask.any(dim=0, keepdim=True).unsqueeze(0).float()
    return F.adaptive_max_pool2d(surface_wet, coarse_shape).bool()


def _coarse_patch_means(
    value: torch.Tensor,
    mask: torch.Tensor,
    latitude: torch.Tensor,
    coarse_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-wise, area-weighted means on a common coarse grid."""
    batch, channels, height, width = value.shape
    coarse_h, coarse_w = coarse_shape
    if height % coarse_h or width % coarse_w:
        raise ValueError(
            f"Grid {(height, width)} does not divide into coarse grid {coarse_shape}."
        )
    patch_h = height // coarse_h
    patch_w = width // coarse_w
    weight = (
        mask[None].to(device=value.device, dtype=value.dtype)
        * torch.cos(torch.deg2rad(latitude.to(device=value.device, dtype=value.dtype)))[
            None, None, :, None
        ]
    )
    weight = weight.expand(batch, channels, height, width)
    weighted_value = (value * weight).reshape(
        batch,
        channels,
        coarse_h,
        patch_h,
        coarse_w,
        patch_w,
    )
    coarse_weight = weight.reshape(
        batch,
        channels,
        coarse_h,
        patch_h,
        coarse_w,
        patch_w,
    ).sum(dim=(3, 5))
    coarse_sum = weighted_value.sum(dim=(3, 5))
    return coarse_sum / coarse_weight.clamp_min(1), coarse_weight > 0


def _prepared_dataset_item(loader: Any, dataset: Any, index: int) -> Any:
    """Read one deterministic item from a specific validation route."""
    if isinstance(loader, TrainDataLoader):
        raw = dataset[index]
        collate_fn = loader._dataloader.collate_fn
        if collate_fn is not None:
            raw = collate_fn([raw])
        return dataset.to_train_data(raw, loader._device)
    if isinstance(loader, RustTrainDataLoader):
        raw = dataset.load_chunk_batch([index], buffer_pool=None)
        return loader._prepare_batch((dataset, raw))
    raise TypeError(f"Unsupported validation loader: {type(loader).__name__}")


def _agreement_metrics(
    trainer: Trainer,
    model: SamudraMulti,
    *,
    max_batches: int | None,
) -> dict[str, Any]:
    """Compare synchronized encodings and renderings across physical grids."""
    loader = trainer.val_loader
    datasets: list[Any]
    shards: list[Any]
    if isinstance(loader, TrainDataLoader):
        datasets = list(loader._datasets.values())
        shards = [dataset.shard for dataset in datasets]
    elif isinstance(loader, RustTrainDataLoader):
        datasets = list(loader._batch_datasets)
        shards = [dataset.shard for dataset in datasets]
    else:
        raise TypeError(f"Unsupported validation loader: {type(loader).__name__}")
    by_route = {
        (
            shard.prognostic_srcs[0].grid_size,
            shard.prognostic_srcs[-1].grid_size,
        ): dataset
        for dataset, shard in zip(datasets, shards, strict=True)
    }
    grids = sorted({route[0] for route in by_route})
    if len(grids) != 2 or any(
        (input_grid, output_grid) not in by_route
        for input_grid in grids
        for output_grid in grids
    ):
        raise ValueError(
            "Cross-resolution agreement requires the complete product of exactly "
            f"two input/output grids; found {sorted(by_route)}."
        )

    low_grid, high_grid = grids
    low_dataset = by_route[(low_grid, low_grid)]
    high_dataset = by_route[(high_grid, high_grid)]
    batches = min(len(low_dataset.shard), len(high_dataset.shard))
    if max_batches is not None:
        batches = min(batches, max_batches)

    latent_sse = 0.0
    latent_scale = 0.0
    latent_count = 0
    latent_cosine_sum = 0.0
    latent_token_count = 0
    output_stats = {
        input_grid: {"sse": 0.0, "scale": 0.0, "count": 0} for input_grid in grids
    }

    with torch.no_grad():
        for index in range(batches):
            prepared = {
                route: _prepared_dataset_item(loader, dataset, index)
                for route, dataset in by_route.items()
            }
            low_data = prepared[(low_grid, low_grid)]
            high_data = prepared[(high_grid, high_grid)]
            low_input, low_boundary = low_data.get_initial_input()
            high_input, high_boundary = high_data.get_initial_input()
            low_latent, low_resolution = model.encode(
                low_input, low_boundary, low_data.ctx
            )
            high_latent, high_resolution = model.encode(
                high_input, high_boundary, high_data.ctx
            )
            if low_latent.shape != high_latent.shape:
                raise ValueError(
                    "Synchronized encoders must produce the same latent shape; got "
                    f"{tuple(low_latent.shape)} and {tuple(high_latent.shape)}."
                )
            coarse_shape = (low_latent.shape[-2], low_latent.shape[-1])
            low_valid = _coarse_valid_mask(low_data.ctx.input_mask, coarse_shape).to(
                low_latent.device
            )
            high_valid = _coarse_valid_mask(high_data.ctx.input_mask, coarse_shape).to(
                high_latent.device
            )
            latent_valid = low_valid & high_valid
            expanded_valid = latent_valid.expand_as(low_latent)
            low_float = low_latent.float()
            high_float = high_latent.float()
            difference = low_float - high_float
            latent_sse += float(difference.square()[expanded_valid].sum())
            latent_scale += float(
                (0.5 * (low_float.square() + high_float.square()))[expanded_valid].sum()
            )
            latent_count += int(expanded_valid.sum())
            cosine = F.cosine_similarity(low_float, high_float, dim=1)
            token_valid = latent_valid[:, 0]
            latent_cosine_sum += float(cosine[token_valid].sum())
            latent_token_count += int(token_valid.sum())

            for input_grid, latent, latent_resolution in (
                (low_grid, low_latent, low_resolution),
                (high_grid, high_latent, high_resolution),
            ):
                low_output_data = prepared[(input_grid, low_grid)]
                high_output_data = prepared[(input_grid, high_grid)]
                low_prediction = model.decode(
                    latent, latent_resolution, low_output_data.ctx
                )
                high_prediction = model.decode(
                    latent, latent_resolution, high_output_data.ctx
                )
                low_mean, low_mean_valid = _coarse_patch_means(
                    low_prediction,
                    low_output_data.ctx.label_mask,
                    low_output_data.ctx.output_resolution_cpu[0],
                    coarse_shape,
                )
                high_mean, high_mean_valid = _coarse_patch_means(
                    high_prediction,
                    high_output_data.ctx.label_mask,
                    high_output_data.ctx.output_resolution_cpu[0],
                    coarse_shape,
                )
                mean_valid = low_mean_valid & high_mean_valid
                mean_difference = low_mean.float() - high_mean.float()
                stats = output_stats[input_grid]
                stats["sse"] += float(mean_difference.square()[mean_valid].sum())
                stats["scale"] += float(
                    (0.5 * (low_mean.float().square() + high_mean.float().square()))[
                        mean_valid
                    ].sum()
                )
                stats["count"] += int(mean_valid.sum())

    return {
        "batches": batches,
        "latent_agreement": {
            "input_grids": [f"{height}x{width}" for height, width in grids],
            "mean_squared_difference": latent_sse / max(latent_count, 1),
            "symmetric_normalized_mse": latent_sse / max(latent_scale, 1e-12),
            "mean_token_cosine_similarity": (
                latent_cosine_sum / max(latent_token_count, 1)
            ),
        },
        "output_patch_mean_consistency": {
            f"{height}x{width}": {
                "mean_squared_difference": stats["sse"] / max(stats["count"], 1),
                "symmetric_normalized_mse": (stats["sse"] / max(stats["scale"], 1e-12)),
            }
            for (height, width), stats in output_stats.items()
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--agreement-batches", type=int, default=32)
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

        agreement = _agreement_metrics(
            trainer,
            model,
            max_batches=args.agreement_batches,
        )
        labels = _channel_labels(
            list(trainer.prognostic_var_names),
            model.out_channels,
        )
        result: dict[str, Any] = {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "routes": {},
            "cross_resolution": agreement,
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
