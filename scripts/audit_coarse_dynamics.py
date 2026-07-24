# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit latent structure and invariants of a coarse-dynamics checkpoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from samudra.config import TrainConfig
from samudra.models.modules.encoder import PatchMomentEncoder
from samudra.models.samudra_multi import SamudraMulti
from samudra.stepper import ablate_boundary_forcing
from samudra.train import Trainer
from samudra.utils.ctx import GridContext
from samudra.utils.device import autocast
from scripts.audit_coarse_inverse import (
    _coarse_patch_means,
    _coarse_valid_mask,
    _prepared_dataset_item,
    _unwrap,
)

DEPTHS = (1, 2, 4)


def _grid_name(grid: tuple[int, int]) -> str:
    return f"{grid[0]}x{grid[1]}"


def _empty_pair_stats() -> dict[str, float | int]:
    return {
        "sse": 0.0,
        "scale": 0.0,
        "count": 0,
        "cosine_sum": 0.0,
        "token_count": 0,
    }


def _update_pair_stats(
    stats: dict[str, float | int],
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Accumulate symmetric error and token cosine over a common wet mask."""
    if left.shape != right.shape:
        raise ValueError(
            f"Compared latent shapes differ: {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if valid.shape[0] not in (1, left.shape[0]) or valid.shape[1] not in (
        1,
        left.shape[1],
    ):
        raise ValueError(
            f"Mask {tuple(valid.shape)} cannot broadcast to {tuple(left.shape)}."
        )
    expanded = valid.expand_as(left)
    left_float = left.float()
    right_float = right.float()
    difference = left_float - right_float
    stats["sse"] += float(difference.square()[expanded].sum())
    stats["scale"] += float(
        (0.5 * (left_float.square() + right_float.square()))[expanded].sum()
    )
    stats["count"] += int(expanded.sum())

    # A token is valid for cosine only when every compared feature is valid.
    token_valid = expanded.all(dim=1)
    cosine = torch.nn.functional.cosine_similarity(left_float, right_float, dim=1)
    stats["cosine_sum"] += float(cosine[token_valid].sum())
    stats["token_count"] += int(token_valid.sum())


def _finish_pair_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    count = int(stats["count"])
    token_count = int(stats["token_count"])
    sse = float(stats["sse"])
    scale = float(stats["scale"])
    return {
        "values": count,
        "tokens": token_count,
        "mean_squared_difference": sse / max(count, 1),
        "symmetric_normalized_mse": sse / max(scale, 1e-12),
        "mean_token_cosine_similarity": (
            float(stats["cosine_sum"]) / max(token_count, 1)
        ),
    }


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


def _checkpoint_model_state(path: Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    return OrderedDict(
        (name.removeprefix("module."), value.detach().cpu())
        for name, value in state.items()
    )


def _inverse_preservation(
    model: SamudraMulti,
    inverse_checkpoint: Path,
) -> dict[str, Any]:
    reference = _checkpoint_model_state(inverse_checkpoint)
    current = OrderedDict(
        (name, value.detach().cpu()) for name, value in model.state_dict().items()
    )
    prefixes = ("encoder.", "decoder.")
    reference_keys = {key for key in reference if key.startswith(prefixes)}
    current_keys = {key for key in current if key.startswith(prefixes)}
    shared = sorted(reference_keys & current_keys)
    maximum = max(
        (float((reference[key] - current[key]).abs().max()) for key in shared),
        default=0.0,
    )
    return {
        "reference_checkpoint": str(inverse_checkpoint.resolve()),
        "shared_tensors": len(shared),
        "missing_from_dynamics": sorted(reference_keys - current_keys),
        "missing_from_inverse": sorted(current_keys - reference_keys),
        "maximum_absolute_difference": maximum,
        "exact": (
            reference_keys == current_keys
            and maximum == 0.0
            and all(torch.equal(reference[key], current[key]) for key in shared)
        ),
    }


def _target_latent(
    model: SamudraMulti,
    data: Any,
    depth: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    target_ctx = GridContext(
        label_mask=data.ctx.label_mask,
        input_resolution_cpu=data.ctx.output_resolution_cpu,
        output_resolution_cpu=data.ctx.output_resolution_cpu,
        input_mask=data.ctx.label_mask,
    )
    with autocast(enabled=model.use_bfloat16, dtype=torch.bfloat16):
        return model.encode(data.get_label(depth - 1), None, target_ctx)


def _rollout_from_state(
    model: SamudraMulti,
    data: Any,
    initial: torch.Tensor,
    latent_resolution: tuple[torch.Tensor, torch.Tensor],
    depths: tuple[int, ...] = DEPTHS,
) -> dict[int, torch.Tensor]:
    """Advance a supplied latent state with the batch's aligned boundaries."""
    selected = set(depths)
    state = initial
    states: dict[int, torch.Tensor] = {}
    with autocast(enabled=model.use_bfloat16, dtype=torch.bfloat16):
        for step in range(max(depths)):
            _, boundary = data.get_input(step)
            state = model.process(
                state,
                latent_resolution,
                iterations=1,
                boundary=boundary,
                boundary_resolution=data.ctx.input_resolution_cpu,
            )
            depth = step + 1
            if depth in selected:
                states[depth] = state
    return states


def _route_datasets(
    trainer: Trainer,
) -> dict[tuple[tuple[int, int], tuple[int, int]], Any]:
    loader: Any = trainer.val_loader
    if hasattr(loader, "_datasets"):
        datasets = list(loader._datasets.values())
        shards = [dataset.shard for dataset in datasets]
    elif hasattr(loader, "_batch_datasets"):
        datasets = list(loader._batch_datasets)
        shards = [dataset.shard for dataset in datasets]
    else:
        raise TypeError(f"Unsupported validation loader: {type(loader).__name__}")
    return {
        (
            shard.prognostic_srcs[0].grid_size,
            shard.prognostic_srcs[-1].grid_size,
        ): dataset
        for dataset, shard in zip(datasets, shards, strict=True)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--inverse-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max-batches", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="samudra-coarse-dynamics-audit-") as root:
        cfg = TrainConfig.from_yaml_and_cli(
            [
                str(args.config),
                f"--resume_ckpt_path={args.checkpoint}",
                "--experiment.name=audit",
                f"--experiment.base_output_dir={root}",
                f"--experiment.data_root={args.data_root}",
                "--experiment.wandb.mode=disabled",
                f"--backend={args.backend}",
            ]
        )
        cfg.prepare_output_dirs()
        trainer = Trainer(cfg)
        trainer.init_data_loaders(max(DEPTHS))
        trainer.val_loader.set_epoch(0)
        model = _unwrap(trainer.model)
        if not isinstance(model, SamudraMulti):
            raise TypeError("The coarse dynamics audit requires SamudraMulti.")
        if model.processor_residual_scale is None:
            raise TypeError("The audited dynamics model must use a residual processor.")
        encoder = _unwrap(model.encoder)
        if not isinstance(encoder, PatchMomentEncoder):
            raise TypeError("The dynamics audit requires a PatchMomentEncoder.")
        model.eval()

        by_route = _route_datasets(trainer)
        grids = sorted({route[0] for route in by_route})
        if len(grids) != 2 or any(
            (input_grid, output_grid) not in by_route
            for input_grid in grids
            for output_grid in grids
        ):
            raise ValueError(
                "Dynamics audit requires the complete product of two grids; "
                f"found {sorted(by_route)}."
            )
        low_grid, high_grid = grids
        batches = min(
            args.max_batches,
            len(by_route[(low_grid, low_grid)].shard),
            len(by_route[(high_grid, high_grid)].shard),
        )

        agreement = {depth: _empty_pair_stats() for depth in (0, *DEPTHS)}
        teacher = {
            grid: {depth: _empty_pair_stats() for depth in DEPTHS} for grid in grids
        }
        forcing = {
            grid: {depth: _empty_pair_stats() for depth in DEPTHS} for grid in grids
        }
        cross_output = {
            grid: {depth: _empty_pair_stats() for depth in DEPTHS} for grid in grids
        }
        physical_forecast = {
            grid: {
                depth: {
                    variant: _empty_pair_stats()
                    for variant in ("full", "persistence", "mean_only_initial")
                }
                for depth in DEPTHS
            }
            for grid in grids
        }
        moment_ablation_effect = {
            grid: {depth: _empty_pair_stats() for depth in DEPTHS} for grid in grids
        }
        zero_depth_reconstruction = {
            grid: {
                variant: _empty_pair_stats()
                for variant in ("full", "mean_only_initial")
            }
            for grid in grids
        }

        try:
            with torch.no_grad(), trainer._test_context():
                inverse_preservation = _inverse_preservation(
                    model, args.inverse_checkpoint
                )
                residual_scale = _parameter_summary(model.processor_residual_scale)
                for index in range(batches):
                    prepared = {
                        route: _prepared_dataset_item(
                            trainer.val_loader, dataset, index
                        )
                        for route, dataset in by_route.items()
                    }
                    states_by_grid: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
                    resolutions: dict[
                        tuple[int, int], tuple[torch.Tensor, torch.Tensor]
                    ] = {}
                    for grid in grids:
                        data = prepared[(grid, grid)]
                        prognostic, boundary = data.get_initial_input()
                        with autocast(enabled=model.use_bfloat16, dtype=torch.bfloat16):
                            initial, initial_resolution = model.encode(
                                prognostic, boundary, data.ctx
                            )
                        states, latent_resolution = model.latent_rollout(
                            data, list(DEPTHS)
                        )
                        zero_states, _ = model.latent_rollout(
                            ablate_boundary_forcing(data, "zero"), list(DEPTHS)
                        )
                        mean_only_initial = initial.clone()
                        mean_only_initial[:, encoder.mean_channels :] = 0
                        mean_only_states = _rollout_from_state(
                            model,
                            data,
                            mean_only_initial,
                            latent_resolution,
                        )
                        states_by_grid[grid] = {0: initial, **states}
                        resolutions[grid] = latent_resolution
                        if any(
                            not torch.equal(left, right)
                            for left, right in zip(
                                initial_resolution,
                                latent_resolution,
                                strict=True,
                            )
                        ):
                            raise ValueError(
                                "Initial and forecast latent grids do not match."
                            )

                        if data.ctx.input_mask is None:
                            raise ValueError(
                                "Dynamics audit requires channel-aware input masks."
                            )
                        valid = _coarse_valid_mask(
                            data.ctx.input_mask,
                            (initial.shape[-2], initial.shape[-1]),
                        ).to(initial.device)
                        input_valid = data.ctx.input_mask[None].to(
                            device=initial.device,
                            dtype=torch.bool,
                        )
                        initial_physical = prognostic[:, -model.decoder.out_channels :]
                        with autocast(
                            enabled=model.use_bfloat16,
                            dtype=torch.bfloat16,
                        ):
                            reconstructed = model.decode(
                                initial,
                                latent_resolution,
                                data.ctx,
                            )
                            mean_only_reconstructed = model.decode(
                                mean_only_initial,
                                latent_resolution,
                                data.ctx,
                            )
                        _update_pair_stats(
                            zero_depth_reconstruction[grid]["full"],
                            reconstructed,
                            initial_physical,
                            input_valid,
                        )
                        _update_pair_stats(
                            zero_depth_reconstruction[grid]["mean_only_initial"],
                            mean_only_reconstructed,
                            initial_physical,
                            input_valid,
                        )
                        for depth in DEPTHS:
                            target, target_resolution = _target_latent(
                                model, data, depth
                            )
                            if any(
                                not torch.allclose(left, right)
                                for left, right in zip(
                                    latent_resolution,
                                    target_resolution,
                                    strict=True,
                                )
                            ):
                                raise ValueError(
                                    "Forecast and target latent grids do not match."
                                )
                            _update_pair_stats(
                                teacher[grid][depth],
                                states[depth],
                                target,
                                valid,
                            )
                            _update_pair_stats(
                                forcing[grid][depth],
                                states[depth],
                                zero_states[depth],
                                valid,
                            )
                            with autocast(
                                enabled=model.use_bfloat16,
                                dtype=torch.bfloat16,
                            ):
                                prediction = model.decode(
                                    states[depth],
                                    latent_resolution,
                                    data.ctx,
                                )
                                mean_only_prediction = model.decode(
                                    mean_only_states[depth],
                                    latent_resolution,
                                    data.ctx,
                                )
                            target = data.get_label(depth - 1)
                            label_valid = data.ctx.label_mask[None].to(
                                device=prediction.device,
                                dtype=torch.bool,
                            )
                            _update_pair_stats(
                                physical_forecast[grid][depth]["full"],
                                prediction,
                                target,
                                label_valid,
                            )
                            _update_pair_stats(
                                physical_forecast[grid][depth]["persistence"],
                                initial_physical,
                                target,
                                label_valid,
                            )
                            _update_pair_stats(
                                physical_forecast[grid][depth]["mean_only_initial"],
                                mean_only_prediction,
                                target,
                                label_valid,
                            )
                            _update_pair_stats(
                                moment_ablation_effect[grid][depth],
                                prediction,
                                mean_only_prediction,
                                label_valid,
                            )

                    low_mask = prepared[(low_grid, low_grid)].ctx.input_mask
                    high_mask = prepared[(high_grid, high_grid)].ctx.input_mask
                    if low_mask is None or high_mask is None:
                        raise ValueError(
                            "Dynamics audit requires channel-aware input masks."
                        )
                    low_valid = _coarse_valid_mask(
                        low_mask,
                        (
                            states_by_grid[low_grid][0].shape[-2],
                            states_by_grid[low_grid][0].shape[-1],
                        ),
                    ).to(states_by_grid[low_grid][0].device)
                    high_valid = _coarse_valid_mask(
                        high_mask,
                        (
                            states_by_grid[high_grid][0].shape[-2],
                            states_by_grid[high_grid][0].shape[-1],
                        ),
                    ).to(states_by_grid[high_grid][0].device)
                    for depth in (0, *DEPTHS):
                        _update_pair_stats(
                            agreement[depth],
                            states_by_grid[low_grid][depth],
                            states_by_grid[high_grid][depth],
                            low_valid & high_valid,
                        )

                    for input_grid in grids:
                        latent_resolution = resolutions[input_grid]
                        for depth in DEPTHS:
                            latent = states_by_grid[input_grid][depth]
                            low_data = prepared[(input_grid, low_grid)]
                            high_data = prepared[(input_grid, high_grid)]
                            with autocast(
                                enabled=model.use_bfloat16, dtype=torch.bfloat16
                            ):
                                low_prediction = model.decode(
                                    latent, latent_resolution, low_data.ctx
                                )
                                high_prediction = model.decode(
                                    latent, latent_resolution, high_data.ctx
                                )
                            coarse_shape = (latent.shape[-2], latent.shape[-1])
                            low_mean, low_mean_valid = _coarse_patch_means(
                                low_prediction,
                                low_data.ctx.label_mask,
                                low_data.ctx.output_resolution_cpu[0],
                                coarse_shape,
                            )
                            high_mean, high_mean_valid = _coarse_patch_means(
                                high_prediction,
                                high_data.ctx.label_mask,
                                high_data.ctx.output_resolution_cpu[0],
                                coarse_shape,
                            )
                            _update_pair_stats(
                                cross_output[input_grid][depth],
                                low_mean,
                                high_mean,
                                low_mean_valid & high_mean_valid,
                            )
        finally:
            trainer.train_loader.close()
            trainer.val_loader.close()
            trainer.finish()

        result = {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "batches": batches,
            "inverse_preservation": inverse_preservation,
            "processor_residual_scale": residual_scale,
            "latent_resolution_agreement": {
                f"depth_{depth}": _finish_pair_stats(stats)
                for depth, stats in agreement.items()
            },
            "teacher_latent_error": {
                _grid_name(grid): {
                    f"depth_{depth}": _finish_pair_stats(stats)
                    for depth, stats in by_depth.items()
                }
                for grid, by_depth in teacher.items()
            },
            "zero_boundary_latent_effect": {
                _grid_name(grid): {
                    f"depth_{depth}": _finish_pair_stats(stats)
                    for depth, stats in by_depth.items()
                }
                for grid, by_depth in forcing.items()
            },
            "cross_output_patch_mean_consistency": {
                _grid_name(grid): {
                    f"depth_{depth}": _finish_pair_stats(stats)
                    for depth, stats in by_depth.items()
                }
                for grid, by_depth in cross_output.items()
            },
            "physical_forecast": {
                _grid_name(grid): {
                    f"depth_{depth}": {
                        variant: _finish_pair_stats(stats)
                        for variant, stats in by_variant.items()
                    }
                    for depth, by_variant in by_depth.items()
                }
                for grid, by_depth in physical_forecast.items()
            },
            "moment_ablation_effect": {
                _grid_name(grid): {
                    f"depth_{depth}": _finish_pair_stats(stats)
                    for depth, stats in by_depth.items()
                }
                for grid, by_depth in moment_ablation_effect.items()
            },
            "zero_depth_reconstruction": {
                _grid_name(grid): {
                    variant: _finish_pair_stats(stats)
                    for variant, stats in by_variant.items()
                }
                for grid, by_variant in zero_depth_reconstruction.items()
            },
        }
        print("DYNAMICS_AUDIT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
