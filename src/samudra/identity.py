# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-sample identity-reconstruction diagnostic for SamudraMulti."""

import json
import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from pydantic import Field

from samudra.aggregator.loss import (
    get_channel_loss_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
from samudra.aggregator.validate.spatial import (
    high_wavenumber_power_ratio,
    patch_seam_jump_ratio,
    zonal_power_spectrum,
)
from samudra.config import SamudraMultiConfig, TrainConfig
from samudra.constants import Lat, Lon
from samudra.datasets import TrainData
from samudra.models.modules.decoder import coordinate_bilinear_resample
from samudra.models.samudra_multi import SamudraMulti
from samudra.stepper import train_batch
from samudra.train import Trainer
from samudra.utils.distributed import (
    all_reduce_mean,
    destroy_distributed_mode,
    get_world_size,
    is_main_process,
)
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


class IdentityConfig(TrainConfig):
    """Training configuration plus fixed-sample identity controls."""

    identity_train_samples: int = Field(default=32, ge=1)
    identity_eval_samples: int = Field(default=32, ge=1)
    identity_train_offset: int = Field(default=0, ge=0)
    identity_eval_offset: int = Field(default=32, ge=0)
    identity_eval_frequency: int = Field(default=1, ge=1)
    identity_eval_only: bool = Field(
        default=False,
        description="Load a finetune checkpoint and evaluate fixed routes without "
        "running backward or updating model parameters.",
    )
    identity_eval_processor_depths: list[int] | None = Field(
        default=None,
        description="Optional processor iteration counts evaluated on the same "
        "checkpoint and held-out reconstruction samples. The configured training "
        "depth is always included.",
    )


@contextmanager
def _processor_depth(trainer: Trainer, depth: int) -> Iterator[None]:
    """Temporarily select a SamudraMulti processor depth for checkpoint evaluation."""
    if depth < 0:
        raise ValueError("Identity evaluation processor depths must be non-negative.")
    model = getattr(trainer.model, "module", trainer.model)
    if not isinstance(model, SamudraMulti):
        raise TypeError("Processor-depth identity evaluation requires SamudraMulti.")
    previous_depth = model.processor_iterations
    model.processor_iterations = depth
    try:
        yield
    finally:
        model.processor_iterations = previous_depth


def set_identity_target(data: TrainData) -> torch.Tensor:
    """Replace the one-step label with the prognostic input and return it."""
    if len(data) != 1:
        raise ValueError("Identity reconstruction requires exactly one model step.")
    prognostic, boundary, label = data[0]
    if prognostic.shape != label.shape:
        raise ValueError(
            "Identity target requires matching prognostic and output shapes; "
            f"got {tuple(prognostic.shape)} and {tuple(label.shape)}."
        )
    data.example_by_step[0] = (prognostic, boundary, prognostic)
    return prognostic


def _route_key(data: TrainData) -> str:
    """Return a stable spatial-shape key for one paired reconstruction route."""
    input_lat, input_lon = data.ctx.input_resolution_cpu
    output_lat, output_lon = data.ctx.output_resolution_cpu
    return f"{len(input_lat)}x{len(input_lon)}_to_{len(output_lat)}x{len(output_lon)}"


def _target_for_mode(data: TrainData, target_time_mode: str) -> torch.Tensor:
    if target_time_mode == "forecast":
        return set_identity_target(data)
    if target_time_mode != "current":
        raise ValueError(f"Unsupported identity target time mode {target_time_mode!r}")
    if len(data) != 1:
        raise ValueError("Identity reconstruction requires exactly one model step.")
    return data.get_label(0)


def _fixed_batches(
    trainer: Trainer,
    requested_samples: int,
    *,
    sample_offset: int,
    route_count: int = 1,
    route_filter: str | None = None,
):
    """Yield an exact deterministic range from the validation stream.

    Offsets and sample counts refer to the global stream and must align with a
    global loader batch. This keeps every distributed rank on the same number of
    batches without slicing ``TrainData`` objects differently across ranks.
    """
    if requested_samples % route_count or sample_offset % route_count:
        raise ValueError(
            "identity sample counts and offsets must divide evenly across "
            f"{route_count} routes"
        )
    requested_per_route = requested_samples // route_count
    offset_per_route = sample_offset // route_count
    trainer.val_loader.set_epoch(0)
    skipped_samples = 0
    selected_samples = 0
    skipped_by_route: dict[str, int] = {}
    selected_by_route: dict[str, int] = {}
    for data in trainer.val_loader:
        route = _route_key(data)
        if route_filter is not None and route != route_filter:
            continue
        batch_samples = data.get_label(0).shape[0] * get_world_size()
        route_skipped = skipped_by_route.get(route, 0)
        if route_skipped < offset_per_route:
            next_skipped = route_skipped + batch_samples
            if next_skipped > offset_per_route:
                raise ValueError(
                    "identity sample offsets must align with global loader batches; "
                    f"per-route offset {offset_per_route} falls inside a {route} "
                    f"batch ending at sample {next_skipped}."
                )
            skipped_by_route[route] = next_skipped
            skipped_samples += batch_samples
            continue

        route_selected = selected_by_route.get(route, 0)
        if route_selected == requested_per_route:
            continue
        next_selected = route_selected + batch_samples
        if next_selected > requested_per_route:
            raise ValueError(
                "identity sample counts must align with global loader batches; "
                f"requested {requested_per_route} samples on {route} but the next "
                f"batch would select {next_selected}."
            )
        selected_by_route[route] = next_selected
        selected_samples += batch_samples
        yield data
        if selected_samples == requested_samples:
            break

    if (
        len(selected_by_route) != route_count
        or any(value != requested_per_route for value in selected_by_route.values())
        or selected_samples < requested_samples
    ):
        raise ValueError(
            "The validation stream is too short for the requested identity sample "
            f"range [{sample_offset}, {sample_offset + requested_samples}); selected "
            f"{selected_samples} samples over {len(selected_by_route)} routes after "
            f"skipping {skipped_samples}."
        )


def _group_values(
    prefix: str, values: torch.Tensor, trainer: Trainer
) -> dict[str, float]:
    logs: dict[str, float] = {}
    for index, channel in enumerate(trainer.tensor_map.prognostic_var_names):
        logs[f"{prefix}/channel/{channel}"] = float(values[index].cpu())
    for variable, indices in trainer.tensor_map.VAR_3D_IDX.items():
        logs[f"{prefix}/variable/{variable}"] = float(
            values[indices.long()].mean().cpu()
        )
    for depth, indices in trainer.tensor_map.DP_3D_IDX.items():
        logs[f"{prefix}/depth/{depth}"] = float(values[indices.long()].mean().cpu())
    return logs


def _channel_stats_and_mask(
    trainer: Trainer, grid_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prognostic means, scales, and wet masks for one configured grid."""
    matching = [
        source
        for source in trainer.data_container.sources
        if source.grid_size == grid_shape
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one source for grid {grid_shape}, found {len(matching)}."
        )
    selected = matching[0].select_channels(
        trainer.tensor_map.prognostic_var_names, prefix="identity_output_stats"
    )
    return (
        torch.as_tensor(selected.statistics.mean, device=trainer.device),
        torch.as_tensor(selected.statistics.std, device=trainer.device),
        matching[0].masks.prognostic.to(trainer.device),
    )


def _masked_physical_resampler_reference(
    input_physical: torch.Tensor,
    input_wet: torch.Tensor,
    input_resolution: tuple[Lat, Lon],
    output_resolution: tuple[Lat, Lon],
    output_means: torch.Tensor,
) -> torch.Tensor:
    """Resample wet values and use destination climatology without source support."""
    resampled = coordinate_bilinear_resample(
        input_physical,
        input_resolution,
        output_resolution,
        valid_mask=input_wet,
    )
    support = coordinate_bilinear_resample(
        input_wet.to(dtype=input_physical.dtype).unsqueeze(0),
        input_resolution,
        output_resolution,
    )
    return torch.where(
        support > 0,
        resampled,
        output_means[None, :, None, None],
    )


@torch.no_grad()
def evaluate_identity(
    trainer: Trainer,
    requested_samples: int,
    patch_size: tuple[int, int],
    *,
    sample_offset: int,
    prefix: str,
    target_time_mode: str = "forecast",
    route_filter: str | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Evaluate fixed-sample identity MSE, spectra, and patch seams."""
    trainer.model.eval()
    loss_sum: torch.Tensor | None = None
    target_spectrum_sum: torch.Tensor | None = None
    generated_spectrum_sum: torch.Tensor | None = None
    seam_ratio_sum: torch.Tensor | None = None
    target_value_sum: torch.Tensor | None = None
    target_square_sum: torch.Tensor | None = None
    generated_value_sum: torch.Tensor | None = None
    generated_square_sum: torch.Tensor | None = None
    valid_value_count: torch.Tensor | None = None
    physical_square_error_sum: torch.Tensor | None = None
    physical_valid_count: torch.Tensor | None = None
    resampler_square_error_sum: torch.Tensor | None = None
    resampler_physical_square_error_sum: torch.Tensor | None = None
    source_normalized_resampler_square_error_sum: torch.Tensor | None = None
    local_samples = 0
    base_channels = len(trainer.tensor_map.prognostic_var_names)

    for data in _fixed_batches(
        trainer,
        requested_samples,
        sample_offset=sample_offset,
        route_filter=route_filter,
    ):
        target = _target_for_mode(data, target_time_mode)
        generated = trainer.model(data)[0]
        loss_per_channel = trainer.loss_fn(generated, target, data.ctx)
        batch_size = target.shape[0]
        target_latest = target[:, -base_channels:]
        generated_latest = generated[:, -base_channels:]
        wet = data.ctx.label_mask[-base_channels:].to(
            device=target.device, dtype=target.dtype
        )
        batch_valid_count = wet.sum(dim=(-2, -1)) * batch_size
        batch_target_sum = (target_latest * wet).sum(dim=(0, 2, 3))
        batch_target_square_sum = (target_latest.square() * wet).sum(dim=(0, 2, 3))
        batch_generated_sum = (generated_latest * wet).sum(dim=(0, 2, 3))
        batch_generated_square_sum = (generated_latest.square() * wet).sum(
            dim=(0, 2, 3)
        )
        input_shape = (
            len(data.ctx.input_resolution_cpu[0]),
            len(data.ctx.input_resolution_cpu[1]),
        )
        output_shape = (
            len(data.ctx.output_resolution_cpu[0]),
            len(data.ctx.output_resolution_cpu[1]),
        )
        input_means, input_stds, input_wet = _channel_stats_and_mask(
            trainer, input_shape
        )
        output_means, output_stds, _ = _channel_stats_and_mask(trainer, output_shape)
        input_means = input_means.to(dtype=target.dtype)
        input_stds = input_stds.to(dtype=target.dtype)
        output_means = output_means.to(dtype=target.dtype)
        output_stds = output_stds.to(dtype=target.dtype)
        batch_physical_square_error = (
            (generated_latest - target_latest).square()
            * output_stds[:, None, None].square()
            * wet
        ).sum(dim=(0, 2, 3))
        input_latest = data.get_initial_input()[0][:, -base_channels:]
        input_physical = (
            input_latest * input_stds[:, None, None] + input_means[:, None, None]
        )
        resampled_physical = _masked_physical_resampler_reference(
            input_physical,
            input_wet,
            data.ctx.input_resolution_cpu,
            data.ctx.output_resolution_cpu,
            output_means,
        )
        target_physical = (
            target_latest * output_stds[:, None, None] + output_means[:, None, None]
        )
        resampled_normalized = (
            resampled_physical - output_means[:, None, None]
        ) / output_stds[:, None, None]
        batch_resampler_square_error = (
            (resampled_normalized - target_latest).square() * wet
        ).sum(dim=(0, 2, 3))
        batch_resampler_physical_square_error = (
            (resampled_physical - target_physical).square() * wet
        ).sum(dim=(0, 2, 3))
        resampled_source_normalized = _masked_physical_resampler_reference(
            input_latest,
            input_wet,
            data.ctx.input_resolution_cpu,
            data.ctx.output_resolution_cpu,
            torch.zeros_like(output_means),
        )
        batch_source_normalized_resampler_square_error = (
            (resampled_source_normalized - target_latest).square() * wet
        ).sum(dim=(0, 2, 3))

        batch_target_spectrum = zonal_power_spectrum(target_latest) * batch_size
        batch_generated_spectrum = zonal_power_spectrum(generated_latest) * batch_size
        batch_seam_ratio = (
            patch_seam_jump_ratio(generated_latest - target_latest, patch_size)
            * batch_size
        )
        if loss_sum is None:
            loss_sum = loss_per_channel * batch_size
            target_spectrum_sum = batch_target_spectrum
            generated_spectrum_sum = batch_generated_spectrum
            seam_ratio_sum = batch_seam_ratio
            target_value_sum = batch_target_sum
            target_square_sum = batch_target_square_sum
            generated_value_sum = batch_generated_sum
            generated_square_sum = batch_generated_square_sum
            valid_value_count = batch_valid_count
            physical_square_error_sum = batch_physical_square_error
            physical_valid_count = batch_valid_count
            resampler_square_error_sum = batch_resampler_square_error
            resampler_physical_square_error_sum = batch_resampler_physical_square_error
            source_normalized_resampler_square_error_sum = (
                batch_source_normalized_resampler_square_error
            )
        else:
            loss_sum += loss_per_channel * batch_size
            assert target_spectrum_sum is not None
            assert generated_spectrum_sum is not None
            assert seam_ratio_sum is not None
            target_spectrum_sum += batch_target_spectrum
            generated_spectrum_sum += batch_generated_spectrum
            seam_ratio_sum += batch_seam_ratio
            assert target_value_sum is not None
            assert target_square_sum is not None
            assert generated_value_sum is not None
            assert generated_square_sum is not None
            assert valid_value_count is not None
            target_value_sum += batch_target_sum
            target_square_sum += batch_target_square_sum
            generated_value_sum += batch_generated_sum
            generated_square_sum += batch_generated_square_sum
            valid_value_count += batch_valid_count
            assert physical_square_error_sum is not None
            assert physical_valid_count is not None
            physical_square_error_sum += batch_physical_square_error
            physical_valid_count += batch_valid_count
            assert resampler_square_error_sum is not None
            assert resampler_physical_square_error_sum is not None
            assert source_normalized_resampler_square_error_sum is not None
            resampler_square_error_sum += batch_resampler_square_error
            resampler_physical_square_error_sum += batch_resampler_physical_square_error
            source_normalized_resampler_square_error_sum += (
                batch_source_normalized_resampler_square_error
            )
        local_samples += batch_size

    if (
        local_samples == 0
        or loss_sum is None
        or target_spectrum_sum is None
        or generated_spectrum_sum is None
        or seam_ratio_sum is None
        or target_value_sum is None
        or target_square_sum is None
        or generated_value_sum is None
        or generated_square_sum is None
        or valid_value_count is None
        or physical_square_error_sum is None
        or physical_valid_count is None
        or resampler_square_error_sum is None
        or resampler_physical_square_error_sum is None
        or source_normalized_resampler_square_error_sum is None
    ):
        raise ValueError("The fixed identity sample set was empty.")

    loss_per_channel = all_reduce_mean(loss_sum / local_samples)
    target_spectrum = all_reduce_mean(target_spectrum_sum / local_samples)
    generated_spectrum = all_reduce_mean(generated_spectrum_sum / local_samples)
    seam_ratio = all_reduce_mean(seam_ratio_sum / local_samples)
    assert target_value_sum is not None
    assert target_square_sum is not None
    assert generated_value_sum is not None
    assert generated_square_sum is not None
    assert valid_value_count is not None
    reduced_target_value_sum = all_reduce_mean(target_value_sum)
    reduced_target_square_sum = all_reduce_mean(target_square_sum)
    reduced_generated_value_sum = all_reduce_mean(generated_value_sum)
    reduced_generated_square_sum = all_reduce_mean(generated_square_sum)
    reduced_valid_value_count = all_reduce_mean(valid_value_count)
    reduced_physical_square_error_sum = all_reduce_mean(physical_square_error_sum)
    reduced_physical_valid_count = all_reduce_mean(physical_valid_count)
    reduced_resampler_square_error_sum = all_reduce_mean(resampler_square_error_sum)
    reduced_resampler_physical_square_error_sum = all_reduce_mean(
        resampler_physical_square_error_sum
    )
    reduced_source_normalized_resampler_square_error_sum = all_reduce_mean(
        source_normalized_resampler_square_error_sum
    )
    safe_value_count = reduced_valid_value_count.clamp_min(1)
    target_mean = reduced_target_value_sum / safe_value_count
    generated_mean = reduced_generated_value_sum / safe_value_count
    target_std = (
        (reduced_target_square_sum / safe_value_count - target_mean.square())
        .clamp_min(0)
        .sqrt()
    )
    generated_std = (
        (reduced_generated_square_sum / safe_value_count - generated_mean.square())
        .clamp_min(0)
        .sqrt()
    )
    amplitude_epsilon = torch.finfo(target_std.dtype).eps
    std_ratio = generated_std / target_std.clamp_min(amplitude_epsilon)
    mean_bias_over_target_std = (generated_mean - target_mean) / target_std.clamp_min(
        amplitude_epsilon
    )
    physical_mse = (
        reduced_physical_square_error_sum / reduced_physical_valid_count.clamp_min(1)
    )
    resampler_mse = (
        reduced_resampler_square_error_sum / reduced_physical_valid_count.clamp_min(1)
    )
    resampler_physical_mse = (
        reduced_resampler_physical_square_error_sum
        / reduced_physical_valid_count.clamp_min(1)
    )
    source_normalized_resampler_mse = (
        reduced_source_normalized_resampler_square_error_sum
        / reduced_physical_valid_count.clamp_min(1)
    )
    high_frequency_ratio = high_wavenumber_power_ratio(
        generated_spectrum, target_spectrum
    )

    logs: dict[str, float] = {
        f"{prefix}/mean/mse": float(loss_per_channel.mean().cpu()),
        f"{prefix}/actual_samples": float(local_samples * get_world_size()),
    }
    for source in (
        get_channel_loss_dict(prefix, loss_per_channel, tensor_map=trainer.tensor_map),
        get_variable_loss_dict(prefix, loss_per_channel, tensor_map=trainer.tensor_map),
        get_depth_loss_dict(prefix, loss_per_channel, tensor_map=trainer.tensor_map),
    ):
        logs.update({key: float(value.cpu()) for key, value in source.items()})
    logs.update(
        _group_values(
            f"{prefix}/high_wavenumber_power_ratio", high_frequency_ratio, trainer
        )
    )
    logs.update(_group_values(f"{prefix}/patch_seam_jump_ratio", seam_ratio, trainer))
    logs.update(_group_values(f"{prefix}/std_ratio", std_ratio, trainer))
    logs.update(_group_values(f"{prefix}/physical_mse", physical_mse, trainer))
    logs.update(
        _group_values(
            f"{prefix}/deterministic_resampler_normalized_mse",
            resampler_mse,
            trainer,
        )
    )
    logs.update(
        _group_values(
            f"{prefix}/deterministic_resampler_physical_mse",
            resampler_physical_mse,
            trainer,
        )
    )
    logs.update(
        _group_values(
            f"{prefix}/source_normalized_resampler_mse",
            source_normalized_resampler_mse,
            trainer,
        )
    )
    logs.update(
        _group_values(
            f"{prefix}/mean_bias_over_target_std",
            mean_bias_over_target_std,
            trainer,
        )
    )
    return (
        logs,
        {
            "target_zonal_power": target_spectrum.cpu(),
            "generated_zonal_power": generated_spectrum.cpu(),
        },
    )


def _identity_routes(trainer: Trainer) -> list[tuple[str, tuple[int, int]]]:
    """List the shape-distinct source/target routes in configured schedule order."""
    grids = [source.grid_size for source in trainer.data_container.sources]
    if trainer.train_schedule == "standard":
        pairs = [(grids[0], grids[0])]
    elif trainer.train_schedule == "match":
        pairs = [(grid, grid) for grid in grids]
    else:
        pairs = [(source, target) for source in grids for target in grids]
    routes = []
    for input_grid, output_grid in pairs:
        key = f"{input_grid[0]}x{input_grid[1]}_to_{output_grid[0]}x{output_grid[1]}"
        routes.append((key, output_grid))
    if len({key for key, _ in routes}) != len(routes):
        raise ValueError(
            "Identity routes must have distinct input/output grid shapes; duplicate "
            "shapes cannot be balanced or reported independently."
        )
    return routes


@torch.no_grad()
def evaluate_identity_routes(
    trainer: Trainer,
    requested_samples: int,
    patch_extent: tuple[float, float],
    *,
    sample_offset: int,
    prefix: str,
    target_time_mode: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Evaluate each configured resolution route, then form equal-route means."""
    routes = _identity_routes(trainer)
    if len(routes) == 1:
        output_grid = routes[0][1]
        patch_size = (
            round(patch_extent[0] * output_grid[0] / 180.0),
            round(patch_extent[1] * output_grid[1] / 360.0),
        )
        return evaluate_identity(
            trainer,
            requested_samples,
            patch_size,
            sample_offset=sample_offset,
            prefix=prefix,
            target_time_mode=target_time_mode,
        )

    route_count = len(routes)
    if requested_samples % route_count or sample_offset % route_count:
        raise ValueError(
            "Identity samples and offsets must divide evenly across configured routes."
        )
    samples_per_route = requested_samples // route_count
    offset_per_route = sample_offset // route_count
    route_logs: list[tuple[str, dict[str, float]]] = []
    all_spectra: dict[str, torch.Tensor] = {}
    for route, output_grid in routes:
        route_prefix = f"{prefix}/route/{route}"
        patch_size = (
            round(patch_extent[0] * output_grid[0] / 180.0),
            round(patch_extent[1] * output_grid[1] / 360.0),
        )
        logs, spectra = evaluate_identity(
            trainer,
            samples_per_route,
            patch_size,
            sample_offset=offset_per_route,
            prefix=route_prefix,
            target_time_mode=target_time_mode,
            route_filter=route,
        )
        route_logs.append((route_prefix, logs))
        all_spectra.update(
            {f"route_{route}_{name}": value for name, value in spectra.items()}
        )

    combined: dict[str, float] = {
        key: value for _, logs in route_logs for key, value in logs.items()
    }
    first_prefix, first_logs = route_logs[0]
    for key in first_logs:
        suffix = key.removeprefix(first_prefix)
        values = [logs[f"{route_prefix}{suffix}"] for route_prefix, logs in route_logs]
        combined_key = f"{prefix}{suffix}"
        combined[combined_key] = (
            sum(values) if suffix == "/actual_samples" else sum(values) / route_count
        )
    return combined, all_spectra


def train_identity(cfg: IdentityConfig) -> None:
    """Fit one deterministic sample set and evaluate a disjoint fixed set."""
    if cfg.loss != "mse":
        raise ValueError("Identity reconstruction requires plain MSE loss.")
    if cfg.steps != [1]:
        raise ValueError("Identity reconstruction requires steps: [1].")
    if cfg.model.pred_residuals:
        raise ValueError("Identity reconstruction does not use residual prediction.")
    if cfg.scheduler is not None:
        raise ValueError("Identity reconstruction expects scheduler: null.")
    if len(cfg.data.sources) > 1 and cfg.target_time_mode != "current":
        raise ValueError(
            "Cross-resolution identity reconstruction requires target_time_mode: "
            "current so labels come from the paired destination source."
        )
    if len(cfg.data.sources) > 1 and cfg.experiment.train_schedule != "mix":
        raise ValueError(
            "Cross-resolution identity reconstruction requires train_schedule: mix."
        )
    if not isinstance(cfg.model, SamudraMultiConfig):
        raise TypeError("Identity reconstruction requires a SamudraMulti model.")
    if cfg.identity_eval_only and (not cfg.finetune or cfg.resume_ckpt_path is None):
        raise ValueError(
            "identity_eval_only requires finetune: true and resume_ckpt_path so "
            "the diagnostic evaluates an explicit model checkpoint."
        )
    if cfg.identity_eval_only and cfg.epochs != 1:
        raise ValueError("identity_eval_only requires epochs: 1.")
    configured_depth = cfg.model.processor_iterations
    evaluation_depths = list(
        dict.fromkeys([configured_depth, *(cfg.identity_eval_processor_depths or [])])
    )
    if any(depth < 0 for depth in evaluation_depths):
        raise ValueError("Identity evaluation processor depths must be non-negative.")
    train_stop = cfg.identity_train_offset + cfg.identity_train_samples
    eval_stop = cfg.identity_eval_offset + cfg.identity_eval_samples
    if max(cfg.identity_train_offset, cfg.identity_eval_offset) < min(
        train_stop, eval_stop
    ):
        raise ValueError(
            "Identity training and held-out sample ranges must be disjoint."
        )

    trainer = Trainer(cfg)
    trainer.best_val_loss = math.inf
    trainer.best_inf_loss = math.inf
    trainer.init_data_loaders(cur_step=1)
    if trainer.model_patch_extent is None:
        raise ValueError("Identity reconstruction requires a physical patch extent.")
    grid = trainer.primary_src.grid_size
    patch_size = (
        round(trainer.model_patch_extent[0] * grid[0] / 180.0),
        round(trainer.model_patch_extent[1] * grid[1] / 360.0),
    )
    routes = _identity_routes(trainer)
    route_count = len(routes)
    world_size = get_world_size()
    batches_per_epoch = math.ceil(
        cfg.identity_train_samples / (cfg.batch_size * world_size)
    )
    trajectory: list[dict[str, float]] = []
    output_dir = Path(trainer.output_dir)

    try:
        for epoch in range(1, cfg.epochs + 1):
            trainer.model.train(not cfg.identity_eval_only)
            trainer.optimizer.zero_grad()
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)
            epoch_start = time.perf_counter()
            remaining_batches = batches_per_epoch % trainer.gradient_accumulation_steps
            final_cycle_start = (
                batches_per_epoch - remaining_batches
                if remaining_batches
                else batches_per_epoch
            )

            batches_seen = 0
            global_samples = 0
            if not cfg.identity_eval_only:
                for batch_index, data in enumerate(
                    _fixed_batches(
                        trainer,
                        cfg.identity_train_samples,
                        sample_offset=cfg.identity_train_offset,
                        route_count=route_count,
                    )
                ):
                    _target_for_mode(data, cfg.target_time_mode)
                    output = train_batch(trainer.model, data, trainer.loss_fn)
                    in_final_cycle = (
                        batch_index + 1 > final_cycle_start and remaining_batches > 0
                    )
                    accumulation = (
                        remaining_batches
                        if in_final_cycle
                        else trainer.gradient_accumulation_steps
                    )
                    (output.loss / accumulation).backward()
                    batches_seen += 1
                    batch_global_samples = data.get_label(0).shape[0] * world_size
                    global_samples += batch_global_samples
                    trainer.num_batches_seen += 1
                    trainer.num_samples_seen += batch_global_samples

                    should_step = (
                        batches_seen % trainer.gradient_accumulation_steps == 0
                        or batches_seen == batches_per_epoch
                    )
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()
                        trainer._ema(model=trainer.model)
                        trainer.num_optimizer_updates += 1

            elapsed = time.perf_counter() - epoch_start
            if epoch % cfg.identity_eval_frequency != 0 and epoch != cfg.epochs:
                continue
            evaluation_start = time.perf_counter()
            train_metrics, train_spectra = evaluate_identity_routes(
                trainer,
                cfg.identity_train_samples,
                tuple(trainer.model_patch_extent),
                sample_offset=cfg.identity_train_offset,
                prefix="identity/train",
                target_time_mode=cfg.target_time_mode,
            )
            heldout_metrics_by_depth: dict[int, dict[str, float]] = {}
            heldout_spectra_by_depth: dict[int, dict[str, torch.Tensor]] = {}
            for depth in evaluation_depths:
                prefix = f"identity/depth/{depth}/heldout"
                with _processor_depth(trainer, depth):
                    depth_metrics, depth_spectra = evaluate_identity_routes(
                        trainer,
                        cfg.identity_eval_samples,
                        tuple(trainer.model_patch_extent),
                        sample_offset=cfg.identity_eval_offset,
                        prefix=prefix,
                        target_time_mode=cfg.target_time_mode,
                    )
                heldout_metrics_by_depth[depth] = depth_metrics
                heldout_spectra_by_depth[depth] = depth_spectra

            configured_prefix = f"identity/depth/{configured_depth}/heldout"
            heldout_metrics = {
                key.replace(configured_prefix, "identity/heldout", 1): value
                for key, value in heldout_metrics_by_depth[configured_depth].items()
            }
            # Preserve the original metric namespace as a held-out alias so
            # historical summary tooling can compare old and new trajectories.
            compatibility_metrics = {
                key.replace("identity/heldout", "identity", 1): value
                for key, value in heldout_metrics.items()
            }
            metrics = {
                **train_metrics,
                **{
                    key: value
                    for depth_metrics in heldout_metrics_by_depth.values()
                    for key, value in depth_metrics.items()
                },
                **heldout_metrics,
                **compatibility_metrics,
            }
            evaluation_elapsed = time.perf_counter() - evaluation_start
            metrics.update(
                {
                    "identity/epoch": float(epoch),
                    "identity/epoch_seconds": elapsed,
                    "identity/training_seconds": elapsed,
                    "identity/evaluation_seconds": evaluation_elapsed,
                    "identity/total_seconds": elapsed + evaluation_elapsed,
                    "identity/samples_per_second": global_samples / elapsed,
                    "identity/optimizer_updates": float(trainer.num_optimizer_updates),
                    "identity/processed_samples": float(trainer.num_samples_seen),
                    "identity/grid_height": float(grid[0]),
                    "identity/grid_width": float(grid[1]),
                    "identity/patch_height": float(patch_size[0]),
                    "identity/patch_width": float(patch_size[1]),
                    "identity/route_count": float(route_count),
                }
            )
            if trainer.device.type == "cuda":
                gibibyte = 1024**3
                metrics.update(
                    {
                        "identity/max_cuda_memory_allocated_gib": (
                            torch.cuda.max_memory_allocated(trainer.device) / gibibyte
                        ),
                        "identity/max_cuda_memory_reserved_gib": (
                            torch.cuda.max_memory_reserved(trainer.device) / gibibyte
                        ),
                    }
                )
            trajectory.append(metrics)
            trainer.wandb_logger.log(metrics, step=epoch)
            mse = metrics["identity/mean/mse"]
            is_best = mse <= trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = mse
            logger.info(
                "Identity epoch %d: mse=%.6g, samples=%d, "
                "training_seconds=%.2f, evaluation_seconds=%.2f",
                epoch,
                mse,
                global_samples,
                elapsed,
                evaluation_elapsed,
            )

            if is_main_process():
                if is_best and not cfg.identity_eval_only:
                    logger.info(
                        "Saving lowest held-out identity checkpoint to %s",
                        trainer.ckpt_paths.best_validation_checkpoint_path,
                    )
                    trainer.save_checkpoint(
                        epoch, trainer.ckpt_paths.best_validation_checkpoint_path
                    )
                with open(output_dir / "identity_metrics.json", "w") as handle:
                    json.dump(trajectory, handle, indent=2, sort_keys=True)
                torch.save(
                    {
                        **{
                            f"train_{key}": value
                            for key, value in train_spectra.items()
                        },
                        **{
                            f"heldout_depth_{depth}_{key}": value
                            for depth, spectra in heldout_spectra_by_depth.items()
                            for key, value in spectra.items()
                        },
                        "channels": trainer.tensor_map.prognostic_var_names,
                        "grid": grid,
                        "patch_size": patch_size,
                        "epoch": epoch,
                    },
                    output_dir / "identity_spectra.pt",
                )
        if is_main_process() and not cfg.identity_eval_only:
            trainer.save_checkpoint(
                cfg.epochs, trainer.ckpt_paths.latest_checkpoint_path
            )
    finally:
        trainer.train_loader.close()
        trainer.val_loader.close()
        trainer.finish()


def main() -> None:
    cfg = IdentityConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    try:
        train_identity(cfg)
    except Exception:
        logger.exception("Identity reconstruction failed with an exception")
        raise
    finally:
        destroy_distributed_mode()


if __name__ == "__main__":
    main()
