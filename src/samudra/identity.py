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
from samudra.datasets import TrainData
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


def _fixed_batches(
    trainer: Trainer,
    requested_samples: int,
    *,
    sample_offset: int,
):
    """Yield an exact deterministic range from the validation stream.

    Offsets and sample counts refer to the global stream and must align with a
    global loader batch. This keeps every distributed rank on the same number of
    batches without slicing ``TrainData`` objects differently across ranks.
    """
    trainer.val_loader.set_epoch(0)
    skipped_samples = 0
    selected_samples = 0
    for data in trainer.val_loader:
        batch_samples = data.get_label(0).shape[0] * get_world_size()
        if skipped_samples < sample_offset:
            next_skipped = skipped_samples + batch_samples
            if next_skipped > sample_offset:
                raise ValueError(
                    "identity sample offsets must align with global loader batches; "
                    f"offset {sample_offset} falls inside a batch ending at "
                    f"sample {next_skipped}."
                )
            skipped_samples = next_skipped
            continue

        next_selected = selected_samples + batch_samples
        if next_selected > requested_samples:
            raise ValueError(
                "identity sample counts must align with global loader batches; "
                f"requested {requested_samples} samples but the next batch would "
                f"select {next_selected}."
            )
        selected_samples = next_selected
        yield data
        if selected_samples == requested_samples:
            break

    if skipped_samples < sample_offset or selected_samples < requested_samples:
        raise ValueError(
            "The validation stream is too short for the requested identity sample "
            f"range [{sample_offset}, {sample_offset + requested_samples}); selected "
            f"{selected_samples} samples after skipping {skipped_samples}."
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


@torch.no_grad()
def evaluate_identity(
    trainer: Trainer,
    requested_samples: int,
    patch_size: tuple[int, int],
    *,
    sample_offset: int,
    prefix: str,
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
    local_samples = 0
    base_channels = len(trainer.tensor_map.prognostic_var_names)

    for data in _fixed_batches(trainer, requested_samples, sample_offset=sample_offset):
        target = set_identity_target(data)
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
    if len(cfg.data.sources) != 1:
        raise ValueError("Run identity reconstruction on exactly one resolution.")
    if not isinstance(cfg.model, SamudraMultiConfig):
        raise TypeError("Identity reconstruction requires a SamudraMulti model.")
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
    world_size = get_world_size()
    batches_per_epoch = math.ceil(
        cfg.identity_train_samples / (cfg.batch_size * world_size)
    )
    trajectory: list[dict[str, float]] = []
    output_dir = Path(trainer.output_dir)

    try:
        for epoch in range(1, cfg.epochs + 1):
            trainer.model.train(True)
            trainer.optimizer.zero_grad()
            epoch_start = time.perf_counter()
            remaining_batches = batches_per_epoch % trainer.gradient_accumulation_steps
            final_cycle_start = (
                batches_per_epoch - remaining_batches
                if remaining_batches
                else batches_per_epoch
            )

            batches_seen = 0
            global_samples = 0
            for batch_index, data in enumerate(
                _fixed_batches(
                    trainer,
                    cfg.identity_train_samples,
                    sample_offset=cfg.identity_train_offset,
                )
            ):
                set_identity_target(data)
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
            train_metrics, train_spectra = evaluate_identity(
                trainer,
                cfg.identity_train_samples,
                patch_size,
                sample_offset=cfg.identity_train_offset,
                prefix="identity/train",
            )
            heldout_metrics_by_depth: dict[int, dict[str, float]] = {}
            heldout_spectra_by_depth: dict[int, dict[str, torch.Tensor]] = {}
            for depth in evaluation_depths:
                prefix = f"identity/depth/{depth}/heldout"
                with _processor_depth(trainer, depth):
                    depth_metrics, depth_spectra = evaluate_identity(
                        trainer,
                        cfg.identity_eval_samples,
                        patch_size,
                        sample_offset=cfg.identity_eval_offset,
                        prefix=prefix,
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
            metrics.update(
                {
                    "identity/epoch": float(epoch),
                    "identity/epoch_seconds": elapsed,
                    "identity/samples_per_second": global_samples / elapsed,
                    "identity/optimizer_updates": float(trainer.num_optimizer_updates),
                    "identity/processed_samples": float(trainer.num_samples_seen),
                    "identity/grid_height": float(grid[0]),
                    "identity/grid_width": float(grid[1]),
                    "identity/patch_height": float(patch_size[0]),
                    "identity/patch_width": float(patch_size[1]),
                }
            )
            trajectory.append(metrics)
            trainer.wandb_logger.log(metrics, step=epoch)
            mse = metrics["identity/mean/mse"]
            logger.info(
                "Identity epoch %d: mse=%.6g, samples=%d, seconds=%.2f",
                epoch,
                mse,
                global_samples,
                elapsed,
            )

            if is_main_process():
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
        if is_main_process():
            trainer.best_val_loss = min(row["identity/mean/mse"] for row in trajectory)
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
