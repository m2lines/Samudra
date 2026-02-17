import logging
from collections.abc import Callable
from os import PathLike

logger = logging.getLogger(__name__)

import torch

from ocean_emulators.aggregator import InferenceEvaluatorAggregator
from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.models.base import BaseModel
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.ensemble import (
    compute_crps_loss_for_ensemble,
    compute_ensemble_metrics,
    compute_physical_ensemble_metrics,
    generate_ensemble_predictions,
    members_for_rank,
)
from ocean_emulators.utils.output import (
    ModelInferenceOutput,
    TrainBatchOutput,
    ValBatchOutput,
)
from ocean_emulators.utils.wandb import get_record_to_wandb
from ocean_emulators.utils.writer import ZarrWriter


class Stepper:
    def __init__(self):
        pass

    @staticmethod
    def train_batch(
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> TrainBatchOutput:
        loss_per_channel = model(batch, loss_fn=loss_fn)
        loss = torch.mean(loss_per_channel)
        return TrainBatchOutput(loss, loss_per_channel)

    @staticmethod
    def train_batch_ensemble(
        model: torch.nn.Module,
        batch: TrainData,
        loss_fn: Callable,
        ensemble_size: int,
        distributed: bool,
        global_step: int = 0,
        prog_mean: torch.Tensor | None = None,
        prog_std: torch.Tensor | None = None,
        area_weights: torch.Tensor | None = None,
    ) -> TrainBatchOutput:
        """Training step with ensemble generation."""
        # Generate ensemble predictions (sharded across GPUs if distributed)
        ensemble_preds, targets = generate_ensemble_predictions(
            model, batch, ensemble_size, distributed, global_step
        )

        # Compute CRPS loss (returns per-channel)
        loss_per_channel = compute_crps_loss_for_ensemble(
            ensemble_preds, targets, loss_fn
        )

        # Compute scalar loss for backprop
        loss = torch.mean(loss_per_channel)

        # Compute ensemble metrics for logging (no grad needed)
        with torch.no_grad():
            spread, skill, spread_skill_ratio, per_step_metrics = compute_ensemble_metrics(
                ensemble_preds, targets
            )

            # Compute physical (denormalized) metrics if normalization params provided
            if prog_mean is not None and prog_std is not None and area_weights is not None:
                physical_metrics = compute_physical_ensemble_metrics(
                    ensemble_preds, targets, prog_mean, prog_std, area_weights
                )
                per_step_metrics.update(physical_metrics)

        return TrainBatchOutput(
            loss,
            loss_per_channel,
            ensemble_spread=spread,
            ensemble_skill=skill,
            spread_skill_ratio=spread_skill_ratio,
            per_step_metrics=per_step_metrics,
        )

    @staticmethod
    @torch.no_grad()
    def validate_batch(
        model: BaseModel | torch.nn.parallel.DistributedDataParallel,
        batch: TrainData,
        loss_fn: Callable,
        ensemble_size: int,
        is_crps: bool = False,
    ) -> ValBatchOutput:
        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)
        # TODO(jder): we need the underlying model so we can use forward_once;
        # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
        model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )

        if ensemble_size > 1:
            # Generate ensemble predictions
            ensemble_outs_list: list[torch.Tensor] = []
            for member_idx in range(ensemble_size):
                outs = model.forward_once(input)
                ensemble_outs_list.append(outs)

            # Stack: (ensemble_size, batch, channels, lat, lon)
            # Note: forward_once may return (B,T,C,H,W) or (B,C,H,W)
            # Ensure we have (E,B,C,H,W) for ensemble_data
            ensemble_outs = torch.stack(ensemble_outs_list, dim=0)

            # Debug: log shape before processing
            logger.debug(
                f"validate_batch: ensemble_outs.shape before = {ensemble_outs.shape}"
            )

            # Remove time dimension if present (take last time step)
            if ensemble_outs.ndim == 6:  # (E, B, T, C, H, W)
                ensemble_outs = ensemble_outs[:, :, -1, :, :, :]  # (E, B, C, H, W)
                logger.debug(
                    f"validate_batch: removed time dim, new shape = {ensemble_outs.shape}"
                )

            # Verify final shape
            if ensemble_outs.ndim != 5:
                raise ValueError(
                    f"Expected ensemble_outs to be 5D (E,B,C,H,W), got {ensemble_outs.ndim}D: {ensemble_outs.shape}"
                )

            # Compute ensemble mean as the prediction
            outs = ensemble_outs.mean(dim=0)

            # Compute loss based on loss function type
            if is_crps:
                # CRPS loss expects (ensemble_size, batch, ...) and computes internally
                loss_per_channel = loss_fn(ensemble_outs, label)
            else:
                # Standard losses expect (batch, ...) and work on ensemble mean
                loss_per_channel = loss_fn(outs, label)

            loss = torch.mean(loss_per_channel)

            # Return with ensemble data for computing statistics
            return ValBatchOutput(
                loss, loss_per_channel, input, label, outs, ensemble_data=ensemble_outs
            )
        else:
            # Deterministic validation
            outs = model.forward_once(input)
            loss_per_channel = loss_fn(outs, label)
            loss = torch.mean(loss_per_channel)

            return ValBatchOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def validate_batch_distributed(
        model: BaseModel | torch.nn.parallel.DistributedDataParallel,
        batch: TrainData,
        loss_fn: Callable,
        ensemble_size: int,
        is_crps: bool = False,
    ) -> ValBatchOutput:
        """Validation with ensemble members distributed across GPUs.

        Unlike validate_batch which runs all ensemble members sequentially on each GPU,
        this version shards ensemble members across GPUs and gathers results.
        This reduces memory per GPU from O(ensemble_size) to O(ensemble_size/world_size).

        Note: All GPUs must process the same batch for this to work correctly.
        """
        import torch.distributed as dist

        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)

        # Unwrap DDP model
        base_model = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )

        if ensemble_size > 1:
            # Get distributed info
            world_size = (
                dist.get_world_size()
                if (dist.is_available() and dist.is_initialized())
                else 1
            )
            rank = (
                dist.get_rank()
                if (dist.is_available() and dist.is_initialized())
                else 0
            )

            # Determine which ensemble members this rank computes
            start_idx, local_count = members_for_rank(ensemble_size, rank, world_size)

            logger.debug(
                f"validate_batch_distributed: rank {rank}/{world_size}, "
                f"computing members {start_idx} to {start_idx + local_count - 1}"
            )

            # Generate local ensemble predictions
            local_ensemble_outs: list[torch.Tensor] = []
            for member_idx in range(local_count):
                outs = base_model.forward_once(input)
                local_ensemble_outs.append(outs)

            # Stack local members: (local_count, batch, ...)
            local_ensemble = torch.stack(local_ensemble_outs, dim=0)

            # Remove time dimension if present (take last time step)
            if local_ensemble.ndim == 6:  # (E, B, T, C, H, W)
                local_ensemble = local_ensemble[:, :, -1, :, :, :]  # (E, B, C, H, W)

            # Gather ensemble members from all ranks
            if world_size > 1:
                # Use all_gather to collect predictions from all ranks
                gathered_list = [
                    torch.zeros_like(local_ensemble) for _ in range(world_size)
                ]

                # Handle uneven distribution - need to pad/gather with different sizes
                # For simplicity, assume ensemble_size is divisible by world_size
                # or use all_gather with different sizes
                local_counts = [
                    members_for_rank(ensemble_size, r, world_size)[1]
                    for r in range(world_size)
                ]

                if all(c == local_count for c in local_counts):
                    # Even distribution - simple all_gather
                    dist.all_gather(gathered_list, local_ensemble)
                    ensemble_outs = torch.cat(gathered_list, dim=0)
                else:
                    # Uneven distribution - need to handle different sizes
                    # Pad to max size, gather, then trim
                    max_count = max(local_counts)
                    if local_count < max_count:
                        padding = torch.zeros(
                            max_count - local_count,
                            *local_ensemble.shape[1:],
                            device=local_ensemble.device,
                            dtype=local_ensemble.dtype,
                        )
                        local_ensemble_padded = torch.cat(
                            [local_ensemble, padding], dim=0
                        )
                    else:
                        local_ensemble_padded = local_ensemble

                    gathered_list = [
                        torch.zeros_like(local_ensemble_padded)
                        for _ in range(world_size)
                    ]
                    dist.all_gather(gathered_list, local_ensemble_padded)

                    # Trim each rank's contribution to actual count
                    trimmed = []
                    for r, gathered in enumerate(gathered_list):
                        count = local_counts[r]
                        trimmed.append(gathered[:count])
                    ensemble_outs = torch.cat(trimmed, dim=0)
            else:
                ensemble_outs = local_ensemble

            # Verify final shape
            if ensemble_outs.ndim != 5:
                raise ValueError(
                    f"Expected ensemble_outs to be 5D (E,B,C,H,W), got {ensemble_outs.ndim}D: {ensemble_outs.shape}"
                )

            logger.debug(
                f"validate_batch_distributed: final ensemble_outs.shape = {ensemble_outs.shape}"
            )

            # Compute ensemble mean as the prediction
            outs = ensemble_outs.mean(dim=0)

            # Compute loss based on loss function type
            if is_crps:
                # CRPS loss expects (ensemble_size, batch, ...) and computes internally
                loss_per_channel = loss_fn(ensemble_outs, label)
            else:
                # Standard losses expect (batch, ...) and work on ensemble mean
                loss_per_channel = loss_fn(outs, label)

            loss = torch.mean(loss_per_channel)

            # Return with ensemble data for computing statistics
            return ValBatchOutput(
                loss, loss_per_channel, input, label, outs, ensemble_data=ensemble_outs
            )
        else:
            # Deterministic validation
            outs = base_model.forward_once(input)
            loss_per_channel = loss_fn(outs, label)
            loss = torch.mean(loss_per_channel)

            return ValBatchOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inference(
        model: BaseModel,
        dataset: InferenceDataset,
        inf_aggregator: InferenceEvaluatorAggregator,
        epoch: int,
        output_dir: str | PathLike | None = None,
        model_path: str | PathLike | None = None,
        num_model_steps_forward: int = 200,
        save_zarr: bool = False,
        ensemble_size: int = 1,
    ) -> None:
        """Run inference with optional ensemble averaging.
        
        Args:
            ensemble_size: Number of ensemble members to generate at each step.
                If > 1, uses ensemble mean as input to next step (prevents noise compounding).
        """
        if save_zarr:
            if output_dir is None or model_path is None:
                raise ValueError(
                    "output_dir and model_path must be provided if save_zarr is True"
                )
            coords = dataset.get_coords_dict()
            if num_model_steps_forward > 0:
                chunk_size = num_model_steps_forward
            else:
                chunk_size = 20
            writer = ZarrWriter(
                output_dir,
                coords=coords,
                hist=inf_aggregator.hist,
                model_path=model_path,
                time_chunk_size=chunk_size,
            )
        else:
            writer = None
        record_logs = get_record_to_wandb(label="inference")
        logger.info(f"Inference [epoch {epoch}]: processing initial prognostic.")
        logs = inf_aggregator.record_initial_prognostic(
            initial_prognostic=dataset.initial_prognostic.to(get_device()),
        )
        record_logs(logs)
        num_model_steps = len(dataset)
        num_steps_list = []

        # If num_model_steps_forward is -1, then we are doing a full forward pass
        if num_model_steps_forward == -1:
            num_steps_list = [num_model_steps]
        else:
            # Windows of partial forward passes
            num_loops = num_model_steps // num_model_steps_forward
            if num_loops > 0:
                num_steps_list = [num_model_steps_forward] * num_loops
                last_model_steps_forward = num_model_steps % num_model_steps_forward
                if last_model_steps_forward > 0:
                    num_steps_list = num_steps_list + [last_model_steps_forward]
            else:
                num_steps_list = [num_model_steps]

        num_loops = len(num_steps_list)
        initial_prognostic = dataset.initial_prognostic
        step = 0
        for loop, num_steps in enumerate(num_steps_list):
            logger.info(
                f"Inference [epoch {epoch}]: loop {loop} of {num_loops - 1}. "
                f"Stepping {num_steps} steps forward."
            )
            IO: ModelInferenceOutput = model.inference(
                dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=step,
                num_steps=num_steps,
                epoch=epoch,
                ensemble_size=ensemble_size,
            )
            # Setting initial prognostic for next loop
            initial_prognostic = IO.prediction[-1].unsqueeze(0).clone()
            if writer:
                logger.info(f"Writing to zarr...")
                writer.record_batch(IO)
                writer.write()

            logger.info(f"Recording logs...")
            logs = inf_aggregator.record_batch(IO)
            logger.info(f"Logging to wandb...")
            record_logs(logs)
            step += num_steps
