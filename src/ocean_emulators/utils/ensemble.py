import torch
import torch.distributed as dist


def generate_ensemble_predictions(
    model: torch.nn.Module,
    train_data,
    ensemble_size: int,
    distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ensemble_size < 2:
        raise ValueError(f"ensemble_size must be >= 2 for CRPS, got {ensemble_size}")

    # Get distributed info
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
        distributed = False

    # Calculate members per GPU
    if ensemble_size % world_size != 0:
        raise ValueError(
            f"ensemble_size ({ensemble_size}) must be divisible by world_size ({world_size}). "
            f"Use ensemble_size={ensemble_size + (world_size - ensemble_size % world_size)} instead."
        )

    members_per_gpu = ensemble_size // world_size

    local_ensemble_outputs = []

    # Generate predictions for local ensemble members (subset on this GPU)
    for local_idx in range(members_per_gpu):
        outputs = model(train_data)  # Returns list[Tensor], one per step

        # Stack all rollout steps for this member
        # For single-step training, outputs will have length 1
        member_predictions = torch.stack(
            outputs, dim=0
        )  # (steps, batch, channels, lat, lon)
        local_ensemble_outputs.append(member_predictions)

    # Stack local ensemble members: (local_ensemble_size, steps, batch, channels, lat, lon)
    local_ensemble_predictions = torch.stack(local_ensemble_outputs, dim=0)

    # Gather ensemble members from all GPUs if distributed
    if distributed:
        # Prepare list for gathering
        gathered_tensors = [
            torch.zeros_like(local_ensemble_predictions) for _ in range(world_size)
        ]

        # All-gather operation
        dist.all_gather(gathered_tensors, local_ensemble_predictions)

        # Concatenate along ensemble dimension
        # (ensemble_size, steps, batch, channels, lat, lon)
        ensemble_predictions = torch.cat(gathered_tensors, dim=0)
    else:
        # Single GPU: just use local predictions
        ensemble_predictions = local_ensemble_predictions

    # Get targets for each step
    targets: list[torch.Tensor] = []
    for step in range(len(train_data)):
        targets.append(train_data.get_label(step))
    targets_stacked = torch.stack(targets, dim=0)  # (steps, batch, channels, lat, lon)

    return ensemble_predictions, targets_stacked


def compute_crps_loss_for_ensemble(
    ensemble_predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    num_steps = targets.shape[0]
    num_channels = targets.shape[2]

    # Accumulate per-channel losses
    total_loss_per_channel = torch.zeros(num_channels, device=targets.device)

    for step in range(num_steps):
        # Get predictions and target for this step
        step_predictions = ensemble_predictions[
            :, step
        ]  # (ensemble, batch, channels, lat, lon)
        step_target = targets[step]  # (batch, channels, lat, lon)

        # Compute CRPS for this step (returns per-channel)
        step_loss = loss_fn(step_predictions, step_target)  # (channels,)

        total_loss_per_channel += step_loss

    # Average over steps and return per-channel losses
    return total_loss_per_channel / num_steps


def forward_ensemble_training(
    model: torch.nn.Module,
    train_data,
    ensemble_size: int,
    loss_fn,
    distributed: bool,
) -> torch.Tensor:
    ensemble_preds, targets = generate_ensemble_predictions(
        model, train_data, ensemble_size, distributed
    )

    return compute_crps_loss_for_ensemble(ensemble_preds, targets, loss_fn)
