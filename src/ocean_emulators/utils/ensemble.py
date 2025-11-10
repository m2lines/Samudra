import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


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
        if dist.is_initialized() and dist.get_rank() == 0:
            import pdb

            print(f"\n[rank0] entering pdb before forward, member={local_idx}\n")
            pdb.set_trace()
        else:
            # Prevent other ranks from running ahead
            if dist.is_initialized():
                dist.barrier()
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
        if differentiable_all_gather is None:
            raise RuntimeError(
                "Differentiable all_gather is not available in this PyTorch version; "
                "upgrade torch.distributed.nn.functional or disable distributed ensemble training."
            )

        gathered = differentiable_all_gather(local_ensemble_predictions)

        # Concatenate along ensemble dimension
        # (ensemble_size, steps, batch, channels, lat, lon)
        ensemble_predictions = torch.cat(list(gathered), dim=0)
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

    # Accumulate per-channel losses
    per_step_losses: list[torch.Tensor] = []

    for step in range(num_steps):
        # Get predictions and target for this step
        step_predictions = ensemble_predictions[
            :, step
        ]  # (ensemble, batch, channels, lat, lon)
        step_target = targets[step]  # (batch, channels, lat, lon)

        # Compute CRPS for this step (returns per-channel)
        step_loss = loss_fn(step_predictions, step_target)  # (channels,)

        per_step_losses.append(step_loss)

    # Average over steps and return per-channel losses
    return torch.stack(per_step_losses, dim=0).mean(dim=0)


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
