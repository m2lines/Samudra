import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as diff_all_gather


def members_for_rank(total_members: int, rank: int, world_size: int) -> tuple[int, int]:
    """Calculate which ensemble members this rank should compute.

    Returns:
        (start_idx, count): start index and number of members for this rank
    """
    base = total_members // world_size
    extra = total_members % world_size
    count = base + (1 if rank < extra else 0)
    start = base * rank + min(rank, extra)
    return start, count


def generate_ensemble_predictions(
    model: torch.nn.Module,
    train_data,
    ensemble_size: int,
    distributed: bool,
    global_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ensemble predictions with proper distributed sharding.

    For CRPS, all ranks process the SAME input samples, but each rank generates
    a disjoint subset of ensemble members. Members are then gathered across ranks
    using differentiable all_gather so gradients flow correctly.

    Args:
        model: The model to run
        train_data: Input batch (replicated across all ranks)
        ensemble_size: Total number of ensemble members
        distributed: Whether running in distributed mode
        global_step: Global training step (for deterministic member seeding)
    """
    if ensemble_size < 2:
        raise ValueError(f"ensemble_size must be >= 2 for CRPS, got {ensemble_size}")

    # Get distributed info
    world_size = (
        dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    )
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    # Determine which ensemble members this rank computes
    start_idx, local_count = members_for_rank(ensemble_size, rank, world_size)

    local_ensemble_outputs = []

    # Generate ensemble members for this rank
    for i in range(local_count):
        outputs = model(train_data)  # Returns list[Tensor], one per step

        # Stack all rollout steps for this member
        member_predictions = torch.stack(
            outputs, dim=0
        )  # (steps, batch, channels, lat, lon)
        local_ensemble_outputs.append(member_predictions)

    # Stack local ensemble members: (local_members, steps, batch, channels, lat, lon)
    local_ensemble_predictions = torch.stack(local_ensemble_outputs, dim=0)

    # Gather ensemble members from all ranks using differentiable all_gather
    if world_size > 1 and distributed:
        gathered = diff_all_gather(local_ensemble_predictions)
        # Concatenate along ensemble dimension: (ensemble_size, steps, batch, channels, lat, lon)
        ensemble_predictions = torch.cat(list(gathered), dim=0)
    else:
        # Single GPU or non-distributed: use local predictions directly
        ensemble_predictions = local_ensemble_predictions

    # Get targets for each step (same on all ranks)
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
