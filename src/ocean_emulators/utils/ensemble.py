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


def compute_ensemble_metrics(
    ensemble_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute spread, skill (RMSE), and spread/skill ratio for ensemble predictions.

    Args:
        ensemble_predictions: (ensemble_size, steps, batch, channels, lat, lon)
        targets: (steps, batch, channels, lat, lon)

    Returns:
        spread: scalar, mean std across ensemble members (last step)
        skill: scalar, RMSE of ensemble mean vs targets (last step)
        spread_skill_ratio: scalar, spread / skill (last step)
        per_step_metrics: dict with per-step spread, skill, spread_skill_ratio
    """
    num_steps = targets.shape[0]
    per_step_metrics: dict[str, torch.Tensor] = {}

    # Compute per-step metrics
    for step in range(num_steps):
        preds_step = ensemble_predictions[:, step]  # (E, B, C, H, W)
        target_step = targets[step]  # (B, C, H, W)

        # Spread: std across ensemble members, averaged over spatial dims
        step_spread = preds_step.std(dim=0).mean()  # scalar

        # Skill: RMSE of ensemble mean vs target
        ensemble_mean = preds_step.mean(dim=0)  # (B, C, H, W)
        mse = ((ensemble_mean - target_step) ** 2).mean()
        step_skill = torch.sqrt(mse)  # scalar

        # Spread/skill ratio (avoid div by zero)
        step_ss_ratio = step_spread / step_skill.clamp(min=1e-12)

        per_step_metrics[f"spread_step{step}"] = step_spread
        per_step_metrics[f"skill_step{step}"] = step_skill
        per_step_metrics[f"spread_skill_ratio_step{step}"] = step_ss_ratio

    # Return last step as the main metrics (for backward compatibility)
    spread = per_step_metrics[f"spread_step{num_steps - 1}"]
    skill = per_step_metrics[f"skill_step{num_steps - 1}"]
    spread_skill_ratio = per_step_metrics[f"spread_skill_ratio_step{num_steps - 1}"]

    return spread, skill, spread_skill_ratio, per_step_metrics


def compute_physical_ensemble_metrics(
    ensemble_predictions: torch.Tensor,
    targets: torch.Tensor,
    prog_mean: torch.Tensor,
    prog_std: torch.Tensor,
    area_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute spread/skill in physical (denormalized) units with area weighting.

    Args:
        ensemble_predictions: (ensemble_size, steps, batch, channels, lat, lon) - NORMALIZED
        targets: (steps, batch, channels, lat, lon) - NORMALIZED
        prog_mean: (num_vars,) - mean for denormalization (single timestep)
        prog_std: (num_vars,) - std for denormalization (single timestep)
        area_weights: (lat, lon) - area weights (e.g., cos(lat))

    Returns:
        dict with physical_spread, physical_skill, physical_spread_skill_ratio
    """
    # Use last step only (like validation)
    preds_last = ensemble_predictions[:, -1]  # (E, B, C, H, W)
    target_last = targets[-1]  # (B, C, H, W)

    E, B, C, H, W = preds_last.shape
    device = preds_last.device
    dtype = preds_last.dtype

    # Handle stacked timesteps: C = num_vars * (hist + 1)
    # prog_mean/std are (num_vars,) so we need to tile them
    num_vars = prog_mean.shape[0]
    hist_plus_1 = C // num_vars
    
    if C != num_vars * hist_plus_1:
        raise ValueError(f"Channels {C} not divisible by num_vars {num_vars}")
    
    # Tile normalization params to match stacked channels
    prog_mean_tiled = prog_mean.repeat(hist_plus_1)  # (C,)
    prog_std_tiled = prog_std.repeat(hist_plus_1)  # (C,)

    # Move normalization params to device and reshape
    prog_mean_tiled = prog_mean_tiled.to(device=device, dtype=dtype).view(1, C, 1, 1)
    prog_std_tiled = prog_std_tiled.to(device=device, dtype=dtype).view(1, C, 1, 1)

    # Denormalize predictions and targets
    preds_phys = preds_last * prog_std_tiled + prog_mean_tiled  # (E, B, C, H, W)
    target_phys = target_last * prog_std_tiled + prog_mean_tiled  # (B, C, H, W)

    # Normalize area weights
    w_hw = area_weights.to(device=device, dtype=dtype)
    w_hw = w_hw / w_hw.sum().clamp_min(1e-12)  # (H, W)

    # Compute spread per channel with area weighting
    # std across ensemble: (B, C, H, W)
    preds_std = preds_phys.std(dim=0)  # (B, C, H, W)
    # weighted mean over spatial dims
    spread_per_channel = (preds_std * w_hw.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, C)
    spread = spread_per_channel.mean()  # scalar

    # Compute skill (RMSE) with area weighting
    ensemble_mean = preds_phys.mean(dim=0)  # (B, C, H, W)
    diff_sq = (ensemble_mean - target_phys) ** 2  # (B, C, H, W)
    mse_per_channel = (diff_sq * w_hw.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, C)
    rmse_per_channel = torch.sqrt(mse_per_channel.mean(dim=0))  # (C,)
    skill = rmse_per_channel.mean()  # scalar

    # Spread/skill ratio
    ss_ratio = spread / skill.clamp(min=1e-12)

    return {
        "physical_spread": spread,
        "physical_skill": skill,
        "physical_spread_skill_ratio": ss_ratio,
    }


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
