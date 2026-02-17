import logging

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ocean_emulators.constants import Grid
from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size

logger = logging.getLogger(__name__)


def decomposed_mse(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Standard MSE loss computed per channel."""
    pred = pred * wet
    target = target * wet
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mse_diff_weighted(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """MSE loss with weighted differences."""
    pred = pred * wet
    target = target * wet
    # Compute standard MSE
    mse = F.mse_loss(pred, target, reduction="none")

    # Weight the differences more heavily
    diff_weight = 2.0  # Adjustable weight factor
    diff_mse = (
        F.mse_loss(
            pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], reduction="none"
        )
        * diff_weight
    )

    # Combine losses
    combined_loss = torch.cat([mse[:, :1], diff_mse], dim=1)
    return combined_loss.mean(dim=(0, 2, 3))


def decomposed_mse_cos_weighted(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """MSE loss weighted by cosine of latitude."""
    pred = pred * wet
    target = target * wet
    weights = cos.view(1, 1, -1, 1)  # Reshape for broadcasting
    mse = F.mse_loss(pred, target, reduction="none")
    weighted_mse = mse * weights
    return weighted_mse.mean(dim=(0, 2, 3))


def decomposed_mse_scaled(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    """MSE loss with scaled residuals."""
    pred = pred * wet
    target = target * wet
    scaled_pred = pred * scaling.view(1, -1, 1, 1)
    scaled_target = target * scaling.view(1, -1, 1, 1)
    return F.mse_loss(scaled_pred, scaled_target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mse_mae(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Combined MSE and MAE loss."""
    pred = pred * wet
    target = target * wet
    mse = F.mse_loss(pred, target, reduction="none")
    mae = F.l1_loss(pred, target, reduction="none")
    combined = (mse + mae) / 2
    return combined.mean(dim=(0, 2, 3))


def crps_ensemble(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
) -> torch.Tensor:
    """CRPS = E|X - Y| - 0.5 * E|X - X'|
    where X are ensemble predictions and Y is ground truth.

    Args:
        pred: Ensemble predictions (ensemble_size, batch, channels, lat, lon)
        target: Ground truth (batch, channels, lat, lon)
        wet: Ocean mask

    Returns:
        CRPS per channel (channels,)

    References:
        GraphCast implementation - https://github.com/google-deepmind/graphcast/blob/main/gencast_mini_demo.ipynb
    """
    if pred.shape[0] < 2:
        raise ValueError("CRPS requires at least 2 ensemble members (dim 0)")

    wet_float = wet.to(device=pred.device, dtype=pred.dtype)
    if wet_float.ndim == 2:
        wet_float = wet_float.unsqueeze(0)
    elif wet_float.ndim != 3:
        raise ValueError("wet mask must have shape (C,H,W) or (H,W)")

    pred = pred * wet_float.unsqueeze(0).unsqueeze(0)
    target = target * wet_float.unsqueeze(0)

    ensemble_size = pred.shape[0]

    # Skill: E|X - Y| = mean over ensemble of |pred - target|
    mean_abs_err = torch.abs(target.unsqueeze(0) - pred).mean(dim=0)

    # Spread: E|X - X'| = mean over all pairs of |pred_i - pred_j|
    # Rename ensemble dim for pairwise computation
    pred_i = pred.unsqueeze(1)  # (ensemble, 1, batch, channels, lat, lon)
    pred_j = pred.unsqueeze(0)  # (1, ensemble, batch, channels, lat, lon)

    # Fair CRPS: exclude diagonal (i == j) for unbiased estimate
    # Sum all pairs, then normalize by ensemble_size * (ensemble_size - 1)
    pairwise_diff = torch.abs(pred_i - pred_j).sum(dim=(0, 1))
    mean_abs_diff = pairwise_diff / (ensemble_size * (ensemble_size - 1))

    # CRPS = skill - 0.5 * spread
    crps = mean_abs_err - 0.5 * mean_abs_diff

    return (crps * wet_float.unsqueeze(0)).mean(dim=(0, 2, 3))


class MseDynamic:
    """A loss function that scales each channel to contribute equally to the loss.

    This uses a rolling estimate of the loss of each channel to scale each
    channel's loss, discouraging the model from focusing on only a few channels.

    See: https://openathena.slack.com/archives/C08CYM42DT3/p1752275713570969
    """

    N_WINDOW = 25
    """Rolling window size to average over. (~number of steps)"""

    def __init__(
        self,
        wet: Grid,
        stds: Float[torch.Tensor, " var"],
        *,
        should_limit: bool,
    ):
        self._wet: Grid = wet
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            stds.shape[0], device=wet.device
        )
        if should_limit:
            vars: Float[torch.Tensor, " var"] = stds.pow(2)
            self._limits: Float[torch.Tensor, " var"] | None = 1.0 / vars
        else:
            self._limits = None

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = decomposed_mse(
            pred, target, self._wet
        )
        # Channels are time-major: (hist+1) * var.
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(-1, self._per_channel_scale.shape[0])
            * self._per_channel_scale
        )
        return scaled_loss_including_history_dimension.reshape(-1)

    def update(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> None:
        """Given the prediction & target for this step, update the per-channel scale."""
        mse_loss = decomposed_mse(pred, target, self._wet)
        mse_loss = torch.where(mse_loss == 0, 1e-8, mse_loss)
        new_target_weights_with_history: Float[torch.Tensor, " hist*var"] = (
            1.0 / mse_loss
        )
        # Reshape from channels * history to channels
        # by averaging along the `hist` dimension
        new_target_weights: Float[torch.Tensor, " var"] = (
            new_target_weights_with_history.reshape(
                -1, self._per_channel_scale.shape[0]
            ).mean(dim=0)
        )
        if self._limits is not None:
            new_target_weights = new_target_weights.min(self._limits)

        if get_world_size() > 1:
            all_reduce_mean(new_target_weights)

        self._per_channel_scale = (
            self._per_channel_scale * (MseDynamic.N_WINDOW - 1) + new_target_weights
        ) / MseDynamic.N_WINDOW

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale

    # new methods for saving and loading state
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {"per_channel_scale": self._per_channel_scale.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from ``state_dict``."""
        if "per_channel_scale" in state:
            self._per_channel_scale = state["per_channel_scale"].to(self._wet.device)


class CrpsDynamic:
    """CRPS loss with dynamic per-channel scaling.

    This uses a rolling estimate of the CRPS of each channel to scale each
    channel's loss, discouraging the model from focusing on only a few channels.

    Similar to MseDynamic but for ensemble CRPS loss.
    
    Optionally includes a spread regularizer that encourages diversity early in training:
        L = CRPS - λ * spread_regularizer
    where λ ramps down from spread_reg_lambda to 0 over spread_reg_steps.
    """

    N_WINDOW = 25
    """Rolling window size to average over. (~number of steps)"""

    def __init__(
        self,
        wet: Grid,
        *,
        limit: float | None,
        num_channels: int,
        var_names: list[str] | None = None,
        spread_reg_lambda: float = 0.0,
        spread_reg_steps: int = 10000,
    ):
        self._wet: Grid = wet
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            num_channels, device=wet.device
        )
        self._limit = limit
        self._var_names = var_names or [f"var_{i}" for i in range(num_channels)]
        # Store last clamping stats for WandB logging
        self._last_clamp_stats: dict[str, float] | None = None
        
        # Spread regularizer parameters
        self._spread_reg_lambda = spread_reg_lambda
        self._spread_reg_steps = spread_reg_steps
        self._step = 0
        self._last_spread_reg_lambda: float = spread_reg_lambda
        self._last_spread_bonus: float = 0.0

    def get_clamp_stats(self) -> dict[str, float] | None:
        """Return last clamping statistics for WandB logging."""
        return self._last_clamp_stats

    def __call__(
        self,
        pred: Float[torch.Tensor, "ensemble batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        """Compute scaled CRPS loss.

        Args:
            pred: Ensemble predictions (ensemble_size, batch, hist*var, lat, lon)
            target: Ground truth (batch, hist*var, lat, lon)

        Returns:
            Scaled CRPS per channel (hist*var,)
        """
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = crps_ensemble(
            pred, target, self._wet
        )
        # Channels are time-major: (hist+1) * var.
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(-1, self._per_channel_scale.shape[0])
            * self._per_channel_scale
        )
        scaled_loss = scaled_loss_including_history_dimension.reshape(-1)
        
        # Add spread regularizer bonus (negative = encourages spread)
        # λ ramps down from spread_reg_lambda to 0 over spread_reg_steps
        if self._spread_reg_lambda > 0 and self._step < self._spread_reg_steps:
            # Current lambda (linear cooldown)
            progress = self._step / self._spread_reg_steps
            current_lambda = self._spread_reg_lambda * (1.0 - progress)
            self._last_spread_reg_lambda = current_lambda
            
            # Compute spread bonus: mean pairwise L1 distance
            # pred shape: (ensemble, batch, hist*var, lat, lon)
            ensemble_size = pred.shape[0]
            wet_float = self._wet.to(device=pred.device, dtype=pred.dtype)
            if wet_float.ndim == 2:
                wet_float = wet_float.unsqueeze(0)
            
            # Compute pairwise differences
            pred_i = pred.unsqueeze(1)  # (E, 1, B, C, H, W)
            pred_j = pred.unsqueeze(0)  # (1, E, B, C, H, W)
            pairwise_l1 = torch.abs(pred_i - pred_j)  # (E, E, B, C, H, W)
            
            # Mean over all pairs (excluding diagonal), batch, and spatial dims
            # Sum over ensemble pairs, normalize by M*(M-1)
            spread_per_channel = pairwise_l1.sum(dim=(0, 1)) / (ensemble_size * (ensemble_size - 1))
            spread_per_channel = (spread_per_channel * wet_float.unsqueeze(0)).mean(dim=(0, 2, 3))
            
            # Spread bonus (negative because we minimize loss)
            spread_bonus = current_lambda * spread_per_channel.mean()
            self._last_spread_bonus = spread_bonus.item()
            
            # Subtract spread bonus (encourages more spread)
            scaled_loss = scaled_loss - spread_bonus
        else:
            self._last_spread_reg_lambda = 0.0
            self._last_spread_bonus = 0.0
        
        return scaled_loss

    def step(self) -> None:
        """Increment step counter for spread regularizer cooldown."""
        self._step += 1

    def get_spread_reg_stats(self) -> dict[str, float]:
        """Return spread regularizer stats for WandB logging."""
        return {
            "spread_reg/lambda": self._last_spread_reg_lambda,
            "spread_reg/bonus": self._last_spread_bonus,
            "spread_reg/step": self._step,
        }

    def update(
        self,
        pred: Float[torch.Tensor, "ensemble batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> None:
        """Given the prediction & target for this step, update the per-channel scale.

        Args:
            pred: Ensemble predictions (ensemble_size, batch, hist*var, lat, lon)
            target: Ground truth (batch, hist*var, lat, lon)
        """
        crps_loss = crps_ensemble(pred, target, self._wet)
        crps_loss = torch.where(crps_loss == 0, 1e-8, crps_loss)
        new_target_weights_with_history: Float[torch.Tensor, " hist*var"] = (
            1.0 / crps_loss
        )
        # Reshape from channels * history to channels
        # by averaging along the `hist` dimension
        new_target_weights: Float[torch.Tensor, " var"] = (
            new_target_weights_with_history.reshape(
                -1, self._per_channel_scale.shape[0]
            ).mean(dim=0)
        )

        if get_world_size() > 1:
            all_reduce_mean(new_target_weights)

        if self._limit is not None:
            min_scale = new_target_weights.min()
            max_scale = min_scale * self._limit
            weights_before = new_target_weights.clone()
            new_target_weights = new_target_weights.clamp(min_scale, max_scale)

            # Store clamping statistics for WandB logging
            ratio = weights_before / min_scale
            clamped_mask = ratio > self._limit
            n_clamped = clamped_mask.sum().item()

            self._last_clamp_stats = {
                "crps_dynamic/min_scale": min_scale.item(),
                "crps_dynamic/max_scale": max_scale.item(),
                "crps_dynamic/channels_clamped": n_clamped,
                "crps_dynamic/max_ratio_before_clamp": ratio.max().item(),
                "crps_dynamic/mean_scale": new_target_weights.mean().item(),
                "crps_dynamic/scale_std": new_target_weights.std().item(),
            }

            # Add per-variable weights (unclamped and clamped)
            for i, var_name in enumerate(self._var_names):
                self._last_clamp_stats[f"crps_dynamic/weight_unclamped/{var_name}"] = (
                    weights_before[i].item()
                )
                self._last_clamp_stats[f"crps_dynamic/weight_clamped/{var_name}"] = (
                    new_target_weights[i].item()
                )

            # Brief debug log (not verbose arrays)
            if n_clamped > 0:
                logger.debug(
                    f"CRPS Dynamic clamping: {n_clamped}/{len(new_target_weights)} channels, "
                    f"max_ratio={ratio.max().item():.2f}"
                )
        else:
            self._last_clamp_stats = {
                "crps_dynamic/min_scale": new_target_weights.min().item(),
                "crps_dynamic/max_scale": new_target_weights.max().item(),
                "crps_dynamic/channels_clamped": 0,
                "crps_dynamic/max_ratio_before_clamp": (
                    new_target_weights.max() / new_target_weights.min()
                ).item(),
                "crps_dynamic/mean_scale": new_target_weights.mean().item(),
                "crps_dynamic/scale_std": new_target_weights.std().item(),
            }

            # Add per-variable weights (no clamping, unclamped == clamped)
            for i, var_name in enumerate(self._var_names):
                self._last_clamp_stats[f"crps_dynamic/weight_unclamped/{var_name}"] = (
                    new_target_weights[i].item()
                )
                self._last_clamp_stats[f"crps_dynamic/weight_clamped/{var_name}"] = (
                    new_target_weights[i].item()
                )

        self._per_channel_scale = (
            self._per_channel_scale * (CrpsDynamic.N_WINDOW - 1) + new_target_weights
        ) / CrpsDynamic.N_WINDOW

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {"per_channel_scale": self._per_channel_scale.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from ``state_dict``."""
        if "per_channel_scale" in state:
            self._per_channel_scale = state["per_channel_scale"].to(self._wet.device)
