import torch
import torch.nn.functional as F


def decomposed_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss computed per channel."""
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mse_diff_weighted(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """MSE loss with weighted differences."""
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
    pred: torch.Tensor, target: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """MSE loss weighted by cosine of latitude."""
    weights = cos.view(1, 1, -1, 1)  # Reshape for broadcasting
    mse = F.mse_loss(pred, target, reduction="none")
    weighted_mse = mse * weights
    return weighted_mse.mean(dim=(0, 2, 3))


def decomposed_mse_scaled(
    pred: torch.Tensor, target: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    """MSE loss with scaled residuals."""
    scaled_pred = pred * scaling.view(1, -1, 1, 1)
    scaled_target = target * scaling.view(1, -1, 1, 1)
    return F.mse_loss(scaled_pred, scaled_target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mse_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined MSE and MAE loss."""
    mse = F.mse_loss(pred, target, reduction="none")
    mae = F.l1_loss(pred, target, reduction="none")
    combined = (mse + mae) / 2
    return combined.mean(dim=(0, 2, 3))
