from typing import Optional

import torch


def weighted_mean_with_nan_as_zero(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: tuple[int, ...] = (-2, -1),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim, keepdim=keepdim)

    weights = weights.to(tensor.device)
    return (tensor * weights).sum(dim=dim, keepdim=keepdim) / weights.expand(
        tensor.shape
    ).sum(dim=dim, keepdim=keepdim)


def weighted_mean(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: tuple[int, ...] = (-2, -1),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.nanmean(dim=dim, keepdim=keepdim)

    weights = weights.to(tensor.device)
    denom = torch.where(torch.isnan(tensor), 0.0, weights.expand(tensor.shape)).sum(
        dim=dim, keepdim=keepdim
    )
    return (tensor * weights).nansum(dim=dim, keepdim=keepdim) / denom


def area_weighted_mean(
    data: torch.Tensor,
    area_weights: torch.Tensor,
    dim: tuple[int, ...] = (-2, -1),
    keepdim: bool = False,
) -> torch.Tensor:
    return weighted_mean(data, area_weights, dim=dim, keepdim=keepdim)


def area_weighted_std(
    data: torch.Tensor,
    area_weights: torch.Tensor,
    keepdim: bool = False,
):
    return area_weighted_mean(
        (data - area_weighted_mean(data, area_weights, keepdim=True)) ** 2,
        area_weights,
        keepdim=keepdim,
    ).sqrt()


def area_weighted_rmse(
    target: torch.Tensor, gen: torch.Tensor, area_weights: torch.Tensor
) -> torch.Tensor:
    area_weights = area_weights.to(target.device)
    return torch.sqrt(area_weighted_mean((gen - target) ** 2, area_weights))


def area_weighted_mean_bias(
    target: torch.Tensor, gen: torch.Tensor, area_weights: torch.Tensor
) -> torch.Tensor:
    area_weights = area_weights.to(target.device)
    return area_weighted_mean(gen - target, area_weights)


def gradient_magnitude(
    tensor: torch.Tensor, dim: tuple[int, ...] = (-2, -1)
) -> torch.Tensor:
    """Compute the magnitude of gradient across the specified dimensions."""
    gradients = torch.gradient(tensor, dim=dim)
    return torch.sqrt(sum([g**2 for g in gradients]))


def weighted_mean_gradient_magnitude(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: tuple[int, ...] = (-2, -1),
) -> torch.Tensor:
    """Compute weighted mean of gradient magnitude across the specified dimensions."""
    return weighted_mean(gradient_magnitude(tensor, dim), weights=weights, dim=dim)


def gradient_magnitude_percent_diff(
    target: torch.Tensor,
    gen: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: tuple[int, ...] = (-2, -1),
) -> torch.Tensor:
    """Compute the percent difference of the weighted mean gradient magnitude across
    the specified dimensions.
    """
    target_grad_mag = weighted_mean_gradient_magnitude(target, weights, dim)
    gen_grad_mag = weighted_mean_gradient_magnitude(gen, weights, dim)
    return 100 * (gen_grad_mag - target_grad_mag) / target_grad_mag


def area_weighted_gradient_magnitude_percent_diff(
    target: torch.Tensor, gen: torch.Tensor, area_weights: torch.Tensor
):
    area_weights = area_weights.to(target.device)
    return gradient_magnitude_percent_diff(target, gen, weights=area_weights)
