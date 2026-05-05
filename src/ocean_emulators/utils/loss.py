from collections.abc import Callable
from functools import partial
from typing import Literal, assert_never

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from jaxtyping import Float

from ocean_emulators.constants import Grid, TensorMap
from ocean_emulators.models.modules.padding import resolved_x_pad_mode

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossMetric = Literal[
    "mse",
    "mae",
    "mse_mae",
    "mse_diff_weighted",
    "mse_cos_weighted",
]


def loss_fn_from_metric(
    metric: LossMetric,
    *,
    wet: Grid,
    y_coord: xr.DataArray,
    device: torch.device,
    spatial_weight: torch.Tensor | None = None,
) -> LossFn:
    match metric:
        case "mse":
            loss_fn: LossFn = partial(
                decomposed_mse, wet=wet, spatial_weight=spatial_weight
            )
        case "mae":
            loss_fn = partial(decomposed_mae, wet=wet, spatial_weight=spatial_weight)
        case "mse_mae":
            loss_fn = partial(
                decomposed_mse_mae, wet=wet, spatial_weight=spatial_weight
            )
        case "mse_diff_weighted":
            loss_fn = partial(
                decomposed_mse_diff_weighted, wet=wet, spatial_weight=spatial_weight
            )
        case "mse_cos_weighted":
            area_weights = np.sqrt(np.cos(np.deg2rad(y_coord))).to_numpy()
            area_weights = torch.from_numpy(area_weights).to(device=device)
            loss_fn = partial(
                decomposed_mse_cos_weighted,
                wet=wet,
                cos=area_weights,
                spatial_weight=spatial_weight,
            )
        case _:
            assert_never(metric)
    return loss_fn


def build_halo_sponge_spatial_weight(
    wet: torch.Tensor,
    *,
    num_halo: int,
    num_sponge: int,
) -> torch.Tensor:
    """Build a simple edge weighting mask for the loaded LLC patch.

    The intended LLC workflow is:
    - the outer `num_halo` cells receive zero weight
    - the next `num_sponge` cells ramp linearly from low weight toward one
    - the remaining interior stays fully weighted
    """
    h, w = wet.shape[-2:]
    y = torch.arange(h, device=wet.device)
    x = torch.arange(w, device=wet.device)

    dist_y = torch.minimum(y, h - 1 - y).view(h, 1)
    dist_x = torch.minimum(x, w - 1 - x).view(1, w)
    boundary_distance = torch.minimum(dist_y, dist_x)

    spatial_weight = torch.ones((h, w), device=wet.device, dtype=torch.float32)
    if num_halo > 0:
        spatial_weight = torch.where(
            boundary_distance < num_halo,
            torch.zeros_like(spatial_weight),
            spatial_weight,
        )
    if num_sponge > 0:
        sponge_mask = (boundary_distance >= num_halo) & (
            boundary_distance < num_halo + num_sponge
        )
        sponge_weight = (
            boundary_distance.to(torch.float32) - num_halo + 1
        ) / (num_sponge + 1)
        spatial_weight = torch.where(sponge_mask, sponge_weight, spatial_weight)

    return spatial_weight.unsqueeze(0).expand_as(wet)


def _weighted_channel_mean(
    loss: torch.Tensor,
    *,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
    extra_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = wet.to(dtype=loss.dtype).unsqueeze(0)
    if spatial_weight is not None:
        weight = weight * spatial_weight.to(dtype=loss.dtype).unsqueeze(0)
    if extra_weight is not None:
        weight = weight * extra_weight.to(dtype=loss.dtype)

    numerator = (loss * weight).sum(dim=(0, 2, 3))
    denominator = weight.sum(dim=(0, 2, 3)).clamp_min(1e-8) * loss.shape[0]
    return numerator / denominator


def decomposed_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard MSE loss (l2) computed per channel."""
    mse = F.mse_loss(pred, target, reduction="none")
    return _weighted_channel_mean(mse, wet=wet, spatial_weight=spatial_weight)


def decomposed_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard MAE loss (l1) computed per channel."""
    mae = F.l1_loss(pred, target, reduction="none")
    return _weighted_channel_mean(mae, wet=wet, spatial_weight=spatial_weight)


# TODO(alxmrs): This used to assume that hist=1; it may need to be fixed in the future.
def decomposed_mse_diff_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
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
    return _weighted_channel_mean(
        combined_loss, wet=wet, spatial_weight=spatial_weight
    )


def decomposed_mse_cos_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    cos: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss weighted by cosine of latitude."""
    weights = cos.view(1, 1, -1, 1)  # Reshape for broadcasting
    mse = F.mse_loss(pred, target, reduction="none")
    return _weighted_channel_mean(
        mse,
        wet=wet,
        spatial_weight=spatial_weight,
        extra_weight=weights,
    )


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
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined MSE and MAE loss."""
    mse = F.mse_loss(pred, target, reduction="none")
    mae = F.l1_loss(pred, target, reduction="none")
    combined = (mse + mae) / 2
    return _weighted_channel_mean(combined, wet=wet, spatial_weight=spatial_weight)


def _spatial_gradients(
    tensor: torch.Tensor, *, pad_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward differences along y and x axes with configurable x padding."""
    grad_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    grad_y_pad_mode = "replicate" if pad_mode == "halo_sponge" else "constant"
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode=grad_y_pad_mode)

    padded_x = F.pad(tensor, (0, 1, 0, 0), mode=resolved_x_pad_mode(pad_mode))
    grad_x = padded_x[:, :, :, 1:] - padded_x[:, :, :, :-1]

    return grad_y, grad_x


def gradient_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    pad_mode: str,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss on spatial gradients, averaged per channel."""
    pred = pred * wet
    target = target * wet

    pred_grad_y, pred_grad_x = _spatial_gradients(pred, pad_mode=pad_mode)
    target_grad_y, target_grad_x = _spatial_gradients(target, pad_mode=pad_mode)

    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")

    grad_loss = (
        _weighted_channel_mean(grad_loss_y, wet=wet, spatial_weight=spatial_weight)
        + _weighted_channel_mean(grad_loss_x, wet=wet, spatial_weight=spatial_weight)
    ) / 2
    return grad_loss


def decomposed_mae_gradient_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    gradient_weight: float,
    pad_mode: str = "constant",
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MAE loss with spatial gradient matching penalty."""
    mae_per_channel = decomposed_mae(
        pred, target, wet, spatial_weight=spatial_weight
    )
    grad_loss = gradient_l1_loss(
        pred, target, wet, pad_mode, spatial_weight=spatial_weight
    )
    return mae_per_channel + gradient_weight * grad_loss


class DynamicLoss:
    """A loss function that scales each channel to contribute equally to the loss.

    This uses a rolling estimate of the loss of each channel to scale each
    channel's loss, discouraging the model from focusing on only a few channels.

    See: https://openathena.slack.com/archives/C08CYM42DT3/p1752275713570969
    """

    N_WINDOW = 25
    """Rolling window size to average over. (~number of steps)"""

    def __init__(
        self,
        loss_fn: LossFn,
        *,
        limit: float | None,
        device: torch.device,
        num_channels: int,
    ):
        self.loss_fn = loss_fn
        self._device = device
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            num_channels, device=self._device
        )
        self._limit = limit

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = self.loss_fn(
            pred, target
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
        # Local import is needed to prevent a circular import error.
        from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size

        loss = self.loss_fn(pred, target)
        loss = torch.where(loss == 0, 1e-8, loss)
        new_target_weights_with_history: Float[torch.Tensor, " hist*var"] = 1.0 / loss
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
            new_target_weights = new_target_weights.clamp(min_scale, max_scale)

        self._per_channel_scale = (
            self._per_channel_scale * (DynamicLoss.N_WINDOW - 1) + new_target_weights
        ) / DynamicLoss.N_WINDOW

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale

    # new methods for saving and loading state
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {"per_channel_scale": self._per_channel_scale.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from ``state_dict``."""
        if "per_channel_scale" in state:
            self._per_channel_scale = state["per_channel_scale"].to(self._device)


class WeightedLoss:
    """A loss wrapper with fixed per-channel weights."""

    def __init__(
        self,
        loss_fn: LossFn,
        *,
        device: torch.device,
        num_channels: int,
    ):
        self.loss_fn = loss_fn
        tensor_map = TensorMap.get_instance()
        if len(tensor_map.prognostic_var_names) != num_channels:
            raise ValueError(
                "WeightedLoss expected one static weight per prognostic channel."
            )

        weights = []
        for channel_name in tensor_map.prognostic_var_names:
            var_name = channel_name.split("_")[0]
            if var_name in {"U", "V"}:
                weights.append(1.0)
            elif var_name in {"Theta", "Salt", "Eta"}:
                weights.append(1.5)
            else:
                raise ValueError(
                    f"WeightedLoss does not have a default static weight for {var_name}."
                )
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.tensor(
            weights, device=device, dtype=torch.float32
        )

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = self.loss_fn(
            pred, target
        )
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(-1, self._per_channel_scale.shape[0])
            * self._per_channel_scale
        )
        return scaled_loss_including_history_dimension.reshape(-1)

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale


class GradientLoss:
    """Combine a base loss with a gradient matching penalty.

    Applies the provided per-channel loss metric then adds an L1 penalty on
    spatial gradients, scaled by ``gradient_weight``.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        *,
        wet: Grid,
        gradient_weight: float,
        pad_mode: str,
        spatial_weight: torch.Tensor | None = None,
    ):
        self.loss_fn = loss_fn
        self._wet = wet
        self._gradient_weight = gradient_weight
        self._pad_mode = pad_mode
        self._spatial_weight = spatial_weight

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        base_loss = self.loss_fn(pred, target)
        grad_loss = gradient_l1_loss(
            pred=pred,
            target=target,
            wet=self._wet,
            pad_mode=self._pad_mode,
            spatial_weight=self._spatial_weight,
        )
        return base_loss + self._gradient_weight * grad_loss
