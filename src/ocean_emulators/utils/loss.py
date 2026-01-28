from collections.abc import Callable
from functools import partial
from typing import Literal, assert_never

import torch
import torch.nn.functional as F
import xarray as xr
from jaxtyping import Float

from ocean_emulators.constants import Grid
from ocean_emulators.utils.data import _xr_to_torch

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossMetric = Literal[
    "mse",
    "mae",
    "mse_mae",
    "mse_diff_weighted",
    "mse_cos_weighted",
]


def loss_fn_from_metric(
    metric: LossMetric, *, wet: Grid, y_coord: xr.DataArray, device: torch.device
) -> LossFn:
    match metric:
        case "mse":
            loss_fn: LossFn = partial(decomposed_mse, wet=wet)
        case "mae":
            loss_fn = partial(decomposed_mae, wet=wet)
        case "mse_mae":
            loss_fn = partial(decomposed_mse_mae, wet=wet)
        case "mse_diff_weighted":
            loss_fn = partial(decomposed_mse_diff_weighted, wet=wet)
        case "mse_cos_weighted":
            lats = _xr_to_torch(y_coord, device=device, dtype=torch.float32)
            area_weights = torch.sqrt(torch.cos(torch.deg2rad(lats)))
            loss_fn = partial(decomposed_mse_cos_weighted, wet=wet, cos=area_weights)
        case _:
            assert_never(metric)
    return loss_fn


def decomposed_mse(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Standard MSE loss (l2) computed per channel."""
    pred = pred * wet
    target = target * wet
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mae(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Standard MAE loss (l1) computed per channel."""
    pred = pred * wet
    target = target * wet
    return F.l1_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


# TODO(alxmrs): This used to assume that hist=1; it may need to be fixed in the future.
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


def _spatial_gradients(
    tensor: torch.Tensor, *, pad_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward differences along y and x axes with configurable x padding."""
    grad_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="constant")

    padded_x = F.pad(tensor, (0, 1, 0, 0), mode=pad_mode)
    grad_x = padded_x[:, :, :, 1:] - padded_x[:, :, :, :-1]

    return grad_y, grad_x


def gradient_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, pad_mode: str
) -> torch.Tensor:
    """L1 loss on spatial gradients, averaged per channel."""
    pred = pred * wet
    target = target * wet

    pred_grad_y, pred_grad_x = _spatial_gradients(pred, pad_mode=pad_mode)
    target_grad_y, target_grad_x = _spatial_gradients(target, pad_mode=pad_mode)

    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")

    grad_loss = (grad_loss_y.mean(dim=(0, 2, 3)) + grad_loss_x.mean(dim=(0, 2, 3))) / 2
    return grad_loss


def decomposed_mae_gradient_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    gradient_weight: float,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """MAE loss with spatial gradient matching penalty."""
    mae_per_channel = decomposed_mae(pred, target, wet)
    grad_loss = gradient_l1_loss(pred, target, wet, pad_mode)
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
        stds: Float[torch.Tensor, " var"],
        *,
        should_limit: bool,
        device: torch.device,
    ):
        self.loss_fn = loss_fn
        self._device = device
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            stds.shape[0], device=self._device
        )
        if should_limit:
            variances: Float[torch.Tensor, " var"] = stds.pow(2)
            self._limits: Float[torch.Tensor, " var"] | None = 1.0 / variances
        else:
            self._limits = None

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = self.loss_fn(
            pred, target
        )
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(self._per_channel_scale.shape[0], -1)
            * self._per_channel_scale.unsqueeze(1)
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
                self._per_channel_scale.shape[0], -1
            ).mean(dim=1)
        )
        if self._limits is not None:
            new_target_weights = new_target_weights.min(self._limits)

        if get_world_size() > 1:
            all_reduce_mean(new_target_weights)

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
    ):
        self.loss_fn = loss_fn
        self._wet = wet
        self._gradient_weight = gradient_weight
        self._pad_mode = pad_mode

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        base_loss = self.loss_fn(pred, target)
        grad_loss = gradient_l1_loss(
            pred=pred, target=target, wet=self._wet, pad_mode=self._pad_mode
        )
        return base_loss + self._gradient_weight * grad_loss
