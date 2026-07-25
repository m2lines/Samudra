# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal, Protocol, assert_never

import torch
import torch.nn.functional as F
from jaxtyping import Float

from samudra.constants import TensorMap
from samudra.utils.ctx import GridContext

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossFnWithContext(Protocol):
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: GridContext,
    ) -> torch.Tensor: ...


LossMetric = Literal[
    "mse",
    "mae",
    "mse_mae",
    "mse_diff_weighted",
]


def loss_fn_from_metric(metric: LossMetric) -> LossFnWithContext:
    match metric:
        case "mse":
            loss_fn: LossFn = decomposed_mse
        case "mae":
            loss_fn = decomposed_mae
        case "mse_mae":
            loss_fn = decomposed_mse_mae
        case "mse_diff_weighted":
            loss_fn = decomposed_mse_diff_weighted
        case _:
            assert_never(metric)

    def loss_fn_with_ctx(
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: GridContext,
    ) -> torch.Tensor:
        wet = ctx.label_mask.to(device=pred.device)
        pred = pred * wet
        target = target * wet
        return loss_fn(pred, target)

    return loss_fn_with_ctx


def decomposed_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss (l2) computed per channel."""
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MAE loss (l1) computed per channel."""
    return F.l1_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


# TODO(alxmrs): This used to assume that hist=1; it may need to be fixed in the future.
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
    pred: torch.Tensor, target: torch.Tensor, pad_mode: str
) -> torch.Tensor:
    """L1 loss on spatial gradients, averaged per channel."""
    pred_grad_y, pred_grad_x = _spatial_gradients(pred, pad_mode=pad_mode)
    target_grad_y, target_grad_x = _spatial_gradients(target, pad_mode=pad_mode)

    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")

    grad_loss = (grad_loss_y.mean(dim=(0, 2, 3)) + grad_loss_x.mean(dim=(0, 2, 3))) / 2
    return grad_loss


def decomposed_mae_gradient_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    gradient_weight: float,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """MAE loss with spatial gradient matching penalty."""
    mae_per_channel = decomposed_mae(pred, target)
    grad_loss = gradient_l1_loss(pred, target, pad_mode)
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
        loss_fn: LossFnWithContext,
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
        ctx: GridContext,
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = self.loss_fn(
            pred, target, ctx
        )
        # Channels are time-major: (hist+1) * var.
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(-1, self._per_channel_scale.shape[0])
            * self._per_channel_scale
        )
        return scaled_loss_including_history_dimension.reshape(-1)

    def update(
        self,
        loss_per_channel: Float[torch.Tensor, " hist*var"],
    ) -> None:
        """Given the unscaled per-channel loss, update the per-channel scale."""
        # Local import is needed to prevent a circular import error.
        from samudra.utils.distributed import all_reduce_mean, get_world_size

        loss = loss_per_channel.detach()
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


class GradientLoss:
    """Combine a base loss with a gradient matching penalty.

    Applies the provided per-channel loss metric then adds an L1 penalty on
    spatial gradients, scaled by ``gradient_weight``.
    """

    def __init__(
        self,
        loss_fn: LossFnWithContext,
        *,
        gradient_weight: float,
        pad_mode: str,
    ):
        self.loss_fn = loss_fn
        self._gradient_weight = gradient_weight
        self._pad_mode = pad_mode

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        ctx: GridContext,
    ) -> Float[torch.Tensor, " hist*var"]:
        base_loss = self.loss_fn(pred, target, ctx)
        # Ensure mask is on the same device as pred for gradient computation
        wet = ctx.label_mask.to(device=pred.device)
        pred = pred * wet
        target = target * wet
        grad_loss = gradient_l1_loss(pred=pred, target=target, pad_mode=self._pad_mode)
        return base_loss + self._gradient_weight * grad_loss


class OtterWeightedRmseLoss:
    """Otter's latitude-weighted RMSE with ocean-column channel weights.

    Every physical prognostic variable receives equal total default weight.
    Depth-resolved channels divide that weight according to normalized layer
    thickness (the ocean analogue of Otter's pressure-level weighting), while
    surface-only variables receive weight one. Explicit variable multipliers
    are applied before the weighted channel RMSEs are summed.
    """

    average_over_rollout = True

    def __init__(
        self,
        tensor_map: TensorMap,
        *,
        variable_weights: dict[str, float] | None = None,
    ) -> None:
        variable_weights = variable_weights or {}
        unknown = set(variable_weights) - set(tensor_map.VAR_SET)
        if unknown:
            raise ValueError(
                "Unknown Otter loss variable weights: " + ", ".join(sorted(unknown))
            )

        weights = torch.zeros(len(tensor_map.prognostic_var_names), dtype=torch.float32)
        for variable in tensor_map.VAR_SET:
            indices = tensor_map.VAR_3D_IDX[variable].to(dtype=torch.long, device="cpu")
            multiplier = variable_weights.get(variable, 1.0)
            if multiplier < 0:
                raise ValueError("Otter loss variable weights must be non-negative.")
            if variable in tensor_map.VAR_SET_2D:
                weights[indices] = multiplier
                continue

            level_indices = [
                int(tensor_map.prognostic_var_names[index].rsplit("_", 1)[1])
                for index in indices.tolist()
            ]
            thickness = torch.tensor(
                [
                    tensor_map.dataset_spec.depth_thickness[level]
                    for level in level_indices
                ],
                dtype=torch.float32,
            )
            weights[indices] = multiplier * thickness / thickness.sum()

        if not torch.any(weights > 0):
            raise ValueError("At least one Otter loss channel weight must be positive.")
        self._channel_weights = weights

    @staticmethod
    def reduce(loss_per_channel: torch.Tensor) -> torch.Tensor:
        """Match Otter's sum over weighted per-channel RMSE values."""
        return loss_per_channel.sum()

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch output*var lat lon"],
        target: Float[torch.Tensor, "batch output*var lat lon"],
        ctx: GridContext,
    ) -> Float[torch.Tensor, " output*var"]:
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {pred.shape} != {target.shape}"
            )
        num_channels = self._channel_weights.numel()
        if pred.shape[1] % num_channels:
            raise ValueError(
                f"{pred.shape[1]} output channels are not a multiple of the "
                f"{num_channels} configured prognostic channels."
            )

        latitude = ctx.output_resolution_cpu[0].to(device=pred.device, dtype=pred.dtype)
        if latitude.ndim == 1:
            latitude = latitude[:, None]
        latitude_weights = torch.cos(torch.deg2rad(latitude)).clamp_min(0)

        wet = ctx.label_mask.to(device=pred.device)
        if wet.shape[0] != pred.shape[1]:
            raise ValueError(
                f"Label mask has {wet.shape[0]} channels but loss has "
                f"{pred.shape[1]} channels."
            )
        spatial_weights = wet * latitude_weights
        squared_error = (pred - target).square()
        numerator = (squared_error * spatial_weights).sum(dim=(0, 2, 3))
        denominator = spatial_weights.sum(dim=(1, 2)) * pred.shape[0]
        mse = numerator / denominator.clamp_min(1)
        # Keeping epsilon inside the square root preserves an exact zero while
        # avoiding the undefined derivative of sqrt(mse) at a perfect forecast.
        epsilon = 1e-12
        rmse = torch.sqrt(mse + epsilon) - torch.sqrt(
            torch.tensor(epsilon, device=mse.device, dtype=mse.dtype)
        )

        repeats = pred.shape[1] // num_channels
        channel_weights = self._channel_weights.to(
            device=pred.device, dtype=pred.dtype
        ).repeat(repeats)
        return rmse * channel_weights
