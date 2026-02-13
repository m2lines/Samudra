from collections.abc import Callable
from typing import Literal, Protocol, assert_never

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ocean_emulators.utils.ctx import GridContext

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


def _elementwise_loss_from_metric(
    metric: LossMetric,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    match metric:
        case "mse":
            return F.mse_loss(pred, target, reduction="none")
        case "mae":
            return F.l1_loss(pred, target, reduction="none")
        case "mse_mae":
            mse = F.mse_loss(pred, target, reduction="none")
            mae = F.l1_loss(pred, target, reduction="none")
            return (mse + mae) / 2
        case "mse_diff_weighted":
            mse = F.mse_loss(pred, target, reduction="none")
            diff_weight = 2.0
            diff_mse = (
                F.mse_loss(
                    pred[:, 1:] - pred[:, :-1],
                    target[:, 1:] - target[:, :-1],
                    reduction="none",
                )
                * diff_weight
            )
            return torch.cat([mse[:, :1], diff_mse], dim=1)
        case _:
            assert_never(metric)


def _masked_elementwise_loss_from_metric(
    metric: LossMetric,
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: GridContext,
) -> torch.Tensor:
    wet = ctx.label_mask.to(device=pred.device)
    pred = pred * wet
    target = target * wet
    return _elementwise_loss_from_metric(metric, pred, target)


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
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        ctx: GridContext,
    ) -> None:
        """Given the prediction & target for this step, update the per-channel scale."""
        # Local import is needed to prevent a circular import error.
        from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size

        loss = self.loss_fn(pred, target, ctx)
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


class SpatialDynamicLoss:
    """Dynamic loss scaling with per-channel spatial scale maps.

    Scales each channel by a learned spatial map that is updated from inverse loss
    estimates with an EMA. Scale maps are maintained at pooled spatial resolution
    and interpolated to current resolution at runtime.
    """

    def __init__(
        self,
        *,
        metric: LossMetric,
        limit: float | None,
        device: torch.device,
        num_channels: int,
        ema_window: int,
        spatial_resolution_lat: float,
    ):
        self._metric = metric
        self._limit = limit
        self._device = device
        self._num_channels = num_channels
        self._ema_window = ema_window
        self._spatial_resolution_lat = spatial_resolution_lat
        self._epsilon = 1e-8
        self._per_channel_scale_map: Float[torch.Tensor, "var lat lon"] | None = None

        self._collect_batch_unscaled = False
        self._batch_unscaled_loss_with_history: torch.Tensor | None = None
        self._last_unscaled_loss_with_history: torch.Tensor | None = None

    def _initialize_scale_map(self, lat: int, lon: int) -> None:
        if self._per_channel_scale_map is not None:
            return
        pooled_lat = max(1, int(round(180.0 / self._spatial_resolution_lat)))
        pooled_lat = min(pooled_lat, lat)
        pooled_lon = max(1, int(round(pooled_lat * lon / lat)))
        pooled_lon = min(pooled_lon, lon)
        self._per_channel_scale_map = torch.ones(
            (self._num_channels, pooled_lat, pooled_lon),
            device=self._device,
        )

    def _loss_with_history_and_maps(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        ctx: GridContext,
    ) -> tuple[
        Float[torch.Tensor, "batch hist var lat lon"],
        Float[torch.Tensor, "batch hist var lat lon"],
    ]:
        batch, channels, lat, lon = pred.shape
        self._initialize_scale_map(lat=lat, lon=lon)
        assert self._per_channel_scale_map is not None
        if channels % self._num_channels != 0:
            raise ValueError(
                f"Expected channels to be divisible by {self._num_channels}, "
                f"got {channels}"
            )
        hist = channels // self._num_channels

        loss_map = _masked_elementwise_loss_from_metric(
            self._metric,
            pred=pred,
            target=target,
            ctx=ctx,
        )
        loss_map = loss_map.reshape(batch, hist, self._num_channels, lat, lon)

        scale_map = F.interpolate(
            self._per_channel_scale_map.unsqueeze(0),
            size=(lat, lon),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        scale_map = scale_map.to(dtype=loss_map.dtype)
        scaled_loss_map = loss_map * scale_map.unsqueeze(0).unsqueeze(0)

        return loss_map, scaled_loss_map

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        ctx: GridContext,
    ) -> Float[torch.Tensor, " hist*var"]:
        unscaled_loss_map, scaled_loss_map = self._loss_with_history_and_maps(
            pred, target, ctx
        )
        unscaled_loss_with_history = unscaled_loss_map.mean(dim=(0, 3, 4)).reshape(-1)
        scaled_loss_with_history = scaled_loss_map.mean(dim=(0, 3, 4)).reshape(-1)

        if self._collect_batch_unscaled:
            if self._batch_unscaled_loss_with_history is None:
                self._batch_unscaled_loss_with_history = (
                    unscaled_loss_with_history.detach().clone()
                )
            else:
                self._batch_unscaled_loss_with_history += (
                    unscaled_loss_with_history.detach()
                )

        return scaled_loss_with_history

    def start_batch(self) -> None:
        self._collect_batch_unscaled = True
        self._batch_unscaled_loss_with_history = None

    def end_batch(self) -> None:
        self._collect_batch_unscaled = False
        self._last_unscaled_loss_with_history = self._batch_unscaled_loss_with_history
        self._batch_unscaled_loss_with_history = None

    def update(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        ctx: GridContext,
    ) -> None:
        from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size

        loss_map, _ = self._loss_with_history_and_maps(pred, target, ctx)
        assert self._per_channel_scale_map is not None

        mean_loss_map = loss_map.mean(dim=(0, 1))
        pooled_hw = (
            int(self._per_channel_scale_map.shape[-2]),
            int(self._per_channel_scale_map.shape[-1]),
        )
        pooled_loss_map = F.adaptive_avg_pool2d(
            mean_loss_map.unsqueeze(0), pooled_hw
        ).squeeze(0)
        pooled_loss_map = pooled_loss_map.clamp_min(self._epsilon)

        new_target_weights = 1.0 / pooled_loss_map

        if get_world_size() > 1:
            all_reduce_mean(new_target_weights)

        if self._limit is not None:
            min_scale = new_target_weights.min()
            max_scale = min_scale * self._limit
            new_target_weights = new_target_weights.clamp(min_scale, max_scale)

        self._per_channel_scale_map = (
            self._per_channel_scale_map * (self._ema_window - 1) + new_target_weights
        ) / self._ema_window

    def last_unscaled_loss_per_channel(self) -> Float[torch.Tensor, " var"] | None:
        if self._last_unscaled_loss_with_history is None:
            return None
        return self._last_unscaled_loss_with_history.reshape(
            -1, self._num_channels
        ).mean(dim=0)

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        if self._per_channel_scale_map is None:
            return torch.ones(self._num_channels, device=self._device)
        return self._per_channel_scale_map.mean(dim=(1, 2))

    def loss_scale_std_per_channel(self) -> Float[torch.Tensor, " var"]:
        if self._per_channel_scale_map is None:
            return torch.zeros(self._num_channels, device=self._device)
        return self._per_channel_scale_map.std(dim=(1, 2), unbiased=False)

    def state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {
            "ema_window": torch.tensor(self._ema_window),
            "spatial_resolution_lat": torch.tensor(self._spatial_resolution_lat),
        }
        if self._per_channel_scale_map is not None:
            state["per_channel_scale_map"] = self._per_channel_scale_map.detach().cpu()
        return state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "ema_window" in state:
            self._ema_window = int(state["ema_window"].item())
        if "spatial_resolution_lat" in state:
            self._spatial_resolution_lat = float(state["spatial_resolution_lat"].item())
        if "per_channel_scale_map" in state:
            self._per_channel_scale_map = state["per_channel_scale_map"].to(
                self._device
            )


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
