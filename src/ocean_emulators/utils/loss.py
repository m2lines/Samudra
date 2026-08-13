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
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard MSE loss (l2) computed per channel."""
    # Explicit pointwise arithmetic keeps ShardTensor on its native dispatch
    # path; F.mse_loss's generic DTensor fallback can introduce Partial layouts.
    mse = (pred - target).square()
    return _weighted_channel_mean(
        mse, wet=wet, spatial_weight=spatial_weight, extra_weight=sample_weight
    )


def decomposed_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard MAE loss (l1) computed per channel."""
    mae = (pred - target).abs()
    return _weighted_channel_mean(
        mae, wet=wet, spatial_weight=spatial_weight, extra_weight=sample_weight
    )


# TODO(alxmrs): This used to assume that hist=1; it may need to be fixed in the future.
def decomposed_mse_diff_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
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
        combined_loss, wet=wet, spatial_weight=spatial_weight, extra_weight=sample_weight
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
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined MSE and MAE loss."""
    error = pred - target
    mse = error.square()
    mae = error.abs()
    combined = (mse + mae) / 2
    return _weighted_channel_mean(
        combined, wet=wet, spatial_weight=spatial_weight, extra_weight=sample_weight
    )


def _spatial_gradients(
    tensor: torch.Tensor, *, pad_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward differences along y and x axes with configurable x padding."""
    grad_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="constant")

    padded_x = F.pad(tensor, (0, 1, 0, 0), mode=resolved_x_pad_mode(pad_mode))
    grad_x = padded_x[:, :, :, 1:] - padded_x[:, :, :, :-1]

    return grad_y, grad_x


def gradient_h_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    pad_mode: str,
    spatial_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss on horizontal spatial gradients, averaged per channel."""
    pred = pred * wet
    target = target * wet

    pred_grad_y, pred_grad_x = _spatial_gradients(pred, pad_mode=pad_mode)
    target_grad_y, target_grad_x = _spatial_gradients(target, pad_mode=pad_mode)

    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")

    grad_loss = (
        _weighted_channel_mean(
            grad_loss_y,
            wet=wet,
            spatial_weight=spatial_weight,
            extra_weight=sample_weight,
        )
        + _weighted_channel_mean(
            grad_loss_x,
            wet=wet,
            spatial_weight=spatial_weight,
            extra_weight=sample_weight,
        )
    ) / 2
    return grad_loss


def gradient_z_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    spatial_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss on normalized vertical derivatives, averaged per channel.

    Every 3D prognostic variable is penalized, taken from the tensor map rather
    than a hard-coded list, so adding one (W, say) to `PROGNOSTIC_VARS` pulls it
    into this term with no change here. 2D variables have no depth column and
    drop out on their own.

    Each level-to-level difference is divided by that pair's centre-to-centre
    spacing, making this a penalty on d/dz rather than on a bare per-level jump.
    That division is not cosmetic on LLC4320: the levels range from 1.07 m apart
    at the surface to 45.5 m apart at depth, so without it the same physical
    stratification error is charged 42x more where the grid is coarse, and the
    tightly stacked near-surface levels -- the ones free to drift apart -- come
    out the least constrained channels in the column. The 1/dz weights are then
    rescaled to average one, which leaves the relative weighting across depth
    intact while holding the size of the whole term, and so the meaning of
    `lambda_z`, where it was before.
    """
    tensor_map = TensorMap.get_instance()
    num_vars = len(tensor_map.prognostic_var_names)
    if pred.shape[1] % num_vars != 0:
        raise ValueError(
            "gradient_z expected time-major channels to be a multiple of "
            f"{num_vars}, got {pred.shape[1]}."
        )

    num_times = pred.shape[1] // num_vars
    pred_by_time = pred.reshape(pred.shape[0], num_times, num_vars, *pred.shape[-2:])
    target_by_time = target.reshape(
        target.shape[0], num_times, num_vars, *target.shape[-2:]
    )
    wet_by_time = wet.reshape(num_times, num_vars, *wet.shape[-2:]).bool()
    spatial_weight_by_time = (
        spatial_weight.reshape(num_times, num_vars, *spatial_weight.shape[-2:])
        if spatial_weight is not None
        else None
    )

    loss_by_time = torch.zeros(
        (num_times, num_vars), device=pred.device, dtype=pred.dtype
    )
    count_by_time = torch.zeros_like(loss_by_time)

    for variable in tensor_map.VAR_SET_3D:
        indices = tensor_map.VAR_3D_IDX[variable].to(
            device=pred.device, dtype=torch.long
        )
        if indices.numel() < 2:
            continue

        lower = indices[:-1]
        upper = indices[1:]
        # Weight each pair by the reciprocal of its spacing, rescaled so the
        # weights average exactly one. The reciprocal is what turns a per-level
        # jump into a d/dz; the rescaling is what keeps `lambda_z` on the scale
        # it already had, since 1/dz averages well above 1 in metres and would
        # otherwise inflate the whole term several-fold on its own.
        spacing = tensor_map.vertical_spacing(variable).to(
            device=pred.device, dtype=pred.dtype
        )
        pair_weight = spacing.reciprocal()
        pair_weight = (pair_weight / pair_weight.mean()).view(1, 1, -1, 1, 1)

        pred_grad_z = (
            pred_by_time[:, :, upper] - pred_by_time[:, :, lower]
        ) * pair_weight
        target_grad_z = (
            target_by_time[:, :, upper] - target_by_time[:, :, lower]
        ) * pair_weight
        # Explicit pointwise arithmetic rather than F.l1_loss, for the reason
        # given in decomposed_mse: it keeps ShardTensor on its native dispatch
        # path under domain parallelism.
        grad_loss = (pred_grad_z - target_grad_z).abs()

        midpoint_weight = (wet_by_time[:, upper] & wet_by_time[:, lower]).to(
            dtype=pred.dtype
        )
        if spatial_weight_by_time is not None:
            midpoint_weight = midpoint_weight * torch.minimum(
                spatial_weight_by_time[:, upper],
                spatial_weight_by_time[:, lower],
            ).to(dtype=pred.dtype)

        # Carrying the per-sample weight into BOTH sums is what keeps this a
        # weighted mean. The expand is load-bearing rather than cosmetic: the
        # denominator sums over the batch axis, so a size-1 axis here would
        # divide a batch-summed numerator by a single sample's cell count and
        # silently scale the whole term by the batch size. It is a free view.
        weight = midpoint_weight.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
        if sample_weight is not None:
            # A per-channel weight has to be split by depth pair exactly as the
            # wet mask is: a vertical difference spans two levels, so it is
            # valid only where BOTH of them are, hence the pairwise minimum.
            # A single-channel weight is a surface field and broadcasts as is.
            sample = sample_weight.to(dtype=weight.dtype)
            channels = sample.shape[1]
            if channels == num_times * num_vars:
                sample = sample.reshape(
                    sample.shape[0], num_times, num_vars, *sample.shape[-2:]
                )
                sample = torch.minimum(sample[:, :, upper], sample[:, :, lower])
            elif channels == 1:
                sample = sample.reshape(sample.shape[0], 1, 1, *sample.shape[-2:])
            else:
                raise ValueError(
                    f"sample_weight has {channels} channels; gradient_z needs "
                    f"either {num_times * num_vars} (one per prognostic channel) "
                    "or 1 (a surface field to broadcast)."
                )
            weight = weight * sample
        valid_cells = weight.sum(dim=(0, 3, 4))
        valid_pair = valid_cells > 0
        numerator = (grad_loss * weight).sum(dim=(0, 3, 4))
        denominator = valid_cells.clamp_min(1e-8)
        pair_loss = torch.where(valid_pair, numerator / denominator, 0.0)

        loss_by_time.index_add_(1, lower, pair_loss)
        loss_by_time.index_add_(1, upper, pair_loss)
        count_increment = valid_pair.to(dtype=pred.dtype)
        count_by_time.index_add_(1, lower, count_increment)
        count_by_time.index_add_(1, upper, count_increment)

    return torch.where(
        count_by_time > 0,
        loss_by_time / count_by_time.clamp_min(1.0),
        loss_by_time,
    ).reshape(-1)


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
    grad_loss = gradient_h_l1_loss(
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
            if var_name in {"U", "V", "W"}:
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
    """Combine a base loss with optional gradient matching penalties.

    Applies the provided per-channel loss metric, then optionally adds
    horizontal gradient_h and normalized vertical gradient_z penalties.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        *,
        wet: Grid,
        lambda_h: float = 0.0,
        lambda_z: float = 0.0,
        pad_mode: str,
        spatial_weight: torch.Tensor | None = None,
    ):
        self.loss_fn = loss_fn
        self._wet = wet
        self._lambda_h = lambda_h
        self._lambda_z = lambda_z
        self._pad_mode = pad_mode
        self._spatial_weight = spatial_weight

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
        sample_weight: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, " hist*var"]:
        """`sample_weight` is a per-sample `[batch, ..., lat, lon]` mask.

        It exists because a batch can now mix tiles with different land: the
        wet mask is a property of the sample, not of the run, and scoring one
        tile's ocean against another tile's mask is simply wrong.
        """
        base_loss = (
            self.loss_fn(pred, target)
            if sample_weight is None
            else self.loss_fn(pred, target, sample_weight=sample_weight)
        )
        total_loss = base_loss
        if self._lambda_h > 0:
            grad_h_loss = gradient_h_l1_loss(
                pred=pred,
                target=target,
                wet=self._wet,
                pad_mode=self._pad_mode,
                spatial_weight=self._spatial_weight,
                sample_weight=sample_weight,
            )
            total_loss = total_loss + self._lambda_h * grad_h_loss
        if self._lambda_z > 0:
            grad_z_loss = gradient_z_l1_loss(
                pred=pred,
                target=target,
                wet=self._wet,
                spatial_weight=self._spatial_weight,
                sample_weight=sample_weight,
            )
            total_loss = total_loss + self._lambda_z * grad_z_loss
        return total_loss
