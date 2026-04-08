import torch

from ocean_emulators.constants import TensorMap


def _index_on_tensor_device(index: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if index.device == tensor.device:
        return index
    return index.to(tensor.device, non_blocking=tensor.device.type == "cuda")


def get_depth_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    tensor_map: TensorMap,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for depth in tensor_map.DEPTH_SET:
        metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
            _index_on_tensor_device(tensor_map.DP_3D_IDX[depth], loss_per_channel)
        ].mean()
    return metrics


def get_variable_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    tensor_map: TensorMap,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for variable in tensor_map.VAR_SET:
        metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
            _index_on_tensor_device(tensor_map.VAR_3D_IDX[variable], loss_per_channel)
        ].mean()
    return metrics


def get_channel_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    tensor_map: TensorMap,
    loss_name: str = "loss",
) -> dict[str, torch.Tensor]:
    return get_channel_dict(label, loss_name, loss_per_channel, tensor_map=tensor_map)


def get_channel_loss_scale_dict(
    label: str,
    loss_scale_per_channel: torch.Tensor,
    *,
    tensor_map: TensorMap,
) -> dict[str, torch.Tensor]:
    return get_channel_dict(
        label,
        "loss_scale",
        loss_scale_per_channel,
        tensor_map=tensor_map,
    )


def get_channel_dict(
    prefix: str,
    measure: str,
    per_channel: torch.Tensor,
    *,
    tensor_map: TensorMap,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for i, channel in enumerate(tensor_map.prognostic_var_names):
        metrics[f"{prefix}/{measure}/channel/{channel}_{measure}"] = per_channel[i]
    return metrics
