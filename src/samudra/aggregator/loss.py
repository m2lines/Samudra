# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.constants import DataLayout


def get_depth_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    data_layout: DataLayout,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for depth in data_layout.depths:
        metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
            data_layout.depth_indices[depth]
        ].mean()
    return metrics


def get_variable_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    data_layout: DataLayout,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for variable in data_layout.variables:
        metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
            data_layout.variable_indices[variable]
        ].mean()
    return metrics


def get_channel_loss_dict(
    label: str,
    loss_per_channel: torch.Tensor,
    *,
    data_layout: DataLayout,
    loss_name: str = "loss",
) -> dict[str, torch.Tensor]:
    return get_channel_dict(label, loss_name, loss_per_channel, data_layout=data_layout)


def get_channel_loss_scale_dict(
    label: str,
    loss_scale_per_channel: torch.Tensor,
    *,
    data_layout: DataLayout,
) -> dict[str, torch.Tensor]:
    return get_channel_dict(
        label,
        "loss_scale",
        loss_scale_per_channel,
        data_layout=data_layout,
    )


def get_channel_dict(
    prefix: str,
    measure: str,
    per_channel: torch.Tensor,
    *,
    data_layout: DataLayout,
) -> dict[str, torch.Tensor]:
    metrics = {}
    for i, channel in enumerate(data_layout.prognostic_var_names):
        metrics[f"{prefix}/{measure}/channel/{channel}_{measure}"] = per_channel[i]
    return metrics
