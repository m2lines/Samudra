import torch

from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.multiton import Multiton


class LossAggregator(Multiton):
    """
    Aggregates loss metrics for different depths and variables.

    Note that this aggregator is a singleton in contrast to other aggregators
    that are initialized every epoch.
    """

    def _initialize(self, loss_scale: dict[str, float] = {}):
        self.tensor_map = TensorMap.get_instance()
        self.loss_scale = loss_scale

    def get_depth_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        for depth in self.tensor_map.DEPTH_SET:
            metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
                self.tensor_map.DP_3D_IDX[depth]
            ].mean()
        return metrics

    def get_variable_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        for variable in self.tensor_map.VAR_SET:
            metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
                self.tensor_map.VAR_3D_IDX[variable]
            ].mean()
        return metrics

    def get_channel_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.get_channel_dict(label, "loss", loss_per_channel)

    def get_channel_loss_scale_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.get_channel_dict(label, "loss_scale", loss_per_channel)

    def get_channel_dict(
        self, prefix: str, measure: str, per_channel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        metrics = {}
        for i, channel in enumerate(self.tensor_map.prognostic_var_names):
            metrics[f"{prefix}/{measure}/channel/{channel}_{measure}"] = per_channel[i]
        return metrics
