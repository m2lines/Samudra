from typing import Dict, Optional

import torch

from constants import TensorMap


class LossAggregator:
    """
    Aggregates loss metrics for different depths and variables.

    Note that this aggregator is a singleton in contrast to other aggregators
    that are initialized every epoch.
    """

    _instance: Optional["LossAggregator"] = None

    def __new__(cls, *args, **kwargs) -> "LossAggregator":
        # Prevent direct instantiation
        raise TypeError(
            "LossAggregator cannot be instantiated directly. "
            "Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "LossAggregator":
        if cls._instance is None:
            raise ValueError("LossAggregator not initialized")
        return cls._instance

    @classmethod
    def init_instance(cls, loss_scale: Dict[str, float] = {}) -> "LossAggregator":
        if cls._instance is not None:
            raise ValueError("LossAggregator already initialized")

        instance = super().__new__(cls)
        instance._initialize(loss_scale)
        cls._instance = instance
        return cls._instance

    def _initialize(self, loss_scale: Dict[str, float]):
        self.tensor_map = TensorMap.get_instance()
        self.loss_scale = loss_scale

    def get_depth_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for depth in self.tensor_map.DEPTH_SET:
            metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
                self.tensor_map.OUT_DP_3D_IDX[depth]
            ].mean()
        return metrics

    def get_variable_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for variable in self.tensor_map.VAR_SET:
            metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
                self.tensor_map.OUT_VAR_3D_IDX[variable]
            ].mean()
        return metrics

    def get_channel_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for i, channel in enumerate(self.tensor_map.outputs):
            metrics[f"{label}/loss/channel/{channel}_loss"] = loss_per_channel[i]
        return metrics
