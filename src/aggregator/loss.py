from typing import Dict, Optional

import torch

from constants import OUT_VARS, get_eval_maps


class LossAggregator:
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
    def init_instance(cls, exp_num_out: str) -> "LossAggregator":
        if cls._instance is not None:
            raise ValueError("LossAggregator already initialized")

        instance = super().__new__(cls)
        instance._initialize(exp_num_out)
        cls._instance = instance
        return cls._instance

    def _initialize(self, exp_num_out: str):
        self.VAR_3D_IDX, self.DP_3D_IDX, self.VAR_SET, self.DEPTH_SET = get_eval_maps(
            exp_num_out
        )
        self.outputs = OUT_VARS[exp_num_out]

    def get_depth_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for depth in self.DEPTH_SET:
            metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
                self.DP_3D_IDX[depth]
            ].mean()
        return metrics

    def get_variable_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for variable in self.VAR_SET:
            metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
                self.VAR_3D_IDX[variable]
            ].mean()
        return metrics

    def get_channel_loss_dict(
        self, label: str, loss_per_channel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        for i, channel in enumerate(self.outputs):
            metrics[f"{label}/loss/channel/{channel}_loss"] = loss_per_channel[i]
        return metrics
