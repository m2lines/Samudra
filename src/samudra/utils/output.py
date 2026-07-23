# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
import xarray as xr

from samudra.utils.ctx import GridContext


class TrainBatchOutput:
    def __init__(self, loss: torch.Tensor, loss_per_channel: torch.Tensor):
        self.loss = loss
        self.loss_per_channel = loss_per_channel


class ValBatchOutput(TrainBatchOutput):
    def __init__(
        self,
        loss: torch.Tensor,
        loss_per_channel: torch.Tensor,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        gen_data: torch.Tensor,
        ctx: GridContext,
    ):
        super().__init__(loss, loss_per_channel)
        assert target_data.shape == gen_data.shape
        self.input_data = input_data
        self.target_data = target_data
        self.gen_data = gen_data
        self.ctx = ctx


class ModelInferenceOutput:
    def __init__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time: xr.DataArray,
        rollout_state: torch.Tensor,
    ):
        assert prediction.shape == target.shape
        self.prediction = prediction
        self.target = target
        self.time = time
        # State carried between inference chunks. This is deliberately model-defined:
        # physical-space models carry their final prediction, while latent models
        # carry the final latent without decoding and re-encoding it.
        self.rollout_state = rollout_state
