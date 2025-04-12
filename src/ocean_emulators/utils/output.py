import torch
import xarray as xr


class TrainStepOutput:
    def __init__(self, loss: torch.Tensor, loss_per_channel: torch.Tensor):
        self.loss = loss
        self.loss_per_channel = loss_per_channel


class ValStepOutput(TrainStepOutput):
    def __init__(
        self,
        loss: torch.Tensor,
        loss_per_channel: torch.Tensor,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        gen_data: torch.Tensor,
    ):
        super().__init__(loss, loss_per_channel)
        assert target_data.shape == gen_data.shape
        self.input_data = input_data
        self.target_data = target_data
        self.gen_data = gen_data


class ModelInferenceOutput:
    def __init__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time: xr.DataArray,
    ):
        assert prediction.shape == target.shape
        self.prediction = prediction
        self.target = target
        self.time = time
