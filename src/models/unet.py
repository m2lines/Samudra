import torch
from hydra.utils import instantiate
from .modules.encoder import UNetEncoder
from .modules.decoder import UNetDecoder
from omegaconf import DictConfig
from typing import Sequence
import copy
from torchvision.transforms import Resize


class BaseUNet(torch.nn.Module):
    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: int,
        output_channels: int,
        presteps: int,
    ):

        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.presteps = presteps

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.encoder = instantiate(
            encoder, input_channels=self._compute_input_channels()
        )
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(
            decoder, output_channels=self._compute_output_channels()
        )

        self.input_size_set = False

    def get_resize_fn(self, input_size):
        new_img_size = copy.deepcopy(input_size)
        if new_img_size[0] % 16 != 0:
            new_img_size[0] = (int(new_img_size[0] / 16) + 1) * 16

        if new_img_size[1] % 16 != 0:
            new_img_size[1] = (int(new_img_size[1] / 16) + 1) * 16
        return Resize(new_img_size)

    def set_input_size(self, input_size):
        self.input_size = input_size
        self.resize_fn = self.get_resize_fn(input_size)
        self.input_size_set = True

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * self.input_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape(self, input) -> torch.Tensor:
        B, T, C, H, W = input.shape
        input = input.reshape(B, T * C, H, W)
        return self.resize_fn(input)

    def _reshape_output(self, output) -> torch.Tensor:
        B, _, H, W = output.shape
        output = output.reshape(B, -1, self.output_channels, H, W)
        return output[:, :, :, : self.input_size[0], : self.input_size[1]]


class UNet(BaseUNet):
    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: int = 9,
        output_channels: int = 3,
        presteps: int = 0,
        pred_residuals: bool = True,
    ):

        super().__init__(
            encoder,
            decoder,
            input_time_dim,
            output_time_dim,
            input_channels,
            output_channels,
            presteps,
        )
        assert input_time_dim == 1
        self.time_dim = 1
        self.pred_residuals = pred_residuals
        if not self.pred_residuals:
            print("Using absolute predictions")

    def forward(
        self,
        inputs: Sequence,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:

        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = self.resize_fn(inputs[0])
            else:
                inputs_0 = outputs[-1]
                input_tensor = self.resize_fn(
                    torch.cat(
                        [inputs_0, inputs[2 * step][:, self.output_channels :]],
                        dim=1,
                    )
                )

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[:, : self.output_channels] + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

            reshaped = reshaped[:, :, : self.input_size[0], : self.input_size[1]]

            if loss_fn is not None:
                if loss is None:
                    loss = loss_fn(
                        reshaped,
                        inputs[2 * step + 1][:, : self.output_channels],
                    )
                else:
                    loss += loss_fn(
                        reshaped,
                        inputs[2 * step + 1][:, : self.output_channels],
                    )

            outputs.append(reshaped)

        if loss_fn is None:
            if output_only_last:
                res = outputs[-1]
            else:
                res = outputs
            return res

        else:
            return loss

    def inference(
        self,
        inputs: Sequence,
        num_steps=None,
        output_only_last=False,
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = self.resize_fn(inputs[0][0].unsqueeze(0))
            else:
                inputs_0 = outputs[-1]
                input_tensor = self.resize_fn(
                    torch.cat(
                        [
                            inputs_0.unsqueeze(0),
                            inputs[step][0][self.output_channels :].unsqueeze(0),
                        ],
                        dim=1,
                    )
                )

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            if self.pred_residuals:
                reshaped = input_tensor[0, : self.output_channels] + decodings.squeeze(
                    0
                )  # Residual prediction
            else:
                reshaped = decodings.squeeze(0)  # Absolute prediction

            reshaped = reshaped[:, : self.input_size[0], : self.input_size[1]]
            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res