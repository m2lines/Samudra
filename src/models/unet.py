import torch
from hydra.utils import instantiate
from .modules.encoder import UNetEncoder
from .modules.decoder import UNetDecoder
from omegaconf import DictConfig
from typing import Sequence
import copy
from torchvision.transforms import Resize
from utils.climate_utils import pairwise
from .modules.blocks import AdamConvBlock
import torch.nn as nn
import numpy as np

class AdamUNet(torch.nn.Module):
    def __init__(self, ch_width, n_out, wet, kernel_size = 3, pad="circular"):
        super().__init__()
        assert kernel_size % 2 !=0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad
        self.pred_residuals=False
        self.output_channels = n_out


        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(AdamConvBlock(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(AdamConvBlock(b,b,pad=pad))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(AdamConvBlock(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(AdamConvBlock(b,b,pad=pad))
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))


        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)

        #self.layers = nn.ModuleList(layer)

    def forward_once(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,AdamConvBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return torch.mul(fts,self.wet)

    def forward(
        self,
        inputs,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:

        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = inputs[0]
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                        [inputs_0, inputs[2 * step][:, self.output_channels :]],
                        dim=1,
                    )


            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[:, : self.output_channels] + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

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
        inputs,
        num_steps=None,
        output_only_last=False,
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = inputs[0][0].unsqueeze(0)
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                        [
                            inputs_0.unsqueeze(0),
                            inputs[step][0][self.output_channels :].unsqueeze(0),
                        ],
                        dim=1,
                    )


            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[0, : self.output_channels] + decodings.squeeze(
                    0
                )  # Residual prediction
            else:
                reshaped = decodings.squeeze(0)  # Absolute prediction

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res

class UNet(torch.nn.Module):
    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        input_time_dim: int,
        output_time_dim: int,
        wet, 
        input_channels: int = 9,
        output_channels: int = 3,
        presteps: int = 0,
        pred_residuals: bool = True,
    ):
        super().__init__()
        assert input_time_dim == 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.presteps = presteps
        self.time_dim = 1
        self.pred_residuals = False
        self.wet = wet

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        if not self.pred_residuals:
            print("Using absolute predictions")

        # Build the model layers
        self.encoder = instantiate(
            encoder, input_channels=self._compute_input_channels()
        )
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(
            decoder, output_channels=self._compute_output_channels()
        )

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * self.input_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels
    
    def forward_once(self, inputs):
        encodings = self.encoder(inputs)
        decodings = self.decoder(encodings)
        decodings = torch.mul(decodings, self.wet)
        return decodings

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
                input_tensor = inputs[0]
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                        [inputs_0, inputs[2 * step][:, self.output_channels :]],
                        dim=1,
                    )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[:, : self.output_channels] + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

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
                input_tensor = inputs[0][0].unsqueeze(0)
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                        [
                            inputs_0.unsqueeze(0),
                            inputs[step][0][self.output_channels :].unsqueeze(0),
                        ],
                        dim=1,
                    )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[0, : self.output_channels] + decodings.squeeze(
                    0
                )  # Residual prediction
            else:
                reshaped = decodings.squeeze(0)  # Absolute prediction

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res
