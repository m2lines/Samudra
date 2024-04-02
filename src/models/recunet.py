import torch
import torch.nn as nn
import torch.nn.functional as F

# from typing import Any, Dict, Optional, Sequence, Union
from typing import Sequence
from utils.train_utils import SmoothedValue, MetricLogger

from utils.dist_utils import all_reduce_mean
import wandb
from torch.cuda import amp
from torchvision.transforms import Resize
import copy


class CappedGELU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.
    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        # self.cap = torch.tensor(cap_value, dtype=torch.float32)
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class TransposedConvUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
            )  # check padding
        )

        if activation is not None:
            upsampler.append(activation)
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class BasicConvBlock(torch.nn.Module):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                torch.nn.Conv2d(
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding="same",
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        return self.convblock(x)


class ConvGRUBlock(torch.nn.Module):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 1,
        downscale_factor: int = 4,
    ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = torch.nn.Conv2d(
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv_can = torch.nn.Conv2d(
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
        )
        self.h = torch.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = torch.zeros_like(inputs)
        combined = torch.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        self.h = torch.zeros_like(self.h)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in Figure 4 of https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        # Convolution block
        convblock = []
        # 7x7 convolution increasing channels
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        )
        # LayerNorm
        # convblock.append(th.nn.LayerNorm([out_channels*upscale_factor, HW, HW]))
        if activation is not None:
            convblock.append(activation)
        # 1x1 convolution decreasing channels
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        )
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)


class UNetEncoder(torch.nn.Module):
    """
    Generic UNet3Encoder that can be applied to arbitrary meshes.
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_channels: Sequence = (136, 68, 34),
        n_layers: Sequence = (2, 2, 1),
        dilations: list = (1, 2, 4),
    ):
        super().__init__()
        self.n_channels = n_channels

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        # Build encoder
        old_channels = input_channels
        self.encoder = []
        for n, curr_channel in enumerate(n_channels):
            modules = list()
            if n > 0:
                modules.append(AvgPool(2))
            else:
                down_pool_module = None

            modules.append(
                ConvNeXtBlock(
                    in_channels=old_channels,
                    latent_channels=curr_channel,
                    out_channels=curr_channel,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                )
            )

            old_channels = curr_channel

            self.encoder.append(torch.nn.Sequential(*modules))

        self.encoder = torch.nn.ModuleList(self.encoder)

    def forward(self, inputs: Sequence) -> Sequence:
        outputs = []
        for layer in self.encoder:
            outputs.append(layer(inputs))
            inputs = outputs[-1]
        return outputs

    def reset(self):
        pass


class UNetDecoder(torch.nn.Module):
    """
    Generic UNetDecoder that can be applied to arbitrary meshes.
    """

    def __init__(
        self,
        n_channels: Sequence = (34, 68, 136),
        n_layers: Sequence = (1, 2, 2),
        output_channels: int = 3,
        dilations: list = (4, 2, 1),
        use_rec=True,
    ):
        super().__init__()
        self.channel_dim = 1  # 1 in previous layout

        # if enable_nhwc and activation is not None:
        #     activation = activation.to(memory_format=torch.channels_last)

        if dilations is None:
            # Defaults to [1, 1, 1...] in accordance with the number of unet levels
            dilations = [1 for _ in range(len(n_channels))]

        self.decoder = []
        for n, curr_channel in enumerate(n_channels):

            # Second half of the synoptic layer does not need an upsampling module
            if n == 0:
                up_sample_module = None
            else:
                up_sample_module = TransposedConvUpsample(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                )

            next_channel = (
                n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            )

            conv_module = ConvNeXtBlock(
                in_channels=(
                    curr_channel * 2 if n > 0 else curr_channel
                ),  # Considering skip connection
                latent_channels=curr_channel,
                out_channels=next_channel,
                dilation=dilations[n],
                n_layers=n_layers[n],
            )

            # Recurrent module
            if use_rec:
                rec_module = ConvGRUBlock(in_channels=next_channel)
            else:
                rec_module = None

            self.decoder.append(
                torch.nn.ModuleDict(
                    {
                        "upsamp": up_sample_module,
                        "conv": conv_module,
                        "recurrent": rec_module,
                    }
                )
            )

        self.decoder = torch.nn.ModuleList(self.decoder)

        # (Linear) Output layer
        self.output_layer = BasicConvBlock(
            in_channels=curr_channel,
            out_channels=output_channels,
            dilation=dilations[-1],
            activation=None,
        )

    def forward(self, inputs: Sequence) -> torch.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                x = torch.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
            if layer["recurrent"] is not None:
                x = layer["recurrent"](x)
        return self.output_layer(x)

    def reset(self):
        for layer in self.decoder:
            layer["recurrent"].reset()


class NoRecUNetSimple(torch.nn.Module):
    def __init__(
        self,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: int = 9,
        output_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.time_dim = 1

        assert input_time_dim == 1

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.encoder = UNetEncoder(input_channels=self._compute_input_channels())
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = UNetDecoder(
            output_channels=self._compute_output_channels(), use_rec=False
        )

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * self.input_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def forward(self, x) -> torch.Tensor:
        return x[:, : self.output_channels] + self.decoder(self.encoder(x))
        # return self.decoder(self.encoder(x))

class NoRecUNet(torch.nn.Module):
    def __init__(
        self,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: int = 9,
        output_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 1

        assert input_time_dim == 1

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.encoder = UNetEncoder(input_channels=self._compute_input_channels())
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = UNetDecoder(
            output_channels=self._compute_output_channels(), use_rec=False
        )

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * self.input_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def forward(
        self,
        inputs: Sequence,
        resize_fn,
        final_img_size,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:

        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = resize_fn(inputs[0])
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                    [inputs_0, resize_fn(inputs[2 * step][:, self.output_channels :])],
                    dim=1,
                )

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            #reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            reshaped = input_tensor[:, :self.output_channels] + decodings  # Residual prediction
            
            if loss_fn is not None:
                if loss is None:
                    loss = loss_fn(reshaped[:, :, :final_img_size[0], :final_img_size[1]], inputs[2*step+1][:, :self.output_channels])
                else:
                    loss += loss_fn(reshaped[:, :, :final_img_size[0], :final_img_size[1]], inputs[2*step+1][:, :self.output_channels])

            outputs.append(reshaped)

        if loss_fn is None:
            if output_only_last:
                res = outputs[-1][:, :, :final_img_size[0], :final_img_size[1]]
            else:
                res = [output[:, :, :final_img_size[0], :final_img_size[1]] for output in outputs]

            return res
        else:
            return loss
    
    def inference(self, inputs: Sequence, resize_fn, final_img_size, num_steps=None, output_only_last=False) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = resize_fn(inputs[0][0].unsqueeze(0))
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                    [
                        inputs_0.unsqueeze(0),
                        resize_fn(inputs[step][0][self.output_channels :].unsqueeze(0)),
                    ],
                    dim=1,
                )

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            # reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            reshaped = input_tensor[0, : self.output_channels] + decodings.squeeze(
                0
            )  # Residual prediction

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res


class RecUNet(torch.nn.Module):
    def __init__(
        self,
        input_time_dim: int,
        output_time_dim: int,
        input_channels: int = 9,
        output_channels: int = 3,
        presteps: int = 1,
        reset_cycle: int = 2,
        verbose=False,
    ):
        """
        Deep Learning Weather Prediction (DLWP) recurrent UNet model on the HEALPix mesh.

        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param reset_cycle: hours after which the recurrent states are reset to zero and re-initialized. Set np.infty
            to never reset the hidden states.
        :param presteps: number of model steps to initialize recurrent states.
        """
        super().__init__()
        self.time_dim = 1

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.reset_cycle = reset_cycle
        self.presteps = presteps
        self.verbose = verbose

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(
                f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                f"{self.output_time_dim} and {self.input_time_dim})"
            )

        # Build the model layers
        self.encoder = UNetEncoder(input_channels=self._compute_input_channels())
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = UNetDecoder(output_channels=self._compute_output_channels())

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
        return max(self.output_time_dim // self.input_time_dim, 1)  # + self.presteps

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

    def _initialize_hidden(
        self, inputs: Sequence, outputs: Sequence, step: int, local_step: int
    ) -> None:
        self.reset()
        for prestep in range(self.presteps):
            if step < self.presteps:
                s = step + prestep
                input_tensor = self._reshape(
                    inputs[:, s * self.input_time_dim : (s + 1) * self.input_time_dim]
                )  # Assuming B, T, C, H, W
                if self.verbose:
                    print(
                        f"Initialize Hidden: Using input indices: {s*self.input_time_dim} : {(s+1)*self.input_time_dim} to produce (not saved) output at indices: {(s+1)*self.input_time_dim} : {(s+2)*self.input_time_dim}"
                    )
            else:
                s_ = step - self.presteps + prestep
                s = local_step - self.presteps + prestep
                input_tensor = self._reshape(
                    torch.cat(
                        [
                            outputs[s_ - 1],
                            inputs[
                                :,
                                (s + 1)
                                * self.input_time_dim : (s + 2)
                                * self.input_time_dim,
                                self.output_channels :,
                            ],
                        ],
                        dim=2,
                    )
                )
                if self.verbose:
                    print(
                        f"Initialize Hidden: Using output indices: {(s_+self.presteps)*self.input_time_dim}:{(s_+1+self.presteps)*self.input_time_dim} to produce (not saved) output at indices: {(s_+1+self.presteps)*self.input_time_dim}:{(s_+2+self.presteps)*self.input_time_dim}"
                    )
                    print(
                        f"Also, Concatenating extra input in range of local indices {(s+1)*self.input_time_dim}:{(s+2)*self.input_time_dim} and global indices {(s_+1)*self.input_time_dim}:{(s_+2)*self.input_time_dim}"
                    )

            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor))

    # [B, T, C, H, W]
    def forward(
        self,
        inputs: Sequence,
        last_outputs: Sequence = None,
        inference=False,
        output_only_last=False,
    ) -> torch.Tensor:
        assert self.input_size_set
        if last_outputs is not None:
            assert inference
            outputs = last_outputs
            step_range = range(len(outputs), len(outputs) + self.integration_steps)
        else:
            self.reset()
            outputs = []
            step_range = range(self.integration_steps)

        for local_step, step in enumerate(step_range):  # use local_step for all inputs
            if self.verbose:
                print(f"Step: {step}")
            # (Re-)initialize recurrent hidden states
            if step % self.reset_cycle == 0:
                if self.verbose:
                    print(f"Reinitializing at Step: {step}")
                self._initialize_hidden(
                    inputs=inputs, outputs=outputs, step=step, local_step=local_step
                )

            if step == 0:
                s = self.presteps
                input_tensor = self._reshape(
                    inputs[:, s * self.input_time_dim : (s + 1) * self.input_time_dim]
                )
                if self.verbose:
                    print(
                        f"Using input at indices: {s*self.input_time_dim} : {(s+1)*self.input_time_dim}, to produce output at indices: {(s+1)*self.input_time_dim} : {(s+2)*self.input_time_dim}"
                    )
            else:
                s = local_step + self.presteps
                s_ = step + self.presteps
                input_tensor = self._reshape(
                    torch.cat(
                        [
                            outputs[-1],
                            inputs[
                                :,
                                s * self.input_time_dim : (s + 1) * self.input_time_dim,
                                self.output_channels :,
                            ],
                        ],
                        dim=2,
                    )
                )
                if self.verbose:
                    # When you refer to indices in output, they will always be shift by 1 as they never output indices from 0:n. Then, additional offset by self.presteps.
                    print(
                        f"Using output at indices {(len(outputs)+self.presteps)*self.input_time_dim}:{(len(outputs)+self.presteps+1)*self.input_time_dim} to produce output at indices: {(len(outputs)+self.presteps+1)*self.input_time_dim}:{(len(outputs)+self.presteps+2)*self.input_time_dim}"
                    )
                    print(
                        f"Also, Concatenating extra input in range of local indices {s*self.input_time_dim}:{(s+1)*self.input_time_dim} and global indices {s_*self.input_time_dim}:{(s_+1)*self.input_time_dim}"
                    )

            if self.verbose:
                print("Passing through UNET")
            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)
            # Absolute prediction
            # reshaped = self._reshape_outputs(decodings)
            # Residual prediction
            reshaped = self._reshape_output(
                input_tensor[:, : self.output_channels * self.input_time_dim]
                + decodings
            )
            outputs.append(reshaped)

        if output_only_last:
            return outputs[-1]

        if inference:
            return outputs

        return torch.cat(outputs, dim=self.time_dim)

    def reset(self):
        if self.verbose:
            print("Resetting hiddens")
        self.encoder.reset()
        self.decoder.reset()


def train_one_epoch(
    model,
    epoch,
    train_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    wandb_flag,
    gscaler,
):

    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    iters = len(train_loader)

    for data_iter_step, data in enumerate(
        metric_logger.log_every(train_loader, 1, header)
    ):

        # optimizer.zero_grad()
        model.zero_grad(set_to_none=True)
        with amp.autocast(enabled=gscaler is not None, dtype=torch.float16):
            outs = model(data[0].to(device=device))
            loss = loss_fn(data[1].to(device=device), outs)

        # loss.backward()
        gscaler.scale(loss).backward()
        gscaler.unscale_(optimizer)
        curr_lr = (
            optimizer.param_groups[-1]["lr"]
            if scheduler is None
            else scheduler.get_last_lr()[0]
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), curr_lr)

        gscaler.step(optimizer)
        gscaler.update()

        loss_value = loss.item()

        # optimizer.step()
        if scheduler is not None:
            # scheduler.step()
            scheduler.step(epoch + data_iter_step / iters)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value)

        lr = curr_lr
        metric_logger.update(lr=lr)

        loss_value_reduce = all_reduce_mean(loss_value)

        if wandb_flag:
            wandb.log({"train_loss_per_batch": loss_value_reduce, "lr_per_batch": lr})

    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, test_loader, device, wandb_flag, gscaler):
    model.eval()
    mse = nn.MSELoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    for data, label in test_loader:
        with torch.no_grad():
            with amp.autocast(enabled=gscaler is not None, dtype=torch.float16):
                outs = model(data.to(device=device))
                loss = mse(outs, label.to(device=device))
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            loss_value_reduce = all_reduce_mean(loss_value)
            if wandb_flag:
                wandb.log({"eval_loss_per_batch": loss_value_reduce})

    metric_logger.synchronize_between_processes()
    print("Averaged eval stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# if __name__ == "__main__":
#     from torchsummary import summary
#     ####
#     # model = RecUNet(2, 8)
#     # # model(torch.zeros([1, 10, 9, 128, 192]))
#     # summary(model.cuda(), (10, 9, 128, 192))

#     ####
#     # model = NoRecUNet(1,8)
#     # res = model(torch.zeros([5, 8, 9, 128, 192]))
#     # print(res.shape)
#     # # summary(model.cuda(), (8, 9, 128, 192))

#     ####
#     model = NoRecUNetSimple(1,8)
#     res = model(torch.zeros([5, 9, 128, 192]))
#     print(res.shape)
#     summary(model.cuda(), (9, 128, 192))

#     model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     weights = [
#                 1.0
#             ] * 8
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#                 optimizer, 10
#             )
#     epochs = 10
#     steps = 8
#     loss_fn = torch.nn.MSELoss()

#     data = torch.rand([16, 1, 9, 128, 192])
#     data2 = torch.rand([16, 1, 9, 128, 192])

#     for data in [data, data2]:
#         optimizer.zero_grad()
#         outs = []
#         out = model(data[0].cuda())
#         outs.append(out)
#         for step in range(1, steps):
#             print("Step: ",step)
#             step_in = torch.concat(
#                 (outs[-1], data[int(step * 2)][:, 3:].cuda()), 1
#             )

#             out = model(step_in)
#             outs.append(out)

#     loss = loss_fn(data[1][:,:3].cuda(), outs[0]) * weights[0]
#     for step in range(1, steps):
#         loss += (
#             loss_fn(data[int(step * 2 + 1)][:,:3].cuda(), outs[step])
#             * weights[step]
#         )
#     loss.backward()

#     loss_value = loss.item()
