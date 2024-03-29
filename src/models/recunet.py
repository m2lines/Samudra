import torch
import torch.nn as nn
import torch.nn.functional as F

# from typing import Any, Dict, Optional, Sequence, Union
from typing import Sequence
from utils.train_utils import SmoothedValue, MetricLogger


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

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)  # + self.presteps

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * self.input_channels

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape(self, input) -> torch.Tensor:
        B, T, C, H, W = input.shape
        return input.reshape(B, T * C, H, W)

    def _reshape_output(self, output) -> torch.Tensor:
        B, _, H, W = output.shape
        return output.reshape(B, -1, self.output_channels, H, W)

    def _initialize_hidden(
        self, inputs: Sequence, outputs: Sequence, step: int
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
                s = step - self.presteps + prestep
                input_tensor = self._reshape(
                    torch.cat(
                        [
                            outputs[s - 1],
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
                        f"Initialize Hidden: Using output indices: {(s+self.presteps)*self.input_time_dim}:{(s+1+self.presteps)*self.input_time_dim} to produce (not saved) output at indices: {(s+1+self.presteps)*self.input_time_dim}:{(s+2+self.presteps)*self.input_time_dim}"
                    )
                    print(
                        f"Also, Concatenating extra input in range of indices {(s+1)*self.input_time_dim}:{(s+2)*self.input_time_dim}"
                    )

            # Forward the data through the model to initialize hidden states
            self.decoder(self.encoder(input_tensor))

    # [B, T, C, H, W]
    def forward(self, inputs: Sequence, output_only_last=False) -> torch.Tensor:
        self.reset()
        outputs = []
        for step in range(self.integration_steps):
            if self.verbose:
                print(f"Step: {step}")
            # (Re-)initialize recurrent hidden states
            if step % self.reset_cycle == 0:
                if self.verbose:
                    print(f"Reinitializing at Step: {step}")
                self._initialize_hidden(inputs=inputs, outputs=outputs, step=step)

            # Construct input: [prognostics|TISR|constants]
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
                s = step + self.presteps
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
                        f"Also, Concatenating extra input in range of indices {s*self.input_time_dim}:{(s+1)*self.input_time_dim}"
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

        return torch.cat(outputs, dim=self.time_dim)

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()

    def train_one_epoch(
        self,
        epoch,
        train_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        wandb_flag,
    ):
        model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch)
        iters = len(train_loader)

        for data_iter_step, data in enumerate(
            metric_logger.log_every(train_loader, 1, header)
        ):

            optimizer.zero_grad()
            outs = model(data[0].to(device=device))
            loss = loss_fn(data[1].to(device=device), outs)
            loss.backward()

            loss_value = loss.item()

            optimizer.step()
            if scheduler is not None:
                # scheduler.step()
                scheduler.step(epoch + data_iter_step / iters)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = all_reduce_mean(loss_value)

            if wandb_flag:
                wandb.log(
                    {"train_loss_per_batch": loss_value_reduce, "lr_per_batch": lr}
                )

        metric_logger.synchronize_between_processes()
        print("Averaged train stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validate(self, test_loader, device, wandb_flag):
        model.eval()
        mse = nn.MSELoss()

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        for data, label in test_loader:
            with torch.no_grad():
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
