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
            # reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            reshaped = (
                input_tensor[:, : self.output_channels] + decodings
            )  # Residual prediction
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
            # reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            reshaped = input_tensor[0, : self.output_channels] + decodings.squeeze(
                0
            )  # Residual prediction

            reshaped = reshaped[:, : self.input_size[0], : self.input_size[1]]
            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res


# Need to update forward function and add inference
class RecUNet(BaseUNet):
    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
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
        super().__init__(
            encoder,
            decoder,
            input_time_dim,
            output_time_dim,
            input_channels,
            output_channels,
            presteps,
        )
        self.time_dim = 1
        self.reset_cycle = reset_cycle
        self.verbose = verbose

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
