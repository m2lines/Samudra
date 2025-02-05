# TODO: Need to return step-wise losses for logging

import torch

from utils.device import get_device


class BaseModel(torch.nn.Module):
    def __init__(
        self, ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
    ):
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.input_channels = ch_width[0]
        self.output_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        inputs,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                """
                For HIST=1, [0->[0in, 1in], 1->[2out, 3out],
                            2->[2in, 3in], 3->[4out, 5out]]
                """
                input_tensor = inputs[0]
            else:
                # Last output corresponds to input at current time step
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[2 * step][
                            :, self.output_channels :
                        ],  # boundary conditions
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but should\
                be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[
                        :,
                        : self.output_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

            if loss_fn is not None:
                assert (
                    reshaped.shape == inputs[2 * step + 1].shape
                ), f"Output shape is {reshaped.shape} but should\
                        be {inputs[2 * step + 1].shape}"
                if loss is None:
                    loss = loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
                    )
                else:
                    loss += loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
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

    # TODO: Remove this function once we fix standalone inference
    def inference(
        self,
        inputs,
        initial_input=None,
        num_steps=None,
        output_only_last=False,
        aggregator=None,
    ):
        outputs: list[torch.Tensor] = []
        for step in range(num_steps):
            if step == 0:
                """
                inputs[0][0] is the input at step 0.
                For HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]];
                            2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
                """
                input_tensor = inputs[0][0].to(device=get_device())

                if initial_input is not None:
                    input_tensor[:, : self.output_channels] = initial_input
            else:
                inputs_0 = outputs[-1].unsqueeze(
                    0
                )  # Last output corresponds to input at current time step
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[step][0][
                            :, self.output_channels :
                        ].to(  # boundary conditions
                            device=get_device()
                        ),
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but \
                should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[
                    0,
                    : self.output_channels,
                ].to(  # Residuals on last state in input
                    device=get_device()
                ) + decodings.squeeze(0)
            else:
                reshaped = decodings.squeeze(0)

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res
