import torch


class BaseModel(torch.nn.Module):
    def __init__(self, ch_width, n_out, wet, pred_residuals, last_kernel_size, pad):
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.output_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

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
        self, inputs, num_steps=None, output_only_last=False, device="cuda"
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = inputs[0][0].unsqueeze(0).to(device=device)
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                    [
                        inputs_0.unsqueeze(0),
                        inputs[step][0][self.output_channels :]
                        .unsqueeze(0)
                        .to(device=device),
                    ],
                    dim=1,
                )

            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[0, : self.output_channels].to(
                    device=device
                ) + decodings.squeeze(
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
