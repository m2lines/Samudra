import torch

# from huggingface_hub import PyTorchModelHubMixin


# class BaseModel(torch.nn.Module, PyTorchModelHubMixin):
class BaseModel(torch.nn.Module):
    def __init__(self, ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad):
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
        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = inputs[0]
            elif step <= self.hist:
                inputs_0 = inputs[2 * step][:, :self.output_channels * (self.hist-step+1)]
                inputs_1 = torch.cat([outputs[i] for i in range(step)], dim=1)
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs_1,
                        inputs[2 * step][:, self.output_channels * (self.hist+1) :],
                    ],
                    dim=1,
                )   
            else:
                inputs_0 = torch.cat([outputs[i] for i in range(-self.hist-1, 0)], dim=1)
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[2 * step][:, self.output_channels * (self.hist+1) :],
                    ],
                    dim=1,
                )
            
            assert input_tensor.shape[1] == self.input_channels, f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[:, self.output_channels * self.hist : self.output_channels * (self.hist+1)] + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

            if loss_fn is not None:
                assert reshaped.shape == inputs[2 * step + 1].shape, f"Output shape is {reshaped.shape} but should be {inputs[2 * step + 1].shape}"
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

    def inference(
        self, inputs, num_steps=None, output_only_last=False, device="cuda"
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = inputs[0][0].to(device=device) # inputs[0][0] is the input at step 0 
            elif step <= self.hist:
                inputs_0 = inputs[step][0][0, :self.output_channels * (self.hist-step+1)].unsqueeze(0).to(device=device) # If we are within the range of using states in inputs[0][0], we use them until we have produced enough outputs
                inputs_1 = torch.cat([outputs[i].unsqueeze(0) for i in range(step)], dim=1) # Outputs we currently have
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs_1,
                        inputs[step][0][0, self.output_channels * (self.hist+1) :].unsqueeze(0) # concatenate the boundary conditions
                        .to(device=device),
                    ],
                    dim=1,
                )
            else:
                inputs_0 = torch.cat([outputs[i].unsqueeze(0) for i in range(-self.hist-1, 0)], dim=1) # Last self.hist+1 outputs
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[step][0][0, self.output_channels*(self.hist+1) :].unsqueeze(0) # concatenate the boundary conditions
                        .to(device=device),
                    ],
                    dim=1,
                )

            assert input_tensor.shape[1] == self.input_channels, f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[0, self.output_channels * self.hist : self.output_channels * (self.hist+1)].to( # Residuals on last state in input
                    device=device
                ) + decodings.squeeze(
                    0
                )
            else:
                reshaped = decodings.squeeze(0)
            
            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res
