# TODO: Need to return step-wise losses for logging

import logging

import torch

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        wet,
        hist,
        last_kernel_size,
        pad,
        static_data,
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wet = wet.bool()
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.hist = hist
        self.static_data = static_data

    def forward(self, fts):
        raise NotImplementedError()
