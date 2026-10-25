"""Experiment-only native 1x1 convolution mitigation, shared by every arm."""

import torch

_original_conv_forward = torch.nn.Conv2d.forward


def _forward(self, input):
    if self.kernel_size == (1, 1) and input.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return _original_conv_forward(self, input)
    return _original_conv_forward(self, input)


def install():
    torch.nn.Conv2d.forward = _forward
