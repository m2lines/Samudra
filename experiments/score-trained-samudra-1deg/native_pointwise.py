# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Torch RTX mitigation from the PR #915 training diagnostics.

Keep Conv2d parameters/checkpoints unchanged, using native PyTorch only for
1x1 convolutions. This is an empirical mitigation, not a library root-cause fix.
Call before constructing a model in each training/evaluation process.
"""

import torch

_original_forward = torch.nn.Conv2d.forward


def _native_pointwise_forward(self, x):
    if self.kernel_size == (1, 1):
        with torch.backends.cudnn.flags(enabled=False):
            return _original_forward(self, x)
    return _original_forward(self, x)


def configure_native_pointwise(enabled: bool):
    torch.nn.Conv2d.forward = (  # type: ignore[method-assign]
        _native_pointwise_forward if enabled else _original_forward
    )
