# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Direct next-latent prediction with explicit, strictly loaded checkpoint mode."""

import torch
from torch import nn

from samudra.experiments.diffusion_latent import PersistentLatentProcessor

_MARKER = "processor.direct_prediction_version"


class DirectLatentProcessor(PersistentLatentProcessor):
    """Reuse the processor backbone, predicting the next state without adding it.

    Inner residual blocks remain unchanged. Only the persistent-state addition is
    removed. Conversion keeps the existing parameter objects, including adapters,
    and does not consume random numbers or initialize an implicit identity path.
    """

    direct_prediction_version: torch.Tensor

    def __init__(self, source: PersistentLatentProcessor):
        nn.Module.__init__(self)
        self.width = source.width
        self.latent_shape = source.latent_shape
        self.conditioning_shape = source.conditioning_shape
        self.stem = source.stem
        self.blocks = source.blocks
        self.head = source.head
        if hasattr(source, "input_adapters"):
            self.input_adapters = source.input_adapters
        self.register_buffer(
            "direct_prediction_version",
            torch.tensor(1, dtype=torch.int64, device=next(source.parameters()).device),
        )
        self.training = source.training

    def _next_latent(self, latent, prediction):
        return prediction


def install_direct_processor(model):
    """Convert an already loaded residual model to direct prediction in place."""
    if type(model.processor) is DirectLatentProcessor:
        return model
    if type(model.processor) is not PersistentLatentProcessor:
        raise TypeError("Direct conversion requires the standard latent processor")
    model.processor = DirectLatentProcessor(model.processor)
    return model


def checkpoint_processor_prediction(saved):
    """Validate metadata against the architecture marker; legacy means residual."""
    state = saved["model"]
    mode = saved.get("processor_prediction", "residual")
    if mode not in ("residual", "direct"):
        raise ValueError(f"Unknown processor prediction mode: {mode!r}")
    has_marker = _MARKER in state
    if has_marker != (mode == "direct"):
        raise ValueError("Processor prediction metadata and state marker disagree")
    if has_marker:
        marker = state[_MARKER]
        if (
            not isinstance(marker, torch.Tensor)
            or marker.shape != torch.Size([])
            or marker.dtype != torch.int64
            or marker.item() != 1
        ):
            raise ValueError("Unsupported direct prediction checkpoint version")
    return mode


def load_latent_checkpoint(model, saved):
    """Configure a fresh model from validated metadata, then load strictly.

    Unmodified historical checkpoints lack both metadata and marker and retain
    the residual architecture. Direct checkpoints require both; ignoring their
    extra state with a non-strict load is never allowed here.
    """
    mode = checkpoint_processor_prediction(saved)
    if mode == "direct":
        install_direct_processor(model)
    elif type(model.processor) is not PersistentLatentProcessor:
        raise ValueError("Residual checkpoint requires a fresh residual model")
    model.load_state_dict(saved["model"], strict=True)
    return model
