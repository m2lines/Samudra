from typing import Literal

import torch.nn as nn

from ocean_emulators.models.modules.activations import CappedGELU, ReLU
from ocean_emulators.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    ConvBlock,
    ConvNeXtBlock,
    MaxPool,
    TransposedConvUpsample,
)
from ocean_emulators.models.modules.dropout import EarlyDropPath

BLOCK_REGISTRY = {
    "conv_block": ConvBlock,
    "conv_next_block": ConvNeXtBlock,
}

DOWNSAMPLE_REGISTRY = {
    "avg_pool": AvgPool,
    "max_pool": MaxPool,
}

UPSAMPLE_REGISTRY = {
    "bilinear_upsample": BilinearUpsample,
    "transposed_conv": TransposedConvUpsample,
}

ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "capped_gelu": CappedGELU,
}


def create_block(
    block_type: str,
    drop_path_rate: float = 0.0,
    early_dropout_epochs: int = 0,
    dropout_schedule: Literal["early_only", "late_only", "constant"] = "early_only",
    linear_decay: bool = True,
    **kwargs,
) -> nn.Module:
    """A UNet block factory that supports stochastic depth dropout."""
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")

    block = BLOCK_REGISTRY[block_type](**kwargs)

    # Add early dropout if specified and block supports it
    if (
        drop_path_rate > 0.0
        and early_dropout_epochs > 0
        and isinstance(block, (ConvNeXtBlock, ConvBlock))
    ):
        block.drop_path = EarlyDropPath(
            drop_prob=drop_path_rate,
            early_epochs=early_dropout_epochs,
            schedule=dropout_schedule,
            linear_decay=linear_decay,
        )

    return block


def create_downsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in DOWNSAMPLE_REGISTRY:
        raise ValueError(f"Unknown downsample type: {block_type}")
    return DOWNSAMPLE_REGISTRY[block_type](**kwargs)


def create_upsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in UPSAMPLE_REGISTRY:
        raise ValueError(f"Unknown upsample type: {block_type}")
    return UPSAMPLE_REGISTRY[block_type](**kwargs)


def get_activation_cl(activation_type: str) -> type[nn.Module]:
    if activation_type not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation type: {activation_type}")
    return ACTIVATION_REGISTRY[activation_type]
