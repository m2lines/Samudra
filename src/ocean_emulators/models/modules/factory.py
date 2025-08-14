import torch.nn as nn

from ocean_emulators.models.modules.activations import CappedGELU, ReLU
from ocean_emulators.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    ConvBlock,
    ConvNeXtBlock,
    DiscoBlock,
    MaxPool,
    TransposedConvUpsample,
)

BLOCK_REGISTRY = {
    "conv_block": ConvBlock,
    "conv_next_block": ConvNeXtBlock,
    "disco_block": DiscoBlock,
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


def create_block(block_type: str, **kwargs) -> nn.Module:
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")

    # Only pass grid_shape to blocks that need it (currently only DiscoBlock)
    if block_type != "disco_block" and "grid_shape" in kwargs:
        kwargs = kwargs.copy()
        del kwargs["grid_shape"]

    return BLOCK_REGISTRY[block_type](**kwargs)


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
