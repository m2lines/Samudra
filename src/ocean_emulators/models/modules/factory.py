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


def create_downsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in DOWNSAMPLE_REGISTRY:
        raise ValueError(f"Unknown downsample type: {block_type}")
    return DOWNSAMPLE_REGISTRY[block_type](**kwargs)


def create_upsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in UPSAMPLE_REGISTRY:
        raise ValueError(f"Unknown upsample type: {block_type}")
    return UPSAMPLE_REGISTRY[block_type](**kwargs)
