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


def create_block(
    block_type: str,
    dropout_manager=None,  # StochasticDepthManager | None
    layer_index: int = 0,
    **kwargs,
) -> nn.Module:
    """A UNet block factory that supports stochastic depth dropout."""
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")

    block = BLOCK_REGISTRY[block_type](**kwargs)

    # Apply dropout if supported (2303.01500)
    if dropout_manager is not None and isinstance(block, (ConvNeXtBlock, ConvBlock)):
        drop_path = dropout_manager.create_drop_path(layer_index)
        if drop_path is not None:
            block.drop_path = drop_path

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
