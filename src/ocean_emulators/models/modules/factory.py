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
