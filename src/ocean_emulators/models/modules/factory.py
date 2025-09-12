from typing import Protocol

from ocean_emulators.models.modules.activations import CappedGELU, ReLU
from ocean_emulators.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    ConvBlock,
    ConvNeXtBlock,
    CoreBlock,
    MaxPool,
    TransposedConvUpsample,
)


class CoreBlockBuilder(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        n_layers: int,
        pad: str,
        checkpoint_simple: bool,
    ) -> CoreBlock: ...


class UpsamplingBlockBuilder(Protocol):
    def __call__(
        self, in_channels: int, out_channels: int
    ) -> BilinearUpsample | TransposedConvUpsample: ...


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
