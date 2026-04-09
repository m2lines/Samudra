"""Validation sub-aggregator that captures and logs attention maps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from ocean_emulators.aggregator.plotting import (
    plot_attention_map,
    plot_attention_receptive_field,
    plot_full_attention_receptive_field,
)
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.models.modules.blocks import (
    AvgPool,
    AxialAttention,
    AxialAttentionBlock,
    BilinearUpsample,
    FullAttention,
    FullAttentionBlock,
    MaxPool,
    TransposedConvUpsample,
    ZonallyPeriodicBilinearUpsample,
)
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.wandb import Metrics, MetricsDict


@dataclass(frozen=True)
class _AttentionBlockInfo:
    name: str
    block: AxialAttentionBlock | FullAttentionBlock


@dataclass(frozen=True)
class _AxialAttentionCapture:
    height_weights: torch.Tensor
    width_weights: torch.Tensor


@dataclass(frozen=True)
class _FullAttentionCapture:
    weights: torch.Tensor
    spatial_shape: tuple[int, int]


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _get_unet_backbone(model: nn.Module) -> UNetBackbone | None:
    unwrapped = _unwrap_model(model)
    for attr in ("unet", "processor"):
        backbone = getattr(unwrapped, attr, None)
        if isinstance(backbone, UNetBackbone):
            return backbone
    for module in unwrapped.modules():
        if isinstance(module, UNetBackbone):
            return module
    return None


def _is_attention_block(layer: nn.Module) -> bool:
    return isinstance(layer, AxialAttentionBlock | FullAttentionBlock)


def _is_upsampling_layer(layer: nn.Module) -> bool:
    return isinstance(
        layer,
        BilinearUpsample | TransposedConvUpsample | ZonallyPeriodicBilinearUpsample,
    )


def _collect_attention_blocks(model: nn.Module) -> list[_AttentionBlockInfo]:
    """Collect attention blocks and assign stable U-Net stage names.

    Stage indices follow the U-Net scale they are attached to, so names like
    ``encoder_0`` and ``decoder_1`` align with config slots rather than with the
    count of attention blocks that happen to be enabled.
    """
    backbone = _get_unet_backbone(model)
    if backbone is None:
        return []

    first_upsample_index = next(
        (i for i, layer in enumerate(backbone.layers) if _is_upsampling_layer(layer)),
        None,
    )
    if first_upsample_index is None:
        return []

    blocks: list[_AttentionBlockInfo] = []
    encoder_stage = 0
    decoder_stage = 0
    in_decoder = False

    for index, layer in enumerate(backbone.layers):
        if index == first_upsample_index:
            in_decoder = True

        if _is_attention_block(layer):
            if index == first_upsample_index - 1:
                name = "bottleneck"
            elif in_decoder:
                name = f"decoder_{decoder_stage}"
            else:
                name = f"encoder_{encoder_stage}"
            blocks.append(_AttentionBlockInfo(name=name, block=layer))

        if not in_decoder and isinstance(layer, AvgPool | MaxPool):
            encoder_stage += 1
        elif (
            in_decoder
            and index != first_upsample_index
            and _is_upsampling_layer(layer)
        ):
            decoder_stage += 1

    return blocks


def has_attention_blocks(model: nn.Module) -> bool:
    return bool(_collect_attention_blocks(model))


@contextmanager
def capture_attention(model: nn.Module):
    """Context manager that enables attention-weight capture on all
    supported attention submodules, then disables it on exit.

    Usage::

        with capture_attention(model):
            output = model(input)
        # Now read the captured weights from the attention submodules.
    """
    attn_modules: list[AxialAttention | FullAttention] = [
        m for m in model.modules() if isinstance(m, AxialAttention | FullAttention)
    ]
    for m in attn_modules:
        m.capture_weights = True
    try:
        yield attn_modules
    finally:
        for m in attn_modules:
            m.capture_weights = False
            # Don't clear last_attn_weights here — the aggregator reads
            # them after the context exits.  They'll be overwritten on the
            # next capture pass or garbage-collected with the module.


class AttentionAggregator(ValidateSubAggregator):
    """Captures attention maps from the last validation batch and logs them.

    Axial blocks produce height/width matrices and a combined receptive field.
    Full-attention blocks produce a matrix overview and a receptive field on the
    2D stage grid.
    """

    def __init__(
        self,
        model: nn.Module,
        query_lat: int | None = None,
        query_lon: int | None = None,
    ):
        """
        Args:
            model: The model (or DDP-wrapped model) to inspect for attention
                blocks inside its U-Net backbone.
            query_lat: Latitude index (in stage spatial dims) for receptive-field
                plots. Defaults to the stage center.
            query_lon: Longitude index (in stage spatial dims) for receptive-field
                plots. Defaults to the stage center.
        """
        self._blocks = _collect_attention_blocks(model)
        self._query_lat = query_lat
        self._query_lon = query_lon
        self._captures: dict[
            str, _AxialAttentionCapture | _FullAttentionCapture
        ] = {}

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss: torch.Tensor,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ):
        """Grab the latest attention weights stored by the capture pass.

        This should be called *after* the model forward pass that ran with
        :func:`capture_attention`.
        """
        for block_info in self._blocks:
            block = block_info.block
            if isinstance(block, AxialAttentionBlock):
                if (
                    block.attn_h.last_attn_weights is not None
                    and block.attn_w.last_attn_weights is not None
                ):
                    self._captures[block_info.name] = _AxialAttentionCapture(
                        height_weights=block.attn_h.last_attn_weights,
                        width_weights=block.attn_w.last_attn_weights,
                    )
            elif isinstance(block, FullAttentionBlock):
                if (
                    block.attn.last_attn_weights is not None
                    and block.attn.last_spatial_shape is not None
                ):
                    self._captures[block_info.name] = _FullAttentionCapture(
                        weights=block.attn.last_attn_weights,
                        spatial_shape=block.attn.last_spatial_shape,
                    )

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        """Render captured attention summaries keyed by stage name."""
        logs: MetricsDict = {}
        if not self._captures:
            return logs

        for name, capture in self._captures.items():
            if isinstance(capture, _AxialAttentionCapture):
                h_np = capture.height_weights.float().numpy()
                w_np = capture.width_weights.float().numpy()
                query_lat = (
                    self._query_lat
                    if self._query_lat is not None
                    else h_np.shape[0] // 2
                )
                query_lon = (
                    self._query_lon
                    if self._query_lon is not None
                    else w_np.shape[0] // 2
                )
                logs[f"{label}/{name}/height"] = plot_attention_map(
                    h_np,
                    axis="height",
                    caption="Height-axis attention (avg over heads, batch, width)",
                )
                logs[f"{label}/{name}/width"] = plot_attention_map(
                    w_np,
                    axis="width",
                    caption="Width-axis attention (avg over heads, batch, height)",
                )
                logs[f"{label}/{name}/receptive_field"] = (
                    plot_attention_receptive_field(
                        h_np,
                        w_np,
                        query_lat=query_lat,
                        query_lon=query_lon,
                        caption=(
                            f"Combined receptive field at {name} "
                            f"({query_lat}, {query_lon})"
                        ),
                    )
                )
            else:
                attn_np = capture.weights.float().numpy()
                height, width = capture.spatial_shape
                query_lat = (
                    self._query_lat if self._query_lat is not None else height // 2
                )
                query_lon = (
                    self._query_lon if self._query_lon is not None else width // 2
                )
                logs[f"{label}/{name}/matrix"] = plot_attention_map(
                    attn_np,
                    axis="full",
                    caption="Full attention (avg over heads and batch)",
                )
                logs[f"{label}/{name}/receptive_field"] = (
                    plot_full_attention_receptive_field(
                        attn_np,
                        grid_shape=capture.spatial_shape,
                        query_lat=query_lat,
                        query_lon=query_lon,
                        caption=(
                            f"Full-attention receptive field at {name} "
                            f"({query_lat}, {query_lon})"
                        ),
                    )
                )

        return logs
