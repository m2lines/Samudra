from __future__ import annotations

from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn

from ocean_emulators.aggregator.plotting import (
    plot_attention_map,
    plot_attention_receptive_field,
    plot_full_attention_receptive_field,
)
from ocean_emulators.aggregator.validate.sub_aggregator import ValidateSubAggregator
from ocean_emulators.models.modules.blocks import (
    AxialAttention,
    AxialAttentionBlock,
    FullAttention,
    FullAttentionBlock,
)
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.wandb import Metrics, MetricsDict


def _get_unet_backbone(model: nn.Module) -> UNetBackbone | None:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    for attr in ("unet", "processor"):
        backbone = getattr(model, attr, None)
        if isinstance(backbone, UNetBackbone):
            return backbone

    for module in model.modules():
        if isinstance(module, UNetBackbone):
            return module

    return None


def _collect_attention_blocks(
    model: nn.Module,
) -> list[tuple[str, AxialAttentionBlock | FullAttentionBlock]]:
    backbone = _get_unet_backbone(model)
    if backbone is None:
        return []

    blocks: list[tuple[str, AxialAttentionBlock | FullAttentionBlock]] = []
    for layer_name, layer in zip(backbone.layer_names, backbone.layers, strict=True):
        if layer_name == "bottleneck_attention":
            blocks.append(
                ("bottleneck", cast(AxialAttentionBlock | FullAttentionBlock, layer))
            )
        elif layer_name.startswith("encoder_skip_attention_"):
            stage = layer_name.removeprefix("encoder_skip_attention_")
            blocks.append(
                (
                    f"encoder_{stage}",
                    cast(AxialAttentionBlock | FullAttentionBlock, layer),
                )
            )
        elif layer_name.startswith("decoder_attention_"):
            stage = layer_name.removeprefix("decoder_attention_")
            blocks.append(
                (
                    f"decoder_{stage}",
                    cast(AxialAttentionBlock | FullAttentionBlock, layer),
                )
            )

    return blocks


@contextmanager
def capture_attention(model: nn.Module):
    attn_modules: list[AxialAttention | FullAttention] = []
    for _, block in _collect_attention_blocks(model):
        if isinstance(block, AxialAttentionBlock):
            axial_block = cast(AxialAttentionBlock, block)
            attn_modules.extend([axial_block.attn_h, axial_block.attn_w])
        else:
            full_block = cast(FullAttentionBlock, block)
            attn_modules.append(full_block.attn)

    for module in attn_modules:
        module.capture_weights = True
    try:
        yield attn_modules
    finally:
        for module in attn_modules:
            module.capture_weights = False


class AttentionAggregator(ValidateSubAggregator):
    """Log attention summaries from the last validation batch."""

    def __init__(
        self,
        model: nn.Module,
        query_lat: int | None = None,
        query_lon: int | None = None,
    ):
        self._blocks = _collect_attention_blocks(model)
        self._query_lat = query_lat
        self._query_lon = query_lon
        self._axial_captures: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._full_captures: dict[str, tuple[torch.Tensor, tuple[int, int]]] = {}

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
        for name, block in self._blocks:
            if isinstance(block, AxialAttentionBlock):
                axial_block = cast(AxialAttentionBlock, block)
                if (
                    axial_block.attn_h.last_attn_weights is not None
                    and axial_block.attn_w.last_attn_weights is not None
                ):
                    self._axial_captures[name] = (
                        axial_block.attn_h.last_attn_weights,
                        axial_block.attn_w.last_attn_weights,
                    )
            else:
                full_block = cast(FullAttentionBlock, block)
                if (
                    full_block.attn.last_attn_weights is None
                    or full_block.attn.last_spatial_shape is None
                ):
                    continue
                self._full_captures[name] = (
                    full_block.attn.last_attn_weights,
                    full_block.attn.last_spatial_shape,
                )

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        logs: MetricsDict = {}

        for name, (height_weights, width_weights) in self._axial_captures.items():
            height_np = height_weights.float().numpy()
            width_np = width_weights.float().numpy()
            query_lat = (
                self._query_lat
                if self._query_lat is not None
                else height_np.shape[0] // 2
            )
            query_lon = (
                self._query_lon
                if self._query_lon is not None
                else width_np.shape[0] // 2
            )

            logs[f"{label}/{name}/height"] = plot_attention_map(
                height_np,
                axis="height",
                caption="Height-axis attention (avg over heads, batch, width)",
            )
            logs[f"{label}/{name}/width"] = plot_attention_map(
                width_np,
                axis="width",
                caption="Width-axis attention (avg over heads, batch, height)",
            )
            logs[f"{label}/{name}/receptive_field"] = plot_attention_receptive_field(
                height_np,
                width_np,
                query_lat=query_lat,
                query_lon=query_lon,
                caption=(
                    f"Combined receptive field at {name} ({query_lat}, {query_lon})"
                ),
            )

        for name, (weights, spatial_shape) in self._full_captures.items():
            weights_np = weights.float().numpy()
            height, width = spatial_shape
            query_lat = self._query_lat if self._query_lat is not None else height // 2
            query_lon = self._query_lon if self._query_lon is not None else width // 2

            logs[f"{label}/{name}/matrix"] = plot_attention_map(
                weights_np,
                axis="full",
                caption="Full attention (avg over heads and batch)",
            )
            logs[f"{label}/{name}/receptive_field"] = (
                plot_full_attention_receptive_field(
                    weights_np,
                    grid_shape=spatial_shape,
                    query_lat=query_lat,
                    query_lon=query_lon,
                    caption=(
                        f"Full-attention receptive field at {name} "
                        f"({query_lat}, {query_lon})"
                    ),
                )
            )

        return logs
