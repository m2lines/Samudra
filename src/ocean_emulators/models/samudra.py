from typing import assert_never

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ocean_emulators.config import SamudraConfig
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.corrector import Correctors
from ocean_emulators.models.modules.blocks import (
    BilinearUpsample,
    CoreBlock,
    TransposedConvUpsample,
)
from ocean_emulators.models.modules.factory import (
    create_block,
    create_downsample,
    create_upsample,
    get_activation_cl,
)
from ocean_emulators.utils.train import pairwise


class Samudra(BaseModel):
    def __init__(self, config: SamudraConfig, hist, wet, area_weights, static_data):
        super().__init__(
            ch_width=config.ch_width,
            n_out=config.n_out,
            wet=wet,
            hist=hist,
            pred_residuals=config.pred_residuals,
            last_kernel_size=config.last_kernel_size,
            pad=config.pad,
            static_data=static_data,
        )

        # Get activation class
        activation = get_activation_cl(config.core_block.activation)

        # Create local copies of config lists that will be reversed
        ch_width = config.ch_width.copy()
        dilation = config.dilation.copy()
        n_layers = config.n_layers.copy()

        match config.checkpointing:
            case "all":
                self.checkpoint_all = True
                checkpoint_simple = False
            case "simple":
                self.checkpoint_all = False
                checkpoint_simple = True
            case None:
                self.checkpoint_all = False
                checkpoint_simple = False
            case _:
                assert_never(config.checkpointing)

        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Calculate drop path rate for this layer (2303.01500, 2201.03545)
            # Per-stage multipliers allow different dropout rates at different U-Net depths
            stage_multiplier = 1.0
            if config.stochastic_depth.per_stage_multipliers:
                stage_multiplier = config.stochastic_depth.per_stage_multipliers[i]

            drop_rate = config.stochastic_depth.drop_path_rate * stage_multiplier

            # Core block
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                    checkpoint_simple=checkpoint_simple,
                    # Early dropout parameters
                    drop_path_rate=drop_rate,
                    early_dropout_epochs=config.stochastic_depth.early_dropout_epochs,
                    dropout_schedule=config.stochastic_depth.dropout_schedule,
                    linear_decay=config.stochastic_depth.linear_decay_to_zero,
                )
            )
            # Down sampling block
            layers.append(create_downsample(config.down_sampling_block))

        # Middle block - apply same dropout settings
        middle_stage_multiplier = 1.0
        if config.stochastic_depth.per_stage_multipliers:
            # Use the last multiplier for middle block
            middle_stage_multiplier = config.stochastic_depth.per_stage_multipliers[-1]

        middle_drop_rate = (
            config.stochastic_depth.drop_path_rate * middle_stage_multiplier
        )

        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
                checkpoint_simple=checkpoint_simple,
                # Early dropout parameters for middle block
                drop_path_rate=middle_drop_rate,
                early_dropout_epochs=config.stochastic_depth.early_dropout_epochs,
                dropout_schedule=config.stochastic_depth.dropout_schedule,
                linear_decay=config.stochastic_depth.linear_decay_to_zero,
            )
        )

        # First upsampling
        layers.append(
            create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
        )

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()

        # going up
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            # For decoder, we can optionally disable dropout or use reduced rates
            # For now, apply same settings as encoder
            decoder_stage_idx = len(ch_width) - 2 - i  # Reverse index for decoder
            decoder_stage_multiplier = 1.0
            if config.stochastic_depth.per_stage_multipliers:
                # Use corresponding encoder multiplier (reversed)
                if decoder_stage_idx < len(
                    config.stochastic_depth.per_stage_multipliers
                ):
                    decoder_stage_multiplier = (
                        config.stochastic_depth.per_stage_multipliers[decoder_stage_idx]
                    )

            decoder_drop_rate = (
                config.stochastic_depth.drop_path_rate * decoder_stage_multiplier
            )

            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                    checkpoint_simple=checkpoint_simple,
                    # Early dropout parameters for decoder
                    drop_path_rate=decoder_drop_rate,
                    early_dropout_epochs=config.stochastic_depth.early_dropout_epochs,
                    dropout_schedule=config.stochastic_depth.dropout_schedule,
                    linear_decay=config.stochastic_depth.linear_decay_to_zero,
                )
            )
            layers.append(
                create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
            )

        # Final conv block - typically no dropout on final layer
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
                checkpoint_simple=checkpoint_simple,
                # No dropout on final block to preserve output quality
                drop_path_rate=0.0,
                early_dropout_epochs=0,
            )
        )

        # Final output conv
        layers.append(nn.Conv2d(b, config.n_out, config.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.corrector = Correctors(config.corrector, hist, area_weights, static_data)
        self.num_steps = int(len(config.ch_width) - 1)

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        fts_input = fts.clone().detach()
        skip_inputs: list[torch.Tensor] = []
        for i in range(self.num_steps):
            skip_inputs.append(torch.zeros_like(fts))
        count = 0
        for layer in self.layers:
            crop: torch.Size | np.ndarray = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            if self.checkpoint_all:
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)  # type: ignore
            else:
                fts = layer(fts)
            if count < self.num_steps:
                if isinstance(layer, CoreBlock):
                    skip_inputs[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(layer, BilinearUpsample) or isinstance(
                    layer, TransposedConvUpsample
                ):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        skip_inputs[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += skip_inputs[int(2 * self.num_steps - count - 1)]
                    count += 1
        fts = self.corrector(fts_input, fts)
        return torch.where(self.wet, fts, 0.0)
