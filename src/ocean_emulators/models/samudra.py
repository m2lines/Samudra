from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ocean_emulators.models.base import BaseModel

if TYPE_CHECKING:
    from ocean_emulators.config import SamudraConfig


class Samudra(BaseModel):
    def __init__(self, config: "SamudraConfig", hist, wet, area_weights, static_data):
        ch_width = config.ch_width.copy()
        if config.pos_channels > 0:
            ch_width[0] += config.pos_channels
        super().__init__(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            wet=wet,
            hist=hist,
            pred_residuals=config.pred_residuals,
            last_kernel_size=config.last_kernel_size,
            pad=config.pad,
            static_data=static_data,
        )

        if config.pos_channels > 0:
            self.positional_params = nn.Parameter(
                torch.empty(config.pos_channels, *wet.shape[-2:])
            )
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        layers = [
            # Add UNet core.
            config.unet.build(pad=self.pad, checkpointing=config.checkpointing),
            # Samudra "decoder".
            nn.Conv2d(
                config.unet.ch_width[1], config.out_channels, config.last_kernel_size
            ),
        ]

        # Importing locally to prevent circular import. Corrector is set to "off" more often than not.
        from ocean_emulators.models.corrector import Correctors

        self.layers = nn.ModuleList(layers)
        self.corrector = Correctors(config.corrector, hist, area_weights, static_data)

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        fts_input = fts.clone().detach()

        if self.positional_params is not None:
            pos = self.positional_params.unsqueeze(0).expand(fts.shape[0], -1, -1, -1)
            fts = torch.cat([fts, pos], dim=1)

        for layer in self.layers:
            # Circular/Globe padding
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )

            # TODO(alxmrs): Find a clean way to checkpoint the decoder Conv block.

            # Apply layer
            fts = layer(fts)

        fts = self.corrector(fts_input, fts)
        return torch.where(self.wet, fts, 0.0)
