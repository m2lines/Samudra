from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint
import xarray as xr

from ocean_emulators.constants import Grid, TensorMap
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import DecoderFiLM, TokenConditioner, build_norm
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.device import autocast

if TYPE_CHECKING:
    from ocean_emulators.config import TokenConditioningConfig


class Samudra(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        unet: UNetBackbone,
        corrector: nn.Module | None,
        pos_channels: int,
        add_3d_coordinates: nn.Module | None,
        hist: int,
        wet: Grid,
        static_data: xr.Dataset | None,
        gradient_detach_interval: int,
        use_bfloat16: bool,
        tensor_map: TensorMap,
        token_conditioning: TokenConditioningConfig | None,
        film_norm: str,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            wet=wet,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            static_data=static_data,
            gradient_detach_interval=gradient_detach_interval,
        )

        if pos_channels > 0:
            self.positional_params = nn.Parameter(
                torch.empty(pos_channels, *wet.shape[-2:])
            )
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        self.add_3d_coordinates = add_3d_coordinates
        self.unet = unet

        self.corrector = corrector
        self.use_bfloat16 = use_bfloat16
        self.tensor_map = tensor_map

        self.token_conditioning = (
            token_conditioning if token_conditioning is not None else None
        )
        self.token_conditioning_enabled = (
            self.token_conditioning is not None and self.token_conditioning.enabled
        )

        if self.token_conditioning_enabled:
            self.decoder = nn.Identity()
        else:
            self.decoder = nn.Conv2d(unet.out_channels, out_channels, last_kernel_size)

        if self.token_conditioning_enabled:
            num_prog = len(tensor_map.prognostic_var_names)
            if out_channels % num_prog != 0:
                raise ValueError(
                    "out_channels must be divisible by the number of prognostic vars"
                )
            self.num_output_steps = out_channels // num_prog
            self.num_prognostic_vars = num_prog
            self.var_prefixes = list(tensor_map.VAR_SET)
            self.var_prefix_indices = [
                [int(i) for i in tensor_map.VAR_3D_IDX[prefix].tolist()]
                for prefix in self.var_prefixes
            ]

            self.token_conditioner = TokenConditioner(
                token_dim=unet.bottleneck_channels,
                num_prefixes=len(self.var_prefixes),
                num_timesteps=self.num_output_steps,
                num_heads=self.token_conditioning.num_heads,
                init_std=self.token_conditioning.token_init_std,
            )
            self.decoder_film = DecoderFiLM(
                token_dim=unet.bottleneck_channels,
                num_tokens=len(self.var_prefixes) * self.num_output_steps,
                block_channels=unet.decoder_film_channels,
                norm=film_norm,
                hidden_mult=self.token_conditioning.mlp_hidden_mult,
            )

            self.output_norm = build_norm(film_norm, unet.out_channels)
            self.output_film_mlps = nn.ModuleList()
            self.output_heads = nn.ModuleList()
            for prefix_idx in range(len(self.var_prefixes)):
                per_prefix_mlps = nn.ModuleList()
                per_prefix_heads = nn.ModuleList()
                out_channels_var = len(self.var_prefix_indices[prefix_idx])
                for _ in range(self.num_output_steps):
                    per_prefix_mlps.append(
                        nn.Sequential(
                            nn.Linear(
                                unet.bottleneck_channels,
                                max(2 * unet.out_channels, unet.bottleneck_channels)
                                * self.token_conditioning.mlp_hidden_mult,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                max(2 * unet.out_channels, unet.bottleneck_channels)
                                * self.token_conditioning.mlp_hidden_mult,
                                2 * unet.out_channels,
                            ),
                        )
                    )
                    per_prefix_heads.append(
                        nn.Conv2d(unet.out_channels, out_channels_var, kernel_size=1)
                    )
                self.output_film_mlps.append(per_prefix_mlps)
                self.output_heads.append(per_prefix_heads)

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        if self.corrector is not None:
            fts_input = fts.clone().detach()

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.positional_params is not None:
                pos = self.positional_params.unsqueeze(0).expand(
                    fts.shape[0], -1, -1, -1
                )
                fts = torch.cat([fts, pos], dim=1)

            if self.add_3d_coordinates is not None:
                fts = self.add_3d_coordinates(fts)

            if self.token_conditioning_enabled:
                fts, tokens = self.unet.forward_with_tokens(
                    fts, self.token_conditioner, self.decoder_film
                )
            else:
                fts = self.unet(fts)
            if not self.token_conditioning_enabled:
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
        # TODO(jder): would be nice to keep inputs in bfloat16 and
        # have the convolution use float32 internally & in output dtype.
        fts = fts.to(torch.float32)
        if self.token_conditioning_enabled:
            fts = self._decode_with_tokens(fts, tokens.to(torch.float32))
        else:
            fts = self.decoder(fts)

        if self.corrector is not None:
            fts = self.corrector(fts_input, fts)
        return torch.where(self.wet, fts, 0.0)

    def _decode_with_tokens(
        self, fts: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = fts.shape
        normed = self.output_norm(fts)
        out = torch.zeros(
            batch,
            self.out_channels,
            height,
            width,
            device=fts.device,
            dtype=fts.dtype,
        )
        for t in range(self.num_output_steps):
            time_offset = t * self.num_prognostic_vars
            for prefix_idx, _ in enumerate(self.var_prefixes):
                token = tokens[:, t, prefix_idx, :]
                gamma_beta = self.output_film_mlps[prefix_idx][t](token)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gamma = gamma.view(batch, channels, 1, 1)
                beta = beta.view(batch, channels, 1, 1)
                modulated = (1.0 + gamma) * normed + beta
                pred = self.output_heads[prefix_idx][t](modulated)
                idxs = [
                    time_offset + i for i in self.var_prefix_indices[prefix_idx]
                ]
                out[:, idxs, :, :] = pred
        return out
