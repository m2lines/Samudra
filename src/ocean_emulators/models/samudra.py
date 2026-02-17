import torch
import torch.nn as nn
import torch.utils.checkpoint
import xarray as xr

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.noise_conditioning import NoiseMLP
from ocean_emulators.models.modules.sdl import StochasticDecompositionLayer
from ocean_emulators.models.modules.unet_backbone import UNetBackbone


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
        noise_channels: int | None = None,
        noise_embed_dim: int | None = None,
        noise_shape: tuple[int, int] | None = None,
        noise_alpha: float | None = None,
        noise_warmup_steps: int | None = None,
        noise_warmup_start: float | None = None,
        noise_warmup_schedule: str | None = None,
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

        self.noise_channels = noise_channels
        self.noise_embed_dim = noise_embed_dim
        self._noise_enabled = True  # Can be disabled for deterministic inference
        
        # SDL for output-level noise injection with warmup schedule
        # Noise scale ramps up during training so model can't learn to suppress early
        self.noise_alpha = noise_alpha
        if noise_alpha is not None:
            self.sdl = StochasticDecompositionLayer(
                num_channels=out_channels,  # Inject at output level
                latent_dim=64,
                alpha=noise_alpha,
                warmup_steps=noise_warmup_steps or 10000,
                warmup_start=noise_warmup_start or 0.1,
                warmup_schedule=noise_warmup_schedule or "linear",
            )
        else:
            self.sdl = None

        if pos_channels > 0:
            self.positional_params = nn.Parameter(
                torch.empty(pos_channels, *wet.shape[-2:])
            )
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        # Initialize noise MLP if noise conditioning is enabled
        if noise_channels is not None and noise_embed_dim is not None:
            if noise_shape is None:
                raise ValueError(
                    "Noise conditioning enabled but no noise_shape was provided. "
                    "Set `model.noise_resolution` in the configuration to supply a noise shape."
                )
            self.noise_mlp: NoiseMLP | None = NoiseMLP(
                noise_channels=noise_channels,
                hidden_dim=noise_embed_dim * 2,  # Hidden dim is 2x output
                output_dim=noise_embed_dim,
                noise_shape=noise_shape,
                warmup_steps=noise_warmup_steps or 10000,
                warmup_start=noise_warmup_start or 0.1,
                warmup_schedule=noise_warmup_schedule or "linear",
            )
        else:
            self.noise_mlp = None

        layers = [
            # Add UNet core.
            unet,
            # Samudra "decoder".
            nn.Conv2d(unet.out_channels, out_channels, last_kernel_size),
        ]

        # This preserves backwards compatibility with previous checkpoints.
        if add_3d_coordinates is not None:
            layers.insert(0, add_3d_coordinates)

        self.layers = nn.ModuleList(layers)
        self.corrector = corrector

    def set_noise_enabled(self, enabled: bool) -> None:
        """Enable or disable noise injection.
        
        When disabled, the model runs deterministically (no uncertainty estimation).
        Useful for stable autoregressive rollouts during inference.
        """
        self._noise_enabled = enabled
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Noise injection {'enabled' if enabled else 'disabled'}")

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        fts_input = fts.clone().detach()

        # Generate and process noise if noise conditioning is enabled AND noise is not disabled
        cond = None
        if self.noise_mlp is not None and self._noise_enabled:
            noise = self.noise_mlp.generate_noise(
                batch_size=fts.shape[0],
                device=fts.device,
                dtype=fts.dtype,
            )
            cond = self.noise_mlp(noise)  # (B, noise_embed_dim)
            
            # DEBUG: Check conditioning variance
            # print(f"[DEBUG] noise std: {noise.std().item():.4f}, cond std: {cond.std().item():.4f}, cond mean: {cond.mean().item():.4f}")

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
            # Apply layer with conditioning if it's the UNet
            if isinstance(layer, UNetBackbone) and cond is not None:
                fts = layer(fts, cond)
            else:
                fts = layer(fts)

        if self.corrector is not None:
            fts = self.corrector(fts_input, fts)
        
        # Apply wet mask first
        fts = torch.where(self.wet, fts, 0.0)
        
        # SDL noise injection at OUTPUT level (after everything)
        # Applied during both training and eval to enable ensemble generation
        # Skip if noise is disabled for deterministic inference
        if self.sdl is not None and self._noise_enabled:
            fts = self.sdl(fts)
        
        return fts
