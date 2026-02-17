"""Stochastic Decomposition Layer (SDL) for ensemble generation.

Based on: "Stochastic Decomposition Layers for Probabilistic Weather Forecasting"

SDL injects structured noise into features via three multiplicative components:
    Fout = Fin + α · R ⊙ S ⊙ M

Where:
    R: Per-pixel Gaussian noise (sampled at full spatial resolution)
    S: Latent-derived "style" tensor (from learned linear transformation of latent Z)
    M: Learned channel-wise modulation parameter
    α: Fixed scaling parameter (non-trainable)
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class StochasticDecompositionLayer(nn.Module):
    """SDL layer that adds structured noise to features.
    
    Full SDL implementation with learnable S (style) and M (modulation) parameters:
        Fout = Fin + α · R ⊙ S ⊙ M
    
    Args:
        num_channels: Number of feature channels at this layer
        latent_dim: Dimension of latent vector Z for style projection
        alpha: Fixed noise scale (non-trainable, default: 0.1)
        warmup_steps: Number of steps to ramp noise scale from warmup_start to 1.0
        warmup_start: Initial noise scale multiplier (default: 0.1)
        warmup_schedule: 'linear' or 'cosine'
    """
    
    def __init__(
        self,
        num_channels: int,
        latent_dim: int = 64,
        alpha: float = 0.1,
        warmup_steps: int = 10000,
        warmup_start: float = 0.1,
        warmup_schedule: str = "linear",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.alpha = alpha  # Fixed, non-trainable
        
        # Warmup schedule parameters
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self.warmup_schedule = warmup_schedule
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        
        # Learnable S: Style projection from latent Z to channel-wise scales
        # S = Linear(Z) where Z ~ N(0, I)
        self.style_projection = nn.Linear(latent_dim, num_channels)
        # Initialize to produce unit variance output
        nn.init.normal_(self.style_projection.weight, std=1.0 / (latent_dim ** 0.5))
        nn.init.zeros_(self.style_projection.bias)
        
        # Learnable M: Per-channel modulation (initialized to 1.0)
        self.M = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        
        # Stats tracking for wandb logging
        self._last_noise_std: float = 0.0
        self._last_noise_mean: float = 0.0
        self._last_fts_std: float = 0.0
        self._last_scale: float = warmup_start
        self._last_S_std: float = 0.0
        self._last_S_mean: float = 0.0
    
    def get_noise_scale(self) -> float:
        """Get current noise scale based on warmup schedule."""
        step = self._step.item()
        if step >= self.warmup_steps:
            return 1.0
        
        progress = step / self.warmup_steps
        
        if self.warmup_schedule == "linear":
            scale = self.warmup_start + (1.0 - self.warmup_start) * progress
        elif self.warmup_schedule == "cosine":
            # Cosine annealing from warmup_start to 1.0
            import math
            scale = self.warmup_start + (1.0 - self.warmup_start) * (1 - math.cos(math.pi * progress)) / 2
        else:
            raise ValueError(f"Unknown warmup schedule: {self.warmup_schedule}")
        
        return scale
    
    def step(self):
        """Increment step counter (call once per training step)."""
        self._step += 1
    
    def forward(self, fts: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SDL noise injection.
        
        Args:
            fts: Input features (B, C, H, W)
            z: Optional latent vector (B, latent_dim). If None, sampled from N(0,I)
            
        Returns:
            Features with structured noise added (B, C, H, W)
        """
        B, C, H, W = fts.shape
        device = fts.device
        dtype = fts.dtype
        
        # Get current noise scale from warmup schedule
        scale = self.get_noise_scale()
        self._last_scale = scale
        
        # Get rank info for per-rank noise seeding
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        
        # CRITICAL: Use rank-aware generator to ensure different noise per GPU
        # Without this, ensemble_parallel=true gives identical members because
        # all ranks have synchronized global RNG from set_seed()
        generator = torch.Generator(device=device)
        # Seed combines: rank (for cross-GPU diversity) + random int (for per-call diversity)
        seed = rank * 1000000 + torch.randint(0, 1000000, (1,), device=device).item()
        generator.manual_seed(seed)
        
        # Generate latent Z if not provided
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device, dtype=dtype, generator=generator)
        
        # S: Style tensor from latent projection (B, C) -> (B, C, 1, 1)
        S = self.style_projection(z).view(B, C, 1, 1)
        
        # R: Per-pixel Gaussian noise
        R = torch.randn(B, C, H, W, device=device, dtype=dtype, generator=generator)
        
        # Full SDL: noise = scale * alpha * R * S * M
        noise = scale * self.alpha * R * S * self.M
        
        # Track stats for logging
        with torch.no_grad():
            self._last_noise_std = noise.std().item()
            self._last_noise_mean = noise.mean().item()
            self._last_fts_std = fts.std().item()
            self._last_S_std = S.std().item()
            self._last_S_mean = S.mean().item()
            
            # Debug: log during eval mode to diagnose train/val spread discrepancy
            if not self.training:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[SDL EVAL] scale={scale:.4f}, alpha={self.alpha:.4f}, "
                    f"R_std={R.std().item():.4f}, S_std={S.std().item():.4f}, "
                    f"M_mean={self.M.mean().item():.4f}, noise_std={self._last_noise_std:.4f}, "
                    f"fts_std={self._last_fts_std:.4f}, rank={rank}"
                )
        
        return fts + noise
    
    def get_stats(self) -> dict[str, float]:
        """Get SDL statistics for wandb logging."""
        return {
            "noise_std": self._last_noise_std,
            "noise_scale": self._last_scale,
            "step": self._step.item(),
            "noise_mean": self._last_noise_mean,
            "alpha": self.alpha,
            "fts_std": self._last_fts_std,
            "noise_to_fts_ratio": self._last_noise_std / max(self._last_fts_std, 1e-8),
            "M_mean": self.M.mean().item(),
            "M_std": self.M.std().item(),
            "S_std": self._last_S_std,
            "S_mean": self._last_S_mean,
        }
    
    def get_latent(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate a latent vector (kept for API compatibility)."""
        return torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)


class MultiScaleSDL(nn.Module):
    """Collection of SDL layers at multiple scales (following the paper's approach).
    
    The paper injects SDL at 3 decoder levels with independent latents Z1, Z2, Z3.
    This class manages multiple SDL layers and their latents.
    
    Args:
        channel_dims: List of channel dimensions at each scale (e.g., [400, 300, 250])
        latent_dim: Dimension of latent vectors (shared across scales)
        alpha: Fixed noise scale (can be per-scale or shared)
    """
    
    def __init__(
        self,
        channel_dims: list[int],
        latent_dim: int = 64,
        alpha: float | list[float] = 0.1,
    ):
        super().__init__()
        self.num_scales = len(channel_dims)
        self.latent_dim = latent_dim
        
        # Handle per-scale or shared alpha
        if isinstance(alpha, (int, float)):
            alphas = [alpha] * self.num_scales
        else:
            alphas = alpha
        
        # Create SDL layer for each scale
        self.sdl_layers = nn.ModuleList([
            StochasticDecompositionLayer(
                num_channels=ch,
                latent_dim=latent_dim,
                alpha=a,
            )
            for ch, a in zip(channel_dims, alphas)
        ])
    
    def forward_scale(self, fts: torch.Tensor, scale_idx: int, z: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SDL at a specific scale.
        
        Args:
            fts: Input features at this scale
            scale_idx: Which scale (0, 1, 2, ...)
            z: Optional latent for this scale
        """
        return self.sdl_layers[scale_idx](fts, z)
    
    def get_all_latents(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        """Generate latents for all scales."""
        return [
            layer.get_latent(batch_size, device, dtype)
            for layer in self.sdl_layers
        ]
