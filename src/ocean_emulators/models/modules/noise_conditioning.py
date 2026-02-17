import math

import torch
import torch.nn as nn
import torch.distributed as dist


class NoiseMLP(nn.Module):
    """Two-layer perceptron that processes Gaussian noise into conditioning embeddings.

    Processes independent Gaussian noise samples ξ ~ N(0, I_n) where n equals the number
    of grid points times the number of noise channels, as described in the AIFS paper.

    Architecture: noise → Linear → GELU → Linear → LayerNorm → embeddings

    These embeddings are used in conditional layer normalizations throughout the network.

    Args:
        noise_channels: Number of input noise channels per grid point
        hidden_dim: Hidden dimension of the MLP
        output_dim: Output dimension (conditioning embedding size)
        noise_shape: Tuple of (height, width) matching the model's processor grid resolution
        warmup_steps: Number of steps to ramp noise scale from warmup_start to 1.0
        warmup_start: Initial noise scale multiplier (default: 0.1)
        warmup_schedule: 'linear' or 'cosine'
    """

    def __init__(
        self,
        noise_channels: int,
        hidden_dim: int,
        output_dim: int,
        noise_shape: tuple[int, int],
        warmup_steps: int = 10000,
        warmup_start: float = 0.1,
        warmup_schedule: str = "linear",
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.noise_shape = noise_shape

        # Warmup schedule parameters
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self.warmup_schedule = warmup_schedule
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self._last_scale: float = warmup_start

        # Input size is noise_channels * num_noise_grid_points
        input_size = noise_channels * noise_shape[0] * noise_shape[1]

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Layer normalization on the output
        self.layer_norm = nn.LayerNorm(output_dim)

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
            scale = self.warmup_start + (1.0 - self.warmup_start) * (1 - math.cos(math.pi * progress)) / 2
        else:
            raise ValueError(f"Unknown warmup schedule: {self.warmup_schedule}")
        
        return scale

    def step(self):
        """Increment step counter (call once per training step)."""
        self._step += 1

    def generate_noise(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Generate independent Gaussian noise samples ξ ~ N(0, I_n).

        For each model instance and forecast step, we generate independent Gaussian noise
        where n = noise_channels × noise_height × noise_width.

        Args:
            batch_size: Number of samples in the batch (or ensemble size)
            device: Device to generate noise on
            dtype: Data type for the noise tensor

        Returns:
            Noise tensor
        """
        # CRITICAL: Use rank-aware generator to ensure different noise per GPU
        # Without this, ensemble_parallel=true gives identical members because
        # all ranks have synchronized global RNG from set_seed()
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        
        generator = torch.Generator(device=device)
        # Seed combines: rank (for cross-GPU diversity) + random int (for per-call diversity)
        seed = rank * 1000000 + torch.randint(0, 1000000, (1,), device=device).item()
        generator.manual_seed(seed)
        
        return torch.randn(
            batch_size,
            self.noise_channels,
            self.noise_shape[0],
            self.noise_shape[1],
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        batch_size = noise.shape[0]

        # Validate noise shape matches expected
        expected_shape = (
            batch_size,
            self.noise_channels,
            self.noise_shape[0],
            self.noise_shape[1],
        )

        if noise.shape != expected_shape:
            raise ValueError(
                f"Noise shape {noise.shape} doesn't match expected {expected_shape}. "
                f"NoiseMLP was initialized with noise_shape={self.noise_shape}, but received "
                f"noise with spatial dims {noise.shape[2:]}."
            )

        noise_flat = noise.reshape(batch_size, -1)

        # Apply MLP
        embedding = self.mlp(noise_flat)

        # Apply layer normalization
        conditioning = self.layer_norm(embedding)
        
        # Apply warmup scale to the conditioning embeddings
        # This scales gamma/beta effectively since they're linear projections of conditioning
        scale = self.get_noise_scale()
        self._last_scale = scale
        conditioning = scale * conditioning

        return conditioning

    def get_stats(self) -> dict[str, float]:
        """Get warmup stats for wandb logging."""
        return {
            "noise_scale": self._last_scale,
            "step": self._step.item(),
        }


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization modulated by noise embeddings.

    Replaces standard BatchNorm with conditional normalization where:
    - The normalized features are scaled by (1 + gamma) and shifted by beta
    - gamma and beta are learned functions of the noise conditioning embeddings

    Args:
        num_features: Number of channels
        cond_dim: Dimension of the noise conditioning embeddings
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features

        # Standard batch normalization (without learnable affine parameters)
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # Learned projection from noise embeddings to scale/shift parameters
        self.noise_projection = nn.Linear(cond_dim, 2 * num_features)

        # Initialize with random weights so noise has immediate effect on output
        # Using std=1.0 for strong initial noise effect - can tune down if too much
        nn.init.normal_(self.noise_projection.weight, std=1.0)
        nn.init.zeros_(self.noise_projection.bias)

        # Stats tracking for debugging
        self._last_gamma_std: float = 0.0
        self._last_gamma_mean: float = 0.0
        self._last_beta_std: float = 0.0
        self._last_beta_mean: float = 0.0
        self._last_normalized_std: float = 0.0
        self._last_modulated_std: float = 0.0
        self._last_modulation_effect: float = 0.0  # diff between modulated and normalized

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply conditional batch normalization.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            cond: Noise conditioning embeddings of shape (batch, cond_dim)

        Returns:
            modulated: Tensor of same shape as x
        """
        # Apply standard batch normalization (without affine transform)
        normalized = self.bn(x)

        # Project noise embeddings to scale and shift parameters
        params = self.noise_projection(cond)  # (B, 2*C)
        gamma, beta = torch.chunk(params, 2, dim=1)  # Each (B, C)

        # Reshape to broadcast over spatial dimensions: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply conditional modulation: out = (1 + gamma) * normalized + beta
        modulated = (1 + gamma) * normalized + beta

        # Track stats for debugging (no grad needed)
        with torch.no_grad():
            self._last_gamma_std = gamma.std().item()
            self._last_gamma_mean = gamma.mean().item()
            self._last_beta_std = beta.std().item()
            self._last_beta_mean = beta.mean().item()
            self._last_normalized_std = normalized.std().item()
            self._last_modulated_std = modulated.std().item()
            # How much does modulation change the output?
            self._last_modulation_effect = (modulated - normalized).abs().mean().item()

        return modulated

    def get_conditioning_stats(self) -> dict[str, float]:
        """Return last gamma/beta statistics for logging."""
        return {
            "gamma_std": self._last_gamma_std,
            "gamma_mean": self._last_gamma_mean,
            "beta_std": self._last_beta_std,
            "beta_mean": self._last_beta_mean,
            "normalized_std": self._last_normalized_std,
            "modulated_std": self._last_modulated_std,
            "modulation_effect": self._last_modulation_effect,
        }
