import torch
import torch.nn as nn


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
    """

    def __init__(
        self,
        noise_channels: int,
        hidden_dim: int,
        output_dim: int,
        noise_shape: tuple[int, int],
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.noise_shape = noise_shape

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
        return torch.randn(
            batch_size,
            self.noise_channels,
            self.noise_shape[0],
            self.noise_shape[1],
            device=device,
            dtype=dtype,
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

        return conditioning


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

        return modulated
