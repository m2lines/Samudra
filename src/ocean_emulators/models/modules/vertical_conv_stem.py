"""Vertical convolution stem for depth-aware processing of 3D ocean variables.

This module introduces an inductive bias that treats depth levels as an ordered,
adjacent dimension rather than independent channels. For each 3D variable group
(e.g., thetao, so, uo, vo), a 1D convolution is applied along the depth
axis before the features are flattened back into the channel dimension for the
main U-Net backbone.

This is motivated by the observation that Samudra's flat-channel design treats
thetao_0 (2.5m) and thetao_18 (6000m) as no more related than thetao_0 and vo_17.
A vertical conv explicitly encodes that adjacent depths are neighbors, giving the
model a structured prior on vertical locality.

Design decisions:
- The 1D conv is shared across all 3D variable types by default. This keeps
  parameters low and encodes the assumption that vertical locality is a universal
  physical prior. A per-variable option is available via ``shared_weights=False``.
- A residual connection is used (depth dimension is always preserved).
- 2D surface variables and boundary variables pass through unchanged.
"""

from typing import cast

import torch
import torch.nn as nn


class VerticalConvStem(nn.Module):
    """Applies 1D convolutions along the depth axis for 3D ocean variables.

    Given an input tensor of shape ``(batch, channels, lat, lon)`` this module:

    1. Splits channels into 3D variable groups, 2D variables, and boundary variables
       based on the known channel layout.
    2. For each 3D group, reshapes to ``(batch, num_3d_vars, num_depths, lat, lon)``
       and applies a small 1D convolution along the depth dimension.
    3. Adds a residual connection and recombines all channels.

    The output has the **same** channel count as the input (channel-preserving).

    Args:
        num_3d_vars: Number of distinct 3D variable types (e.g., 4 for uo, vo,
            thetao, so).
        num_depths: Number of depth levels per 3D variable (e.g., 19).
        num_2d_vars: Number of 2D prognostic variables (e.g., 1 for zos).
        num_boundary_vars: Number of boundary forcing variables per timestep.
        hist: History length (number of past timesteps included, e.g., 1).
        kernel_size: Kernel size for the depth convolution. Must be odd.
        mid_channels: Number of intermediate channels in the depth conv block.
            Defaults to ``num_depths`` if not set.
        shared_weights: If True (default), all 3D variable types share the same
            depth conv weights. If False, each variable type gets its own conv.
    """

    def __init__(
        self,
        num_3d_vars: int,
        num_depths: int,
        num_2d_vars: int,
        num_boundary_vars: int,
        hist: int,
        kernel_size: int = 3,
        mid_channels: int | None = None,
        shared_weights: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"

        self.num_3d_vars = num_3d_vars
        self.num_depths = num_depths
        self.num_2d_vars = num_2d_vars
        self.num_boundary_vars = num_boundary_vars
        self.hist = hist
        self.shared_weights = shared_weights

        mid = mid_channels if mid_channels is not None else num_depths
        pad = (kernel_size - 1) // 2

        def _make_depth_conv() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(1, mid, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv1d(mid, 1, kernel_size=kernel_size, padding=pad),
            )

        if shared_weights:
            self.depth_conv = _make_depth_conv()
        else:
            self.depth_convs = nn.ModuleList(
                [_make_depth_conv() for _ in range(num_3d_vars)]
            )

        # Pre-compute the channel layout metadata.
        self._channels_per_ts = num_3d_vars * num_depths + num_2d_vars
        self._total_ts = hist + 1
        self._num_prog_channels = self._total_ts * self._channels_per_ts
        self._num_3d_per_ts = num_3d_vars * num_depths
        self._expected_channels = self._num_prog_channels + (
            self._total_ts * num_boundary_vars
        )

        # Pre-compute and register index tensors so they move with .to(device).
        self.register_buffer(
            "_idx_3d", torch.tensor(self._compute_3d_indices(), dtype=torch.long)
        )
        self.register_buffer(
            "_idx_2d", torch.tensor(self._compute_2d_indices(), dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _compute_indices(self, start: int, length: int) -> list[int]:
        """Flat indices for a fixed-size channel block across all timesteps."""
        out: list[int] = []
        for t in range(self._total_ts):
            base = t * self._channels_per_ts + start
            out.extend(range(base, base + length))
        return out

    def _compute_3d_indices(self) -> list[int]:
        """Flat indices of all 3D variable channels across all timesteps."""
        return self._compute_indices(start=0, length=self._num_3d_per_ts)

    def _compute_2d_indices(self) -> list[int]:
        """Flat indices of all 2D variable channels across all timesteps."""
        return self._compute_indices(
            start=self._num_3d_per_ts,
            length=self.num_2d_vars,
        )

    def _boundary_slice(self, total_channels: int) -> slice:
        """Slice for boundary channels (everything after prognostic)."""
        return slice(self._num_prog_channels, total_channels)

    def _apply_depth_conv(self, x_3d_perm: torch.Tensor) -> torch.Tensor:
        """Apply shared or per-variable depth convs to (B, G, H, W, D) input."""
        B, total_groups, H, W, _ = x_3d_perm.shape

        if self.shared_weights:
            x_3d_flat = x_3d_perm.reshape(-1, 1, self.num_depths)
            return self.depth_conv(x_3d_flat)

        y_3d_perm = torch.empty_like(x_3d_perm)
        for g in range(total_groups):
            # Groups are ordered time-major as [t0v0, t0v1, ..., t1v0, t1v1, ...],
            # so modulo num_3d_vars recovers the variable-specific conv to use.
            depth_conv = self.depth_convs[g % self.num_3d_vars]
            group = x_3d_perm[:, g].reshape(-1, 1, self.num_depths)
            group_out = depth_conv(group)
            y_3d_perm[:, g] = group_out.reshape(B, H, W, self.num_depths)
        return y_3d_perm.reshape(-1, 1, self.num_depths)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vertical convolution stem.

        Args:
            x: Input tensor of shape ``(batch, channels, lat, lon)``.

        Returns:
            Tensor of shape ``(batch, channels, lat, lon)`` (same shape as input).
            Channel ordering is preserved; only 3D variable channels are modified.
        """
        B, C, H, W = x.shape
        idx_3d = cast(torch.Tensor, self._idx_3d)
        if C != self._expected_channels:
            raise ValueError(
                f"Expected {self._expected_channels} channels for stem with "
                f"hist={self.hist}, got {C}."
            )

        # Extract 3D channels and reshape into (B, T*V, D, H, W)
        x_3d = x[:, idx_3d]  # (B, T*V*D, H, W)
        total_groups = self._total_ts * self.num_3d_vars
        x_3d = x_3d.reshape(B, total_groups, self.num_depths, H, W)

        # Depth conv expects (N, 1, D).
        # Flatten spatial + batch + groups: (B * G * H * W, 1, D)
        x_3d_perm = x_3d.permute(0, 1, 3, 4, 2).contiguous()  # (B, G, H, W, D)
        y_flat = self._apply_depth_conv(x_3d_perm)

        # Reshape back: (N, 1, D) -> (B, G, H, W, D) -> (B, G, D, H, W) -> (B, G*D, H, W)
        y = y_flat.reshape(B, total_groups, H, W, self.num_depths)
        y = y.permute(0, 1, 4, 2, 3)  # (B, G, D, H, W)
        y = y.reshape(B, total_groups * self.num_depths, H, W)

        # Residual connection on 3D channels
        y = y + x[:, idx_3d]

        # Write processed 3D channels back into their original positions,
        # preserving 2D and boundary channels in place.
        out = x.clone()
        out[:, idx_3d] = y
        return out
