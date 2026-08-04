# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The spectral transform `samudra.metrics.spectra` was ported from, verbatim.

Vendored so the port can be checked against it in the test suite rather than
against numbers copied out of a run somebody else did. Nothing in `src/`
imports this; it exists to be disagreed with.

    source : YuanYuan98/Ocean_Emulator, viz_jupyter/obs_evaluation.py
    commit : 33d4ae2e
    symbols: _detrend_linear_torch, compute_isotropic_spectrum_torch

Unmodified apart from this header. Update it by re-copying those two functions,
not by editing them: the moment it is tidied it stops being the thing under
test.
"""

import numpy as np
import torch

def _detrend_linear_torch(data: torch.Tensor) -> torch.Tensor:
    bsz, channels, height, width = data.shape
    y_coords = torch.linspace(-1, 1, height, device=data.device, dtype=data.dtype)
    x_coords = torch.linspace(-1, 1, width, device=data.device, dtype=data.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    design = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x).flatten()], dim=1)

    flat = data.reshape(bsz * channels, height * width)
    coeffs, _, _, _ = torch.linalg.lstsq(design, flat.T)
    plane = (design @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(bsz * channels, height, width)
    detrended = data.reshape(bsz * channels, height, width) - plane
    return detrended.reshape(bsz, channels, height, width)


def compute_isotropic_spectrum_torch(
    data: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    num_bins: int | None = None,
    n_factor: int = 4,
    remove_mean: bool = True,
    detrend: str | None = None,
    window: str = "Hann",
    truncate: bool = True,
    cutoff_before_bins: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dim = data.dim()
    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)
    elif orig_dim == 3:
        data = data.unsqueeze(1)
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")

    bsz, channels, height, width = data.shape
    merged = bsz * channels
    lx = width * dx
    ly = height * dy

    if num_bins is None:
        num_bins = min(height, width) // n_factor

    if detrend == "linear":
        data = _detrend_linear_torch(data)
    elif detrend == "constant" or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == "hann":
        win_y = torch.hann_window(height, device=data.device, dtype=data.dtype).unsqueeze(1)
        win_x = torch.hann_window(width, device=data.device, dtype=data.dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, height, width)
        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    fft_2d = torch.fft.rfft2(data, norm="forward")
    psd_2d = (torch.abs(fft_2d) ** 2 / window_correction) * (lx * ly)

    k_x = torch.fft.rfftfreq(width, d=dx, device=data.device, dtype=data.dtype)
    k_y = torch.fft.fftfreq(height, d=dy, device=data.device, dtype=data.dtype)
    k_y_grid, k_x_grid = torch.meshgrid(k_y, k_x, indexing="ij")
    k_mag = torch.sqrt(k_x_grid**2 + k_y_grid**2)

    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)
    k_max_domain = k_mag.max()
    if truncate and cutoff_before_bins:
        k_max = min(k_max_domain, min(k_x_nyq, k_y_nyq))
    else:
        k_max = k_max_domain

    k_bins = torch.linspace(0, k_max, num_bins + 1, device=data.device, dtype=data.dtype)
    if truncate and not cutoff_before_bins:
        k_bins = k_bins[k_bins < min(k_x_nyq, k_y_nyq)]
        num_bins = k_bins.numel() - 1
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2

    k_mag_flat = k_mag.flatten()
    # bucketize maps values above the final edge to the final bin. Exclude those Fourier-corner
    # modes explicitly so they cannot contaminate the highest retained isotropic bin.
    retained_modes = k_mag_flat <= k_bins[-1]
    k_mag_binned = k_mag_flat[retained_modes]
    bin_indices = torch.bucketize(k_mag_binned, k_bins[1:-1], right=True)
    psd_flat = psd_2d.reshape(merged, k_mag_flat.shape[0])[:, retained_modes]
    binned_sum = torch.zeros(merged, num_bins, device=data.device, dtype=data.dtype)
    binned_sum.scatter_add_(dim=1, index=bin_indices.expand(merged, -1), src=psd_flat)
    binned_counts = torch.bincount(bin_indices, minlength=num_bins).to(data.dtype)
    iso = binned_sum / torch.clamp(binned_counts, min=1).unsqueeze(0)
    iso = iso * k_centers.unsqueeze(0)

    if orig_dim == 2:
        iso = iso.reshape(num_bins)
    elif orig_dim == 3:
        iso = iso.reshape(bsz, num_bins)
    else:
        iso = iso.reshape(bsz, channels, num_bins)
    return k_centers, iso
