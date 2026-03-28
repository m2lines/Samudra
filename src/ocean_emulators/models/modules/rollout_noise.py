import math
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn


class RolloutNoiseInjector(nn.Module):
    """Inject training-time noise into autoregressive prognostic states.

    The injector supports two complementary perturbations:
      1) additive Gaussian noise,
      2) structured signed magnitude bias concentrated around fronts.
    """

    def __init__(
        self,
        *,
        wet: torch.Tensor,
        probability: float = 1.0,
        apply_to_initial_input: bool = False,
        gaussian_std: float = 0.0,
        gaussian_mode: Literal["pixel", "lattice"] = "lattice",
        gaussian_lattice_stride: int = 8,
        gaussian_blur_kernel: int = 7,
        gaussian_blur_sigma: float = 2.0,
        structured_scale: float = 0.0,
        structured_front_power: float = 1.0,
        structured_mask_smoothing_kernel: int = 7,
        structured_seed_prob: float = 0.02,
        structured_patch_kernel: int = 19,
        structured_patch_sigma: float = 4.0,
        structured_patch_quantile: float = 0.97,
        structured_sign_mode: Literal["batch", "pixel", "patch"] = "patch",
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        if gaussian_std < 0.0:
            raise ValueError("gaussian_std must be non-negative.")
        if gaussian_lattice_stride < 1:
            raise ValueError("gaussian_lattice_stride must be >= 1.")
        if gaussian_blur_kernel < 1 or gaussian_blur_kernel % 2 == 0:
            raise ValueError("gaussian_blur_kernel must be an odd integer >= 1.")
        if gaussian_blur_sigma < 0.0:
            raise ValueError("gaussian_blur_sigma must be non-negative.")
        if structured_scale < 0.0:
            raise ValueError("structured_scale must be non-negative.")
        if structured_front_power <= 0.0:
            raise ValueError("structured_front_power must be positive.")
        if structured_mask_smoothing_kernel < 1 or structured_mask_smoothing_kernel % 2 == 0:
            raise ValueError("structured_mask_smoothing_kernel must be an odd integer >= 1.")
        if not 0.0 <= structured_seed_prob <= 1.0:
            raise ValueError("structured_seed_prob must be in [0, 1].")
        if structured_patch_kernel < 1 or structured_patch_kernel % 2 == 0:
            raise ValueError("structured_patch_kernel must be an odd integer >= 1.")
        if structured_patch_sigma <= 0.0:
            raise ValueError("structured_patch_sigma must be positive.")
        if not 0.0 <= structured_patch_quantile < 1.0:
            raise ValueError("structured_patch_quantile must be in [0, 1).")

        self.register_buffer("wet_mask", wet.bool(), persistent=False)
        self.probability = probability
        self.apply_to_initial_input = apply_to_initial_input
        self.gaussian_std = gaussian_std
        self.gaussian_mode = gaussian_mode
        self.gaussian_lattice_stride = gaussian_lattice_stride
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.structured_scale = structured_scale
        self.structured_front_power = structured_front_power
        self.structured_mask_smoothing_kernel = structured_mask_smoothing_kernel
        self.structured_seed_prob = structured_seed_prob
        self.structured_patch_kernel = structured_patch_kernel
        self.structured_patch_sigma = structured_patch_sigma
        self.structured_patch_quantile = structured_patch_quantile
        self.structured_sign_mode = structured_sign_mode

    @property
    def enabled(self) -> bool:
        return (
            self.probability > 0.0
            and (self.gaussian_std > 0.0 or self.structured_scale > 0.0)
        )

    def forward(
        self,
        prognostic: Float[torch.Tensor, "batch channel height width"],
    ) -> Float[torch.Tensor, "batch channel height width"]:
        if not self.training or not self.enabled:
            return prognostic

        noise = torch.zeros_like(prognostic)

        if self.gaussian_std > 0.0:
            noise = noise + self._gaussian_component(prognostic)

        if self.structured_scale > 0.0:
            noise = noise + self._structured_front_noise(prognostic)

        wet_mask = self.wet_mask.to(prognostic.device).unsqueeze(0)
        if wet_mask.shape[1] != prognostic.shape[1]:
            raise ValueError(
                "RolloutNoiseInjector wet mask channel count does not match prognostic "
                f"channels: {wet_mask.shape[1]} vs {prognostic.shape[1]}."
            )

        noise = torch.where(wet_mask, noise, torch.zeros_like(noise))
        sample_scale = self._sample_probability_scale(
            batch_size=prognostic.shape[0],
            device=prognostic.device,
            dtype=prognostic.dtype,
        )
        return prognostic + noise * sample_scale

    def _gaussian_component(
        self, prognostic: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch channel height width"]:
        if self.gaussian_mode == "pixel":
            return self.gaussian_std * torch.randn_like(prognostic)
        if self.gaussian_mode != "lattice":
            raise ValueError(f"Unknown gaussian_mode: {self.gaussian_mode}")

        batch, channels, height, width = prognostic.shape
        coarse_h = max(1, math.ceil(height / self.gaussian_lattice_stride))
        coarse_w = max(1, math.ceil(width / self.gaussian_lattice_stride))

        coarse = torch.randn(
            (batch, channels, coarse_h, coarse_w),
            device=prognostic.device,
            dtype=prognostic.dtype,
        )
        lattice = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        lattice = self._gaussian_blur_2d(
            lattice,
            kernel_size=self.gaussian_blur_kernel,
            sigma=self.gaussian_blur_sigma,
        )
        lattice = lattice - lattice.mean(dim=(-2, -1), keepdim=True)
        lattice_std = lattice.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        lattice = lattice / lattice_std
        return self.gaussian_std * lattice

    def _structured_front_noise(
        self, prognostic: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch channel height width"]:
        front_mask = self._front_mask(prognostic)
        seeds = self._sample_front_seeds(front_mask)
        patch_field = self._gaussian_blur_2d(
            seeds,
            kernel_size=self.structured_patch_kernel,
            sigma=self.structured_patch_sigma,
        )
        patch_field = patch_field * front_mask

        if self.structured_patch_quantile > 0.0:
            abs_patch = patch_field.abs().flatten(start_dim=1)
            threshold = torch.quantile(
                abs_patch,
                q=self.structured_patch_quantile,
                dim=1,
                keepdim=True,
            ).view(-1, 1, 1, 1)
            patch_field = torch.where(
                patch_field.abs() >= threshold,
                patch_field,
                torch.zeros_like(patch_field),
            )

        patch_norm = patch_field.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        patch_field = patch_field / patch_norm

        if self.structured_sign_mode == "batch":
            sign = self._sample_sign(
                batch_size=prognostic.shape[0],
                height=1,
                width=1,
                device=prognostic.device,
                dtype=prognostic.dtype,
            )
            patch_field = patch_field.abs() * sign
        elif self.structured_sign_mode == "pixel":
            sign = self._sample_sign(
                batch_size=prognostic.shape[0],
                height=prognostic.shape[-2],
                width=prognostic.shape[-1],
                device=prognostic.device,
                dtype=prognostic.dtype,
            )
            patch_field = patch_field.abs() * sign
        elif self.structured_sign_mode != "patch":
            raise ValueError(f"Unknown structured_sign_mode: {self.structured_sign_mode}")

        magnitude = prognostic.abs()
        return self.structured_scale * patch_field * magnitude

    def _front_mask(
        self, prognostic: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch 1 height width"]:
        if prognostic.shape[-1] > 1:
            dx = F.pad(
                torch.diff(prognostic, dim=-1),
                (0, 1, 0, 0),
                mode="replicate",
            )
        else:
            dx = torch.zeros_like(prognostic)

        if prognostic.shape[-2] > 1:
            dy = F.pad(
                torch.diff(prognostic, dim=-2),
                (0, 0, 0, 1),
                mode="replicate",
            )
        else:
            dy = torch.zeros_like(prognostic)

        front = torch.sqrt(dx.square() + dy.square() + 1e-12).mean(dim=1, keepdim=True)
        max_per_sample = front.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        front = (front / max_per_sample).pow(self.structured_front_power)

        if self.structured_mask_smoothing_kernel > 1:
            k = self.structured_mask_smoothing_kernel
            front = F.avg_pool2d(front, kernel_size=k, stride=1, padding=k // 2)

        return front.clamp(0.0, 1.0)

    def _sample_front_seeds(
        self,
        front_mask: Float[torch.Tensor, "batch 1 height width"],
    ) -> Float[torch.Tensor, "batch 1 height width"]:
        seed_prob_map = (self.structured_seed_prob * front_mask).clamp(0.0, 1.0)
        seeds = torch.where(
            torch.rand_like(front_mask) < seed_prob_map,
            torch.randn_like(front_mask),
            torch.zeros_like(front_mask),
        )
        for b in range(seeds.shape[0]):
            if torch.count_nonzero(seeds[b]) == 0:
                strongest = int(front_mask[b, 0].reshape(-1).argmax().item())
                sign = -1.0 if torch.rand((), device=front_mask.device) < 0.5 else 1.0
                seeds[b, 0].reshape(-1)[strongest] = sign
        return seeds

    def _gaussian_blur_2d(
        self,
        tensor: Float[torch.Tensor, "batch channel height width"],
        *,
        kernel_size: int,
        sigma: float,
    ) -> Float[torch.Tensor, "batch channel height width"]:
        if kernel_size <= 1 or sigma <= 0.0:
            return tensor
        radius = (kernel_size - 1) / 2.0
        coords = (
            torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - radius
        )
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
            tensor.shape[1], 1, 1, 1
        )
        return F.conv2d(
            tensor,
            kernel,
            padding=kernel_size // 2,
            groups=tensor.shape[1],
        )

    def _sample_probability_scale(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Float[torch.Tensor, "batch 1 1 1"]:
        if self.probability >= 1.0:
            return torch.ones((batch_size, 1, 1, 1), device=device, dtype=dtype)
        if self.probability <= 0.0:
            return torch.zeros((batch_size, 1, 1, 1), device=device, dtype=dtype)
        return (
            (torch.rand((batch_size, 1, 1, 1), device=device) < self.probability)
            .to(dtype=dtype)
            .view(batch_size, 1, 1, 1)
        )

    def _sample_sign(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "batch 1 height width"]:
        shape = (batch_size, 1, height, width)
        signed = torch.where(
            torch.rand(shape, device=device) < 0.5,
            -torch.ones(shape, device=device, dtype=dtype),
            torch.ones(shape, device=device, dtype=dtype),
        )
        return signed
