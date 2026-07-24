# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Test whether coarse latent channels retain subpatch information for dynamics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

if __package__:
    from scripts.probe_coarse_latent import (
        CoarseLatentProbe,
        DecoderName,
        EncoderName,
        ProbeConfig,
        coarse_resolution,
        make_resolution,
    )
else:
    from probe_coarse_latent import (  # type: ignore[import-not-found, no-redef]
        CoarseLatentProbe,
        DecoderName,
        EncoderName,
        ProbeConfig,
        coarse_resolution,
        make_resolution,
    )


@dataclass(frozen=True)
class DynamicsConfig:
    encoder: EncoderName
    decoder: DecoderName
    coarse_height: int
    coarse_width: int
    patch_height: int
    patch_width: int
    channels: int
    latent_channels: int
    batch_size: int
    eval_samples: int
    steps: int
    learning_rate: float
    reconstruction_weight: float
    anomaly_amplitude: float
    anomaly_modes: int
    moment_count: int
    seed: int
    device: str


def parse_args() -> DynamicsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=("mean", "perceiver", "attention", "resolved", "moments"),
        default="resolved",
    )
    parser.add_argument(
        "--decoder",
        choices=("bilinear", "coordinate", "anchored", "hybrid"),
        default="anchored",
    )
    parser.add_argument("--coarse-height", type=int, default=12)
    parser.add_argument("--coarse-width", type=int, default=12)
    parser.add_argument("--patch-height", type=int, default=3)
    parser.add_argument("--patch-width", type=int, default=5)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--anomaly-amplitude", type=float, default=1.0)
    parser.add_argument("--anomaly-modes", type=int, default=8)
    parser.add_argument("--moment-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.patch_height < 2 or args.patch_width < 3:
        parser.error("Counterfactual patches require at least 2x3 cells.")
    if args.anomaly_modes < 1:
        parser.error("At least one anomaly mode is required.")
    return DynamicsConfig(**vars(args))


def probe_config(config: DynamicsConfig) -> ProbeConfig:
    return ProbeConfig(
        encoder=config.encoder,
        decoder=config.decoder,
        coarse_height=config.coarse_height,
        coarse_width=config.coarse_width,
        input_channels=config.channels,
        latent_channels=config.latent_channels,
        batch_size=config.batch_size,
        eval_samples=config.eval_samples,
        steps=config.steps,
        learning_rate=config.learning_rate,
        attention_heads=4,
        attention_dim_head=32,
        moment_count=config.moment_count,
        neighborhood_radius=1,
        position_bias_strength=8.0,
        spectral_decay=0.0,
        seed=config.seed,
        device=config.device,
    )


def anomaly_basis(
    patch_height: int,
    patch_width: int,
    modes: int,
    *,
    device: str,
) -> torch.Tensor:
    y = (torch.arange(patch_height, device=device) + 0.5) / patch_height
    x = (torch.arange(patch_width, device=device) + 0.5) / patch_width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    basis = []
    for index in range(modes):
        ky = index // 3 + 1
        kx = index % 3 + 1
        value = torch.sin(2 * torch.pi * (ky * yy + kx * xx))
        value = value + 0.5 * torch.cos(2 * torch.pi * (kx * xx - ky * yy + index / 7))
        basis.append(value - value.mean())
    stacked = torch.stack(basis)
    return stacked / stacked.square().mean(dim=(-2, -1), keepdim=True).sqrt()


def synthesize_state(
    config: DynamicsConfig,
    *,
    samples: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(
        samples,
        config.channels,
        config.coarse_height,
        config.coarse_width,
        generator=generator,
    ).to(config.device)
    coefficients = torch.randn(
        samples,
        config.channels,
        config.coarse_height,
        config.coarse_width,
        config.anomaly_modes,
        generator=generator,
    ).to(config.device)
    basis = anomaly_basis(
        config.patch_height,
        config.patch_width,
        config.anomaly_modes,
        device=config.device,
    )
    anomaly = torch.einsum("bchwm,mij->bchwij", coefficients, basis)
    anomaly = anomaly / config.anomaly_modes**0.5
    patches = coarse[..., None, None] + config.anomaly_amplitude * anomaly
    return rearrange(patches, "b c h w ph pw -> b c (h ph) (w pw)")


def patch_mean(
    x: torch.Tensor,
    coarse_height: int,
    coarse_width: int,
) -> torch.Tensor:
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b c h w ph pw",
        h=coarse_height,
        w=coarse_width,
    ).mean(dim=(-2, -1))


class CoarseResidualProcessor(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels * 2
        self.input_projection = nn.Conv2d(channels, hidden, kernel_size=3)
        self.hidden = nn.Conv2d(hidden, hidden, kernel_size=3)
        self.output_projection = nn.Conv2d(hidden, channels, kernel_size=3)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def pad(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return F.pad(x, (0, 0, 1, 1), mode="replicate")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.input_projection(self.pad(latent)))
        update = F.gelu(self.hidden(self.pad(update)))
        update = self.output_projection(self.pad(update))
        return latent + self.scale * update


class DynamicsProbe(nn.Module):
    def __init__(self, config: DynamicsConfig) -> None:
        super().__init__()
        self.inverse = CoarseLatentProbe(probe_config(config))
        self.processor = CoarseResidualProcessor(config.latent_channels)

    def encode(
        self,
        source: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.inverse.encoder(source, resolution)

    def decode(
        self,
        latent: torch.Tensor,
        physical_resolution: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        latent_resolution = coarse_resolution(
            physical_resolution,
            latent.shape[-2],
            latent.shape[-1],
        )
        return self.inverse.decoder(
            latent,
            physical_resolution,
            source_resolution=latent_resolution,
        )

    def forward(
        self,
        source: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(source, resolution)
        reconstruction = self.decode(latent, resolution)
        forecast_latent = self.processor(latent)
        forecast = self.decode(forecast_latent, resolution)
        return reconstruction, forecast, latent


def counterfactual_pair(
    config: DynamicsConfig,
    *,
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    amplitude = torch.randn(
        samples,
        config.channels,
        config.coarse_height,
        config.coarse_width,
        generator=generator,
    ).to(config.device)
    base = torch.randn(
        samples,
        config.channels,
        config.coarse_height,
        config.coarse_width,
        generator=generator,
    ).to(config.device)
    pattern_a = torch.full(
        (config.patch_height, config.patch_width),
        -1.0 / (config.patch_height * config.patch_width - 1),
        device=config.device,
    )
    pattern_b = pattern_a.clone()
    pattern_a[:, -1] = 1.0 / config.patch_height
    pattern_b[:, -2] = 1.0 / config.patch_height
    pattern_a = pattern_a - pattern_a.mean()
    pattern_b = pattern_b - pattern_b.mean()
    patches_a = base[..., None, None] + amplitude[..., None, None] * pattern_a
    patches_b = base[..., None, None] + amplitude[..., None, None] * pattern_b
    return (
        rearrange(patches_a, "b c h w ph pw -> b c (h ph) (w pw)"),
        rearrange(patches_b, "b c h w ph pw -> b c (h ph) (w pw)"),
    )


def normalized_mse(output: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (F.mse_loss(output, target) / target.var(unbiased=False).clamp_min(1e-12)).cpu()
    )


def evaluate(
    model: DynamicsProbe,
    config: DynamicsConfig,
) -> dict[str, float]:
    resolution = make_resolution(
        config.coarse_height * config.patch_height,
        config.coarse_width * config.patch_width,
    )
    model.eval()
    with torch.no_grad():
        source = synthesize_state(
            config,
            samples=config.eval_samples,
            seed=config.seed + 10_000_019,
        )
        target = torch.roll(source, shifts=1, dims=-1)
        reconstruction, forecast, _ = model(source, resolution)
        target_coarse = patch_mean(target, config.coarse_height, config.coarse_width)
        forecast_coarse = patch_mean(
            forecast, config.coarse_height, config.coarse_width
        )

        state_a, state_b = counterfactual_pair(
            config,
            samples=config.eval_samples,
            seed=config.seed + 20_000_033,
        )
        target_a = torch.roll(state_a, shifts=1, dims=-1)
        target_b = torch.roll(state_b, shifts=1, dims=-1)
        latent_a = model.encode(state_a, resolution)
        latent_b = model.encode(state_b, resolution)
        forecast_a = model.decode(model.processor(latent_a), resolution)
        forecast_b = model.decode(model.processor(latent_b), resolution)
        target_pair_difference = patch_mean(
            target_a, config.coarse_height, config.coarse_width
        ) - patch_mean(target_b, config.coarse_height, config.coarse_width)
        predicted_pair_difference = patch_mean(
            forecast_a, config.coarse_height, config.coarse_width
        ) - patch_mean(forecast_b, config.coarse_height, config.coarse_width)
        target_pair_rms = target_pair_difference.square().mean().sqrt()
        predicted_pair_rms = predicted_pair_difference.square().mean().sqrt()
        return {
            "reconstruction_normalized_mse": normalized_mse(reconstruction, source),
            "forecast_normalized_mse": normalized_mse(forecast, target),
            "coarse_forecast_normalized_mse": normalized_mse(
                forecast_coarse, target_coarse
            ),
            "counterfactual_initial_patch_mean_max_difference": float(
                (
                    patch_mean(state_a, config.coarse_height, config.coarse_width)
                    - patch_mean(state_b, config.coarse_height, config.coarse_width)
                )
                .abs()
                .max()
                .cpu()
            ),
            "counterfactual_latent_rms_difference": float(
                (latent_a - latent_b).square().mean().sqrt().cpu()
            ),
            "counterfactual_target_coarse_difference_rms": float(target_pair_rms.cpu()),
            "counterfactual_predicted_coarse_difference_rms": float(
                predicted_pair_rms.cpu()
            ),
            "counterfactual_response_ratio": float(
                (predicted_pair_rms / target_pair_rms.clamp_min(1e-12)).cpu()
            ),
            "counterfactual_difference_normalized_mse": normalized_mse(
                predicted_pair_difference, target_pair_difference
            ),
            "processor_scale_mean_abs": float(model.processor.scale.abs().mean().cpu()),
        }


def train(config: DynamicsConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    model = DynamicsProbe(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    resolution = make_resolution(
        config.coarse_height * config.patch_height,
        config.coarse_width * config.patch_width,
    )
    started = time.perf_counter()
    trajectory: list[dict[str, float]] = []
    report_every = max(1, config.steps // 10)
    model.train()
    for step in range(config.steps):
        source = synthesize_state(
            config,
            samples=config.batch_size,
            seed=config.seed * 1_000_003 + step + 1,
        )
        target = torch.roll(source, shifts=1, dims=-1)
        optimizer.zero_grad()
        reconstruction, forecast, _ = model(source, resolution)
        reconstruction_loss = F.mse_loss(reconstruction, source)
        forecast_loss = F.mse_loss(forecast, target)
        loss = forecast_loss + config.reconstruction_weight * reconstruction_loss
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % report_every == 0:
            trajectory.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "reconstruction_mse": float(reconstruction_loss.detach().cpu()),
                    "forecast_mse": float(forecast_loss.detach().cpu()),
                    "processor_scale_mean_abs": float(
                        model.processor.scale.detach().abs().mean().cpu()
                    ),
                }
            )
    return {
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation": evaluate(model, config),
        "trajectory": trajectory,
    }


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
