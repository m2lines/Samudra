# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fit the production Perceiver decoder to a controlled synthetic copy task."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from perceiver_pytorch.perceiver_io import Attention, FeedForward, PerceiverIO, PreNorm
from torch import nn

from samudra.models.modules.decoder import (
    LocalCoordinateAttentionCorrection,
    PerceiverDecoder,
    ResampleAttentionResidualDecoder,
    ResampleProjectionDecoder,
)


@dataclass(frozen=True)
class ProbeConfig:
    architecture: str
    data_mode: str
    grid_size: int
    output_grid_size: int
    output_longitude_shift: float
    target_channels: int
    input_channels: int
    samples: int
    steps: int
    learning_rate: float
    depth: int
    num_latents: int
    latent_dim: int
    queries_dim: int
    cross_heads: int
    cross_dim_head: int
    position_bias_strength: float
    window_patches: int
    context_patches: int
    fresh_batches: bool
    eval_samples: int
    seed: int
    device: str


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=(
            "perceiver",
            "direct-cross-attention",
            "anchored-cross-attention",
            "resample-projection",
            "coordinate-resample-projection",
            "resample-attention-residual",
        ),
        default="perceiver",
    )
    parser.add_argument(
        "--data-mode", choices=("random", "analytic"), default="analytic"
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--output-grid-size",
        type=int,
        default=0,
        help="Output-grid side length; zero matches --grid-size.",
    )
    parser.add_argument(
        "--output-longitude-shift",
        type=float,
        default=0.0,
        help="Output longitude shift measured as a fraction of one output cell.",
    )
    parser.add_argument("--target-channels", type=int, default=16)
    parser.add_argument(
        "--input-channels",
        type=int,
        default=32,
        help="Width of the learned representation passed to the decoder.",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-latents", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--queries-dim", type=int, default=64)
    parser.add_argument("--cross-heads", type=int, default=1)
    parser.add_argument("--cross-dim-head", type=int, default=64)
    parser.add_argument("--position-bias-strength", type=float, default=16.0)
    parser.add_argument(
        "--window-patches",
        type=int,
        default=0,
        help="Decode-window side length; zero uses the full synthetic grid.",
    )
    parser.add_argument("--context-patches", type=int, default=1)
    parser.add_argument(
        "--fresh-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = ProbeConfig(**vars(args))
    if config.input_channels < 1 or config.target_channels < 1:
        parser.error("Channel counts must be positive")
    if config.grid_size < 1:
        parser.error("--grid-size must be positive")
    if config.output_grid_size < 0:
        parser.error("--output-grid-size cannot be negative")
    if config.data_mode == "random" and config.output_grid_size not in (
        0,
        config.grid_size,
    ):
        parser.error("Cross-resolution probes require --data-mode=analytic")
    if config.data_mode == "random" and config.output_longitude_shift != 0:
        parser.error("Shifted-grid probes require --data-mode=analytic")
    return config


def make_resolution(
    size: int, *, longitude_shift: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    spacing_lat = 180.0 / size
    spacing_lon = 360.0 / size
    latitude = torch.linspace(
        -90.0 + spacing_lat / 2,
        90.0 - spacing_lat / 2,
        size,
    )
    longitude = (
        torch.arange(size, dtype=torch.float32) + longitude_shift
    ) * spacing_lon
    return latitude, longitude


def output_grid_size(config: ProbeConfig) -> int:
    return config.output_grid_size or config.grid_size


def evaluate_analytic_fields(
    coefficients: torch.Tensor,
    resolution: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    latitude, longitude = resolution
    latitude_radians = torch.deg2rad(latitude)[:, None]
    longitude_radians = torch.deg2rad(longitude)[None, :]
    basis = torch.stack(
        (
            torch.ones_like(latitude_radians + longitude_radians),
            torch.sin(latitude_radians).expand(-1, len(longitude)),
            torch.cos(longitude_radians).expand(len(latitude), -1),
            torch.cos(latitude_radians) * torch.sin(2 * longitude_radians),
            torch.sin(2 * latitude_radians) * torch.cos(longitude_radians),
        ),
        dim=0,
    )
    return torch.einsum("bcf,fhw->bchw", coefficients, basis) / math.sqrt(
        basis.shape[0]
    )


def make_copy_data(
    config: ProbeConfig, *, seed_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    if config.data_mode == "random":
        target = torch.randn(
            config.samples,
            config.target_channels,
            config.grid_size,
            config.grid_size,
            generator=generator,
        )
        source = target
    else:
        coefficients = torch.randn(
            config.samples,
            config.target_channels,
            5,
            generator=generator,
        )
        source = evaluate_analytic_fields(
            coefficients, make_resolution(config.grid_size)
        )
        target = evaluate_analytic_fields(
            coefficients,
            make_resolution(
                output_grid_size(config),
                longitude_shift=config.output_longitude_shift,
            ),
        )
    return source.to(config.device), target.to(config.device)


class DirectCrossAttentionIO(nn.Module):
    """Decode output queries directly from spatial data tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        queries_dim: int,
        output_dim: int,
        heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()
        self.cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim,
                input_dim,
                heads=heads,
                dim_head=dim_head,
            ),
            context_dim=input_dim,
        )
        self.feed_forward = PreNorm(queries_dim, FeedForward(queries_dim))
        self.to_logits = nn.Linear(queries_dim, output_dim)

    def forward(self, data: torch.Tensor, *, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(data.shape[0], -1, -1)
        decoded = queries + self.cross_attn(queries, context=data)
        decoded = decoded + self.feed_forward(decoded)
        return self.to_logits(decoded)


class AnchoredCrossAttentionIO(nn.Module):
    """Direct cross-attention with a strong relative-position routing prior."""

    def __init__(
        self,
        *,
        input_dim: int,
        queries_dim: int,
        output_dim: int,
        heads: int,
        dim_head: int,
        context_patches: int,
        position_bias_strength: float,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.context_patches = context_patches
        self.position_bias_strength = position_bias_strength
        self.query_norm = nn.LayerNorm(queries_dim)
        self.context_norm = nn.LayerNorm(input_dim)
        self.to_q = nn.Linear(queries_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(input_dim, 2 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, queries_dim)
        self.feed_forward = PreNorm(queries_dim, FeedForward(queries_dim))
        self.to_logits = nn.Linear(queries_dim, output_dim)
        self.last_attention: torch.Tensor | None = None

    def relative_position_bias(
        self,
        query_count: int,
        key_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_side = math.isqrt(query_count)
        key_side = math.isqrt(key_count)
        if query_side**2 != query_count or key_side**2 != key_count:
            raise ValueError("The synthetic anchored probe requires square windows.")
        source_side = key_side - 2 * self.context_patches
        if source_side < 1:
            raise ValueError("Context leaves no interior source grid.")
        query_axis = (
            (torch.arange(query_side, device=device, dtype=dtype) + 0.5)
            * source_side
            / query_side
        )
        key_axis = (
            torch.arange(key_side, device=device, dtype=dtype)
            - self.context_patches
            + 0.5
        )
        query_positions = torch.stack(
            torch.meshgrid(query_axis, query_axis, indexing="ij"), dim=-1
        ).flatten(0, 1)
        key_positions = torch.stack(
            torch.meshgrid(key_axis, key_axis, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coordinate_delta = query_positions[:, None] - key_positions[None, :]
        coordinate_delta[..., 1] = (
            torch.remainder(coordinate_delta[..., 1] + source_side / 2, source_side)
            - source_side / 2
        )
        distance_squared = coordinate_delta.square().sum(dim=-1)
        return -self.position_bias_strength * distance_squared

    def forward(self, data: torch.Tensor, *, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(data.shape[0], -1, -1)
        batch, query_count, _ = queries.shape
        key_count = data.shape[1]
        q = (
            self.to_q(self.query_norm(queries))
            .view(batch, query_count, self.heads, -1)
            .transpose(1, 2)
        )
        k, v = self.to_kv(self.context_norm(data)).chunk(2, dim=-1)
        k = k.view(batch, key_count, self.heads, -1).transpose(1, 2)
        v = v.view(batch, key_count, self.heads, -1).transpose(1, 2)
        bias = self.relative_position_bias(
            query_count,
            key_count,
            device=data.device,
            dtype=data.dtype,
        )
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = (logits + bias[None, None]).softmax(dim=-1)
        self.last_attention = attention.detach()
        decoded = torch.matmul(attention, v).transpose(1, 2).flatten(2)
        decoded = queries + self.to_out(decoded)
        decoded = decoded + self.feed_forward(decoded)
        return self.to_logits(decoded)


class CoordinateAttentionDecoder(nn.Module):
    """Use the production position-anchored attention correction without a base."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        heads: int,
        dim_head: int,
        neighborhood_radius: int,
        position_bias_strength: float,
    ) -> None:
        super().__init__()
        self.attention = LocalCoordinateAttentionCorrection(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            heads=heads,
            dim_head=dim_head,
            neighborhood_radius=neighborhood_radius,
            position_bias_strength=position_bias_strength,
        )
        nn.init.xavier_uniform_(self.attention.output_projection.weight)

    def forward(
        self,
        x: torch.Tensor,
        resolution: tuple[torch.Tensor, torch.Tensor],
        *,
        source_resolution: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if source_resolution is None:
            raise ValueError("Position-anchored attention requires source coordinates.")
        return self.attention(x, source_resolution, resolution)


def make_decoder(config: ProbeConfig) -> nn.Module:
    if config.architecture in (
        "resample-projection",
        "coordinate-resample-projection",
    ):
        return ResampleProjectionDecoder(
            in_channels=config.input_channels,
            out_channels=config.target_channels,
            coordinate_resampling=(
                config.architecture == "coordinate-resample-projection"
            ),
        ).to(config.device)
    if config.architecture == "resample-attention-residual":
        base = ResampleProjectionDecoder(
            in_channels=config.input_channels,
            out_channels=config.target_channels,
            coordinate_resampling=True,
        )
        correction = LocalCoordinateAttentionCorrection(
            in_channels=config.input_channels,
            out_channels=config.target_channels,
            hidden_dim=config.queries_dim,
            heads=config.cross_heads,
            dim_head=config.cross_dim_head,
            neighborhood_radius=config.context_patches,
            position_bias_strength=config.position_bias_strength,
        )
        return ResampleAttentionResidualDecoder(base, correction).to(config.device)
    if config.architecture == "anchored-cross-attention":
        return CoordinateAttentionDecoder(
            in_channels=config.input_channels,
            out_channels=config.target_channels,
            hidden_dim=config.queries_dim,
            heads=config.cross_heads,
            dim_head=config.cross_dim_head,
            neighborhood_radius=config.context_patches,
            position_bias_strength=config.position_bias_strength,
        ).to(config.device)
    if config.architecture == "perceiver":
        decoder_io: nn.Module = PerceiverIO(
            depth=config.depth,
            dim=config.input_channels,
            queries_dim=config.queries_dim,
            logits_dim=config.target_channels,
            num_latents=config.num_latents,
            latent_dim=config.latent_dim,
            cross_heads=config.cross_heads,
            cross_dim_head=config.cross_dim_head,
            weight_tie_layers=True,
            decoder_ff=True,
        )
    elif config.architecture == "direct-cross-attention":
        decoder_io = DirectCrossAttentionIO(
            input_dim=config.input_channels,
            queries_dim=config.queries_dim,
            output_dim=config.target_channels,
            heads=config.cross_heads,
            dim_head=config.cross_dim_head,
        )
    else:
        raise AssertionError(f"Unhandled architecture: {config.architecture}")
    extent = 180.0 / config.grid_size, 360.0 / config.grid_size
    window_patches = config.window_patches or config.grid_size
    return PerceiverDecoder(
        in_channels=config.input_channels,
        out_channels=config.target_channels,
        patch_extent=extent,
        queries_dim=config.queries_dim,
        perceiver_io=decoder_io,
        window_patches=window_patches,
        context_patches=config.context_patches,
        window_batch_size=None,
    ).to(config.device)


class LearnedInverseProbe(nn.Module):
    """A learned pointwise encoder followed by one of the decoder candidates."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(config.target_channels, config.input_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.input_channels, config.input_channels, kernel_size=1),
        )
        self.decoder = make_decoder(config)

    def forward(
        self,
        source: torch.Tensor,
        source_resolution: tuple[torch.Tensor, torch.Tensor],
        output_resolution: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        representation = self.encoder(source)
        return self.decoder(
            representation,
            output_resolution,
            source_resolution=source_resolution,
        )


def spatial_spread_ratio(output: torch.Tensor, target: torch.Tensor) -> float:
    output_spread = output.var(dim=(-2, -1), unbiased=False).mean()
    target_spread = target.var(dim=(-2, -1), unbiased=False).mean()
    return float((output_spread / target_spread).detach().cpu())


def field_diagnostics(output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Report amplitude, bias, and high-wavenumber fidelity."""
    target_std = target.std(unbiased=False).clamp_min(1e-12)
    output_std = output.std(unbiased=False)
    normalized_bias = (output.mean() - target.mean()) / target_std

    output_power = (
        torch.fft.rfft2(output.to(torch.float32), norm="ortho").abs().square()
    )
    target_power = (
        torch.fft.rfft2(target.to(torch.float32), norm="ortho").abs().square()
    )
    latitude_frequency = torch.fft.fftfreq(output.shape[-2], device=output.device)
    longitude_frequency = torch.fft.rfftfreq(output.shape[-1], device=output.device)
    radius = torch.sqrt(
        latitude_frequency[:, None].square() + longitude_frequency[None, :].square()
    )
    high_wavenumber = radius >= 0.25
    high_wavenumber_ratio = output_power[..., high_wavenumber].mean() / target_power[
        ..., high_wavenumber
    ].mean().clamp_min(1e-12)
    return {
        "standard_deviation_ratio": float((output_std / target_std).cpu()),
        "normalized_mean_bias": float(normalized_bias.cpu()),
        "high_wavenumber_power_ratio": float(high_wavenumber_ratio.cpu()),
    }


def attention_diagnostics(
    decoder: PerceiverDecoder,
    features: torch.Tensor,
    resolution: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Measure attention concentration and its approximate spatial transport."""
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            attention = cast(Any, module)
            queries = attention.to_q(args[0])
            keys, _ = attention.to_kv(kwargs["context"]).chunk(2, dim=-1)
            batch, query_count, _ = queries.shape
            key_count = keys.shape[1]
            queries = queries.view(batch, query_count, attention.heads, -1).transpose(
                1, 2
            )
            keys = keys.view(batch, key_count, attention.heads, -1).transpose(1, 2)
            logits = torch.matmul(queries, keys.transpose(-1, -2)) * attention.scale
            captured[name] = logits.softmax(dim=-1).detach()

        return hook

    decoder_io = cast(Any, decoder.perceiver_io)
    handles: list[Any]
    if isinstance(decoder_io, AnchoredCrossAttentionIO):
        handles = []
    elif isinstance(decoder_io, DirectCrossAttentionIO):
        handles = [
            decoder_io.cross_attn.fn.register_forward_pre_hook(
                capture("direct"), with_kwargs=True
            )
        ]
    else:
        input_attention = decoder_io.cross_attend_blocks[0].fn
        output_attention = decoder_io.decoder_cross_attn.fn
        handles = [
            input_attention.register_forward_pre_hook(
                capture("input"), with_kwargs=True
            ),
            output_attention.register_forward_pre_hook(
                capture("output"), with_kwargs=True
            ),
        ]
    try:
        with torch.no_grad():
            decoder(features, resolution)
    finally:
        for handle in handles:
            handle.remove()
    if isinstance(decoder_io, AnchoredCrossAttentionIO):
        assert decoder_io.last_attention is not None
        captured["direct"] = decoder_io.last_attention

    metrics: dict[str, float] = {}
    for name, weights in captured.items():
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        key_count = weights.shape[-1]
        normalized_entropy = (
            entropy / math.log(key_count)
            if key_count > 1
            else torch.zeros_like(entropy)
        )
        singular_values = torch.linalg.svdvals(weights.flatten(0, 1).to(torch.float32))
        stable_rank = singular_values.square().sum(dim=-1) / singular_values[
            :, 0
        ].square().clamp_min(1e-12)
        metrics[f"{name}_attention_normalized_entropy"] = float(
            normalized_entropy.mean().cpu()
        )
        metrics[f"{name}_attention_max_mass"] = float(
            weights.max(dim=-1).values.mean().cpu()
        )
        metrics[f"{name}_attention_stable_rank"] = float(stable_rank.mean().cpu())

    if "direct" in captured:
        approximate_transport = captured["direct"].mean(dim=1)
    else:
        input_weights = captured["input"].mean(dim=1)
        output_weights = captured["output"].mean(dim=1)
        approximate_transport = output_weights @ input_weights
    if approximate_transport.shape[-2] == approximate_transport.shape[-1]:
        diagonal = approximate_transport.diagonal(dim1=-2, dim2=-1)
        predicted_source = approximate_transport.argmax(dim=-1)
        expected_source = torch.arange(
            approximate_transport.shape[-1], device=predicted_source.device
        )
        metrics["approximate_transport_diagonal_mass"] = float(diagonal.mean().cpu())
        metrics["approximate_transport_top1_accuracy"] = float(
            (predicted_source == expected_source).float().mean().cpu()
        )
    return metrics


def train_probe(config: ProbeConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    model = LearnedInverseProbe(config).to(config.device)
    source, target = make_copy_data(config, seed_offset=1)
    eval_config = ProbeConfig(**{**asdict(config), "samples": config.eval_samples})
    eval_source, eval_target = make_copy_data(eval_config, seed_offset=2)
    source_resolution = make_resolution(config.grid_size)
    resolution = make_resolution(
        output_grid_size(config),
        longitude_shift=config.output_longitude_shift,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trajectory: list[dict[str, float]] = []
    best_mse = math.inf
    report_every = max(1, config.steps // 10)
    started = time.perf_counter()

    for step in range(config.steps + 1):
        if config.fresh_batches and step > 0:
            source, target = make_copy_data(config, seed_offset=step + 2)
        optimizer.zero_grad()
        output = model(source, source_resolution, resolution)
        loss = torch.nn.functional.mse_loss(output, target)
        best_mse = min(best_mse, float(loss.detach().cpu()))
        if step == config.steps:
            break
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % report_every == 0:
            trajectory.append(
                {
                    "step": float(step + 1),
                    "mse": float(loss.detach().cpu()),
                    "spatial_spread_ratio": spatial_spread_ratio(output, target),
                }
            )

    elapsed = time.perf_counter() - started
    optimizer.zero_grad()
    output = model(source, source_resolution, resolution)
    final_loss = torch.nn.functional.mse_loss(output, target)
    final_loss.backward()
    decoder = model.decoder
    if isinstance(decoder, PerceiverDecoder):
        query_gradient = decoder.query_embed.weight.grad
        query_gradient_norm = (
            0.0 if query_gradient is None else query_gradient.norm().item()
        )
    else:
        query_gradient_norm = 0.0
    per_query_difference = (output - output[..., :1, :1]).abs().max().item()
    decoder.eval()
    model.eval()
    with torch.no_grad():
        eval_representation = model.encoder(eval_source)
        eval_output = decoder(
            eval_representation,
            resolution,
            source_resolution=source_resolution,
        )
        eval_mse = torch.nn.functional.mse_loss(eval_output, eval_target)
    attention_metrics = (
        attention_diagnostics(decoder, eval_representation, resolution)
        if isinstance(decoder, PerceiverDecoder)
        else {}
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "config": asdict(config),
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "final_mse": float(final_loss.detach().cpu()),
        "best_training_mse": best_mse,
        "heldout_mse": float(eval_mse.cpu()),
        "final_rmse": math.sqrt(float(final_loss.detach().cpu())),
        "spatial_spread_ratio": spatial_spread_ratio(output, target),
        "training_field_diagnostics": field_diagnostics(output.detach(), target),
        "heldout_field_diagnostics": field_diagnostics(eval_output, eval_target),
        "max_output_difference_across_queries": per_query_difference,
        "query_embed_gradient_norm": query_gradient_norm,
        "attention": attention_metrics,
        "trajectory": trajectory,
    }


def main() -> None:
    report = train_probe(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
