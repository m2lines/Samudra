# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare Perceiver decoders on held-out real OM4 patches.

The probe deliberately isolates the representation heads from SamudraMulti's
processor. It uses the same original-Perceiver encoder pattern as the production
model and either the current full PerceiverIO decoder or the direct Perceiver IO
decode stage. Data are split by time so memorizing fixed patch locations cannot
solve held-out ocean states.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import nn

from samudra.models.modules import DirectCrossAttentionIO, Perceiver, PerceiverIO
from samudra.models.modules.augment_input import make_3d_coordinate_grid

DEFAULT_SOURCE = (
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-07/om4_twodeg/OM4.zarr"
)

Architecture = Literal["perceiver_io", "direct_cross_attention"]
Task = Literal["reconstruction", "forecast"]


@dataclass(frozen=True)
class ProbeConfig:
    architecture: Architecture
    task: Task
    source: str
    cache: Path
    train_times: int
    eval_times: int
    patch_height: int
    patch_width: int
    samples_per_time: int
    steps: int
    batch_size: int
    learning_rate: float
    encoder_latents: int
    decoder_latents: int
    latent_dim: int
    representation_dim: int
    queries_dim: int
    cross_heads: int
    cross_dim_head: int
    seed: int
    device: str


def parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("perceiver_io", "direct_cross_attention"),
        default="direct_cross_attention",
    )
    parser.add_argument(
        "--task", choices=("reconstruction", "forecast"), default="reconstruction"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--cache", type=Path, default=Path(".data_cache/om4_twodeg_probe.pt")
    )
    parser.add_argument("--train-times", type=int, default=16)
    parser.add_argument("--eval-times", type=int, default=8)
    parser.add_argument("--patch-height", type=int, default=3)
    parser.add_argument("--patch-width", type=int, default=5)
    parser.add_argument("--samples-per-time", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-latents", type=int, default=16)
    parser.add_argument("--decoder-latents", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=256)
    parser.add_argument("--queries-dim", type=int, default=128)
    parser.add_argument("--cross-heads", type=int, default=2)
    parser.add_argument("--cross-dim-head", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()
    config = ProbeConfig(**vars(args))
    if (
        min(
            config.train_times,
            config.eval_times,
            config.patch_height,
            config.patch_width,
            config.samples_per_time,
            config.steps,
            config.batch_size,
        )
        < 1
    ):
        parser.error("time, patch, sample, step, and batch counts must be positive")
    if 90 % config.patch_height != 0 or 180 % config.patch_width != 0:
        parser.error("patch dimensions must evenly divide the 2-degree 90x180 grid")
    return config


def prognostic_names() -> list[str]:
    names: list[str] = []
    for prefix in ("thetao", "so", "uo", "vo"):
        names.extend(f"{prefix}_{depth}" for depth in range(19))
    names.append("zos")
    return names


def load_or_cache(config: ProbeConfig) -> dict[str, torch.Tensor]:
    required_times = (
        config.train_times + config.eval_times + int(config.task == "forecast")
    )
    if config.cache.exists():
        cached = torch.load(config.cache, map_location="cpu", weights_only=True)
        if cached["state"].shape[0] >= required_times:
            return cached

    names = prognostic_names()
    dataset = xr.open_zarr(config.source)
    state_array = (
        dataset[names]
        .isel(time=slice(0, required_times))
        .to_array("channel")
        .transpose("time", "channel", "y", "x")
        .load()
        .values
    )
    masks = []
    for name in names:
        depth = int(name.rsplit("_", 1)[1]) if "_" in name else 0
        masks.append(dataset[f"mask_{depth}"].load().values)

    payload = {
        "state": torch.from_numpy(state_array).float(),
        "mask": torch.from_numpy(np.stack(masks)).bool(),
        "latitude": torch.from_numpy(dataset.y.values).float(),
        "longitude": torch.from_numpy(dataset.x.values).float(),
    }
    config.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, config.cache)
    return payload


def normalize_state(
    state: torch.Tensor, mask: torch.Tensor, train_times: int
) -> torch.Tensor:
    wet = mask.unsqueeze(0).expand(train_times, -1, -1, -1)
    training = state[:train_times]
    count = wet.sum(dim=(0, 2, 3)).clamp_min(1)
    mean = torch.where(wet, training, 0).sum(dim=(0, 2, 3)) / count
    centered = torch.where(wet, training - mean[None, :, None, None], 0)
    variance = centered.square().sum(dim=(0, 2, 3)) / count
    std = variance.sqrt().clamp_min(1e-6)
    normalized = (state - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.where(mask[None], normalized, 0)


def patchify(tensor: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    return rearrange(
        tensor,
        "t c (h ph) (w pw) -> t (h w) c ph pw",
        ph=patch_height,
        pw=patch_width,
    )


class PatchPerceiverAutoencoder(nn.Module):
    def __init__(self, config: ProbeConfig, channels: int) -> None:
        super().__init__()
        self.encoder = Perceiver(
            num_freq_bands=4,
            max_freq=max(config.patch_height, config.patch_width),
            depth=2,
            input_axis=2,
            input_channels=channels,
            num_latents=config.encoder_latents,
            latent_dim=config.latent_dim,
            cross_heads=config.cross_heads,
            cross_dim_head=config.cross_dim_head,
            num_classes=config.representation_dim,
            weight_tie_layers=True,
            self_per_cross_attn=2,
        )
        self.patch_position = nn.Linear(3, config.representation_dim)
        self.query_embed = nn.Linear(3, config.queries_dim)
        if config.architecture == "perceiver_io":
            self.decoder: nn.Module = PerceiverIO(
                depth=2,
                dim=config.representation_dim,
                queries_dim=config.queries_dim,
                logits_dim=channels,
                num_latents=config.decoder_latents,
                latent_dim=config.latent_dim,
                cross_heads=config.cross_heads,
                cross_dim_head=config.cross_dim_head,
                weight_tie_layers=True,
                decoder_ff=True,
            )
        else:
            self.decoder = DirectCrossAttentionIO(
                input_dim=config.representation_dim,
                queries_dim=config.queries_dim,
                output_dim=channels,
                heads=config.cross_heads,
                dim_head=config.cross_dim_head,
            )

    def forward(self, state: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        representation = self.encoder(rearrange(state, "b c h w -> b h w c"))
        center = coordinates.mean(dim=1)
        representation = representation + self.patch_position(center)
        queries = self.query_embed(coordinates)
        decoded = self.decoder(representation[:, None], queries=queries)
        return rearrange(
            decoded,
            "b (h w) c -> b c h w",
            h=state.shape[-2],
            w=state.shape[-1],
        )


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return (prediction.sub(target).square() * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    coordinates: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    wet_count = 0
    prediction_energy = 0.0
    target_energy = 0.0
    for start in range(0, len(source), batch_size):
        batch = slice(start, start + batch_size)
        prediction = model(source[batch], coordinates[batch])
        batch_mask = mask[batch]
        squared_error += float(
            (prediction.sub(target[batch]).square() * batch_mask).sum().cpu()
        )
        wet_count += int(batch_mask.sum().cpu())
        prediction_energy += float((prediction.square() * batch_mask).sum().cpu())
        target_energy += float((target[batch].square() * batch_mask).sum().cpu())
    return {
        "masked_mse": squared_error / wet_count,
        "rms_ratio": math.sqrt(prediction_energy / target_energy),
    }


def main(config: ProbeConfig) -> None:
    torch.manual_seed(config.seed)
    payload = load_or_cache(config)
    state = normalize_state(payload["state"], payload["mask"], config.train_times)
    state_patches = patchify(state, config.patch_height, config.patch_width)
    mask_patches = patchify(
        payload["mask"].unsqueeze(0), config.patch_height, config.patch_width
    ).squeeze(0)

    coordinate_grid = make_3d_coordinate_grid(payload["latitude"], payload["longitude"])
    coordinate_patches = patchify(
        coordinate_grid.unsqueeze(0), config.patch_height, config.patch_width
    ).squeeze(0)
    coordinate_patches = rearrange(coordinate_patches, "p c h w -> p (h w) c")

    generator = torch.Generator().manual_seed(config.seed)
    wet_patch = mask_patches.any(dim=(1, 2, 3))
    candidate_patches = wet_patch.nonzero(as_tuple=False).squeeze(1)
    if config.samples_per_time > len(candidate_patches):
        raise ValueError(
            f"samples_per_time={config.samples_per_time} exceeds the "
            f"{len(candidate_patches)} patches containing ocean cells."
        )
    selected = torch.stack(
        [
            candidate_patches[
                torch.randperm(len(candidate_patches), generator=generator)[
                    : config.samples_per_time
                ]
            ]
            for _ in range(config.train_times + config.eval_times)
        ]
    )

    source_times = torch.arange(config.train_times + config.eval_times)
    target_times = source_times + int(config.task == "forecast")
    source = state_patches[source_times[:, None], selected]
    target = state_patches[target_times[:, None], selected]
    masks = mask_patches[selected]
    coordinates = coordinate_patches[selected]

    source = rearrange(source, "t p c h w -> (t p) c h w")
    target = rearrange(target, "t p c h w -> (t p) c h w")
    masks = rearrange(masks, "t p c h w -> (t p) c h w").float()
    coordinates = rearrange(coordinates, "t p n c -> (t p) n c")

    split = config.train_times * config.samples_per_time
    device = torch.device(config.device)
    train_source, train_target, train_masks, train_coordinates = (
        tensor[:split].to(device) for tensor in (source, target, masks, coordinates)
    )
    eval_source, eval_target, eval_masks, eval_coordinates = (
        tensor[split:].to(device) for tensor in (source, target, masks, coordinates)
    )

    model = PatchPerceiverAutoencoder(config, channels=source.shape[1]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_metrics = evaluate(
        model,
        eval_source,
        eval_target,
        eval_masks,
        eval_coordinates,
        batch_size=config.batch_size,
    )
    for step in range(config.steps):
        model.train()
        indices = torch.randint(split, (config.batch_size,), generator=generator).to(
            device
        )
        source_batch, target_batch, mask_batch, coordinate_batch = (
            tensor[indices]
            for tensor in (
                train_source,
                train_target,
                train_masks,
                train_coordinates,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source_batch, coordinate_batch)
        loss = masked_mse(prediction, target_batch, mask_batch)
        loss.backward()
        optimizer.step()
        if step in {0, config.steps - 1} or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "train_masked_mse": loss.item()}))

    metrics = evaluate(
        model,
        eval_source,
        eval_target,
        eval_masks,
        eval_coordinates,
        batch_size=config.batch_size,
    )
    print(
        json.dumps(
            {
                "config": asdict(config),
                "parameters": parameter_count,
                "held_out_initial": initial_metrics,
                "held_out_final": metrics,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main(parse_args())
