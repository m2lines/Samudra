# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/microsoft/aurora/blob/main/aurora/model/encoder.py
# - https://github.com/lucidrains/vit-pytorch

import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from torch import nn

from samudra.constants import Input, Lat, Lon
from samudra.models.modules.augment_input import FourierFeatures2D


def patch_from(
    patch_extent: tuple[float, float], input_height: int, input_width: int
) -> tuple[int, int]:
    """Calculate the patch size in lat/lng pixels (or coords) from the patch spatial extent and input grid size."""
    lat_spacing = 180.0 / input_height  # Full sphere is 180 degrees (pole to pole)
    lon_spacing = 360.0 / input_width  # Full circle is 360 degrees

    # Calculate patch size to match target extent
    patch_h = int(round(patch_extent[0] / lat_spacing))
    patch_w = int(round(patch_extent[1] / lon_spacing))

    return patch_h, patch_w


class SpatialQueryPerceiver(nn.Module):
    """Encode a patch as ordered, coordinate-conditioned Perceiver outputs."""

    def __init__(
        self,
        *,
        query_shape: tuple[int, int],
        queries_dim: int,
        channels_per_query: int,
        perceiver_io: nn.Module,
        num_freq_bands: int,
        max_freq: float,
    ) -> None:
        super().__init__()
        query_h, query_w = query_shape
        if query_h < 1 or query_w < 1:
            raise ValueError("query_shape entries must be positive.")
        self.query_shape = query_shape
        self.channels_per_query = channels_per_query
        self.input_position_features = FourierFeatures2D(
            num_freq_bands=num_freq_bands,
            max_freq=max_freq,
        )
        self.query_embed = nn.Linear(2, queries_dim)
        self.query_offset = nn.Parameter(torch.zeros(query_h * query_w, queries_dim))
        self.perceiver_io = perceiver_io

        query_lat = torch.linspace(-1.0, 1.0, query_h)
        query_lon = torch.linspace(-1.0, 1.0, query_w)
        positions = torch.stack(
            torch.meshgrid(query_lat, query_lon, indexing="ij"), dim=-1
        ).flatten(0, 1)
        self.register_buffer("query_positions", positions, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = self.input_position_features(x)
        data = rearrange(data, "b ph pw v -> b (ph pw) v")
        queries = self.query_embed(
            self.query_positions.to(device=x.device, dtype=x.dtype)
        )
        queries = queries + self.query_offset.to(dtype=queries.dtype)
        encoded = self.perceiver_io(data, queries=queries)
        expected = (queries.shape[0], self.channels_per_query)
        if encoded.shape[1:] != expected:
            raise ValueError(
                "Spatial-query Perceiver returned an unexpected shape: "
                f"{tuple(encoded.shape)}."
            )
        return encoded


class SpatialLatentGridEncoder(nn.Module):
    """Keep coordinate-conditioned patch queries as an explicit spatial grid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        spatial_perceiver: SpatialQueryPerceiver,
    ) -> None:
        super().__init__()
        if spatial_perceiver.channels_per_query != out_channels:
            raise ValueError(
                "Spatial-grid query channels must equal encoder out_channels."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_extent = patch_extent
        self.spatial_perceiver = spatial_perceiver
        query_h, query_w = spatial_perceiver.query_shape
        self.output_patch_extent = (
            patch_extent[0] / query_h,
            patch_extent[1] / query_w,
        )
        self.pos_embed = nn.Linear(out_channels, out_channels)
        self.scale_embed = nn.Linear(out_channels, out_channels)

    def forward(self, x: Input, resolution: tuple[Lat, Lon]) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        lat, lon = resolution
        patch_h, patch_w = patch_from(self.patch_extent, height, width)
        query_h, query_w = self.spatial_perceiver.query_shape
        if patch_h % query_h or patch_w % query_w:
            raise ValueError(
                "Patch pixels must divide evenly over the spatial query grid; got "
                f"patch={(patch_h, patch_w)} and queries={(query_h, query_w)}."
            )
        if height % patch_h or width % patch_w:
            raise ValueError("Input grid must divide evenly into Perceiver groups.")
        coarse_h, coarse_w = height // patch_h, width // patch_w
        patches = rearrange(
            x,
            "b c (h ph) (w pw) -> (b h w) ph pw c",
            ph=patch_h,
            pw=patch_w,
        )
        encoded = self.spatial_perceiver(patches)
        encoded = rearrange(
            encoded,
            "(b h w) (qh qw) c -> b (h qh w qw) c",
            b=batch,
            h=coarse_h,
            w=coarse_w,
            qh=query_h,
            qw=query_w,
        )
        pos_encode, scale_encode = pos_scale_enc(
            self.out_channels,
            lat,
            lon,
            (patch_h // query_h, patch_w // query_w),
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        encoded = encoded + self.pos_embed(
            pos_encode.to(dtype=encoded.dtype, device=encoded.device)
        ).unsqueeze(0)
        encoded = encoded + self.scale_embed(
            scale_encode.to(dtype=encoded.dtype, device=encoded.device)
        ).unsqueeze(0)
        return rearrange(
            encoded,
            "b (h w) c -> b c h w",
            h=coarse_h * query_h,
            w=coarse_w * query_w,
        )


class PerceiverEncoder(nn.Module):
    """A perceiver-based encoder for Samudra's flattened data (a whole column of the ocean, with history).

    We adopt some of Aurora's positional encodings[1], which uses log-spaced fourier features with geometry-informed
    wavelengths. These encode 2d positions (the average latitude and longitude of each patch) as well as grid cell area
    (measured in km^2) for each token before it enters the processor.

    > Note: We assume that data along the lat/lon coordinates are positioned at the center of each grid point! Please
    > ensure this is the case at the data processing time.

    This encoder is designed to make the same number of patches with the same spatial extents across different scales
    of input data (input data may vary in resolution of lat/lng grid). To accomplish this with a single perceiver model,
    our `forward` call requires supplementary information: the resolution (a pair of Lat/Lon tensors), which is used to
    make consistent positional encodings for patches across different scales. While higher resolution scales will
    contain more data per patch, the patch will refer to the same physical area on Earth as all other scales.

    Args:
        in_channels (int): the number of input channels (roughly:  time x variable x (surface + depths)).
        out_channels (int): size of the latent dimension (aka, the embedding dimension).
        patch_extent (tuple[float, float]): spatial extent of each patch measured in degrees of lat/lon.
        perceiver (nn.Module): the perceiver module implementation to use.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_extent: tuple[float, float],
        perceiver: nn.Module,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels: int = out_channels  # aka, `embed_dim`.
        self.patch_extent = patch_extent
        self.perceiver = perceiver
        # TODO(#451): The input to these position and scale linear units could be a hparam.
        self.pos_embed = nn.Linear(self.out_channels, self.out_channels)
        self.scale_embed = nn.Linear(self.out_channels, self.out_channels)

    def forward(
        self, x: Input, resolution: tuple[Lat, Lon]
    ) -> Float[torch.Tensor, "batch {self.embed_dim} h w"]:
        _, V, H, W = x.shape
        lat, lon = resolution
        patch_h, patch_w = patch_from(self.patch_extent, H, W)
        # V is a cross product of variable, level (encoded in vars), and time (has history).
        assert V == self.in_channels
        # Ensure patch_size is appropriate for the data.
        assert H % patch_h == 0, f"{H} % {patch_h} != 0."
        assert W % patch_w == 0, f"{W} % {patch_w} != 0."

        # Perceiver experiment ideas:
        # 1. leave it as it is: treating each pixel as a token -- i.e. all channels (includes depths) per pixel
        # 2. change to original plan, where each float is its own token
        # 3. Add a third dim -- ph pw d v -- so each spatial position is a token
        x = rearrange(
            x,
            "b v (h ph) (w pw) -> (b h w) ph pw v",
            ph=patch_h,
            pw=patch_w,
        )
        # NB(alxmrs): This is includes a mean and LayerNorm before linear projection!
        x = self.perceiver(x)  # (B_H_W, ..., V) -> (B_H_W, out_channels)

        # Make `x` amenable to adding position + scale encoding
        x = rearrange(
            x,
            "(b h w) l -> b (h w) l ",
            h=(H // patch_h),
            w=(W // patch_w),
        )

        # Calculate and add positional + scale encoding
        pos_encode, scale_encode = pos_scale_enc(
            self.out_channels,  # aka "embed_dim"
            lat,
            lon,
            (patch_h, patch_w),
            # TODO(#452): Pos and scale wavelengths range all the way to the whole Earth by default; we could probably
            #  better tune these for our Oceans modeling use case.
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encoding = self.pos_embed(
            pos_encode.to(dtype=x.dtype, device=x.device)
        ).unsqueeze(0)
        scale_encoding = self.scale_embed(
            scale_encode.to(dtype=x.dtype, device=x.device)
        ).unsqueeze(0)
        x = x + pos_encoding + scale_encoding

        # Unpack spatial channels, move channel dimension to correct location.
        x = rearrange(
            x,
            "b (h w) l -> b l h w",
            h=(H // patch_h),
            w=(W // patch_w),
        )

        return x
