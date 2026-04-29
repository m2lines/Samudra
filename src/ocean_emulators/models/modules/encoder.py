# Sources inspired by the following implementations:
# - https://github.com/microsoft/aurora/blob/main/aurora/model/patchembed.py
# - https://github.com/microsoft/aurora/blob/main/aurora/model/encoder.py
# - https://github.com/lucidrains/vit-pytorch
# - https://github.com/lucidrains/perceiver-pytorch (intra-patch Fourier encoding)


import torch
from aurora.model.fourier import pos_expansion, scale_expansion
from aurora.model.posencoding import pos_scale_enc
from einops import rearrange
from jaxtyping import Float
from perceiver_pytorch.perceiver_pytorch import fourier_encode
from torch import nn

from ocean_emulators.constants import Boundary, Prognostic
from ocean_emulators.utils.ctx import GridContext

# Stream-type indices for the learned stream embedding.
_PROG_STREAM = 0
_BOUNDARY_STREAM = 1


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


def _intra_patch_fourier(
    ph: int,
    pw: int,
    num_bands: int,
    max_freq: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "n d"]:
    """2-D Fourier positional encoding for one ``ph x pw`` patch.

    Mirrors what ``perceiver_pytorch`` does internally when ``input_axis=2``
    and ``fourier_encode_data=True``: each pixel's normalized position in
    ``[-1, 1]^2`` is encoded with ``num_bands`` log-spaced frequencies.

    Returns ``(ph * pw, 2 * (2 * num_bands + 1))``: row-major flatten over
    the patch, then concatenated ``[sin, cos, raw]`` features per axis.
    """
    y = torch.linspace(-1.0, 1.0, ph, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, pw, device=device, dtype=dtype)
    pos = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    enc = fourier_encode(pos, max_freq, num_bands)
    return rearrange(enc, "ph pw a d -> (ph pw) (a d)")


class PerceiverEncoder(nn.Module):
    """Single-perceiver encoder over a concatenated prog + boundary sequence.

    Each input stream is patchified in 2-D using the same ``patch_extent``
    in degrees, so both produce the same patch grid regardless of native
    resolution.  Within each patch:

      * Prog and boundary pixels are flattened to 1-D token sequences and
        linearly projected to a common ``token_dim``.
      * A 2-D Fourier positional encoding (in normalized patch coordinates
        ``[-1, 1]^2``) is added to every token; the same projection is
        applied to both streams, so geometrically equivalent positions
        share the same feature.
      * A learned stream-type embedding is added so the Perceiver can tell
        prog tokens apart from boundary tokens.

    The two token sequences are concatenated and fed to a single Perceiver,
    which pools each patch's sequence to a latent vector.  We project that
    to ``out_channels`` and add Aurora-style patch-level positional and
    scale encodings [1] (computed on the prognostic grid).

    Args:
        prog_channels: Number of prognostic input channels.
        boundary_channels: Number of boundary input channels.
        out_channels: Final embedding dimension produced by this encoder.
        token_dim: Per-token dimension fed into the Perceiver.
        latent_dim: Pooled output dimension of the Perceiver.
        patch_extent: Spatial extent of each patch in degrees ``(lat, lon)``.
        max_patch_size: Largest ``(ph, pw)`` across input sources, used to
            set the intra-patch Fourier max frequency (Nyquist-style).
        perceiver: Perceiver mapping ``(B, N, token_dim) -> (B, latent_dim)``.
        num_freq_bands: Fourier frequency bands for intra-patch encoding.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2405.13063#A2.SS4
    """

    # TODO(alxmrs): Implement gradient checkpointing
    def __init__(
        self,
        prog_channels: int,
        boundary_channels: int,
        out_channels: int,
        token_dim: int,
        latent_dim: int,
        patch_extent: tuple[float, float],
        max_patch_size: tuple[int, int],
        perceiver: nn.Module,
        num_freq_bands: int = 4,
    ) -> None:
        super().__init__()
        self.prog_channels = prog_channels
        self.boundary_channels = boundary_channels
        self.out_channels: int = out_channels
        self.token_dim = token_dim
        self.patch_extent = patch_extent
        self.num_freq_bands = num_freq_bands
        # Nyquist-style cap on the intra-patch Fourier features, mirroring
        # the previous behavior when the Perceiver did its own Fourier.
        self.max_intra_patch_freq = float(max(*max_patch_size))

        self.prog_proj = nn.Linear(prog_channels, token_dim)
        self.boundary_proj = nn.Linear(boundary_channels, token_dim)

        fourier_channels = 2 * (2 * num_freq_bands + 1)
        self.pos_proj = nn.Linear(fourier_channels, token_dim)

        # 0 -> prog, 1 -> boundary.
        self.stream_embed = nn.Embedding(2, token_dim)

        self.perceiver = perceiver

        self.fusion_proj = nn.Linear(latent_dim, out_channels)

        # TODO(#451): The input to these position and scale linear units could be a hparam.
        self.pos_embed = nn.Linear(self.out_channels, self.out_channels)
        self.scale_embed = nn.Linear(self.out_channels, self.out_channels)

    def _patchify_params(
        self, shape: torch.Size, expected_channels: int
    ) -> tuple[int, int, int, int]:
        """Validate channels and compute patch / latent-grid dims.

        Returns ``(ph, pw, lat_h, lat_w)``.
        """
        _, v, h, w = shape
        assert v == expected_channels, (
            f"Expected {expected_channels} channels, got {v}."
        )
        ph, pw = patch_from(self.patch_extent, h, w)
        assert h % ph == 0, f"{h} % {ph} != 0."
        assert w % pw == 0, f"{w} % {pw} != 0."
        return ph, pw, h // ph, w // pw

    def _tokenize(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        ph: int,
        pw: int,
        stream_id: int,
    ) -> Float[torch.Tensor, "bp n d"]:
        """Flatten patches to 1-D, project, add intra-patch Fourier + stream embed.

        Returns ``(B * lat_h * lat_w, ph * pw, token_dim)``.
        """
        tokens = rearrange(
            x,
            "b v (h ph) (w pw) -> (b h w) (ph pw) v",
            ph=ph,
            pw=pw,
        )
        # Per-pixel channel mix into a common token_dim — equivalent to a 1x1
        # conv on each patch, and analogous to how the lucidrains Perceiver
        # lifts raw channels before adding its positional features.  Lets us
        # concat prog and boundary (different V) and add the additive Fourier
        # / stream signals at a meaningful scale.
        tokens = proj(tokens)

        pos_enc = _intra_patch_fourier(
            ph,
            pw,
            self.num_freq_bands,
            self.max_intra_patch_freq,
            tokens.device,
            tokens.dtype,
        )
        # TODO(alxmrs): Should we make a pos_proj for each stream, or use a shared projection?
        tokens = tokens + self.pos_proj(pos_enc).unsqueeze(0)

        stream_idx = torch.tensor(stream_id, device=tokens.device)
        tokens = tokens + self.stream_embed(stream_idx)

        return tokens

    def forward(
        self,
        prog: Prognostic,
        boundary: Boundary,
        ctx: GridContext,
    ) -> Float[torch.Tensor, "batch {self.out_channels} h w"]:
        prog_ph, prog_pw, lat_h, lat_w = self._patchify_params(
            prog.shape, self.prog_channels
        )
        b_ph, b_pw, b_lat_h, b_lat_w = self._patchify_params(
            boundary.shape, self.boundary_channels
        )
        assert lat_h == b_lat_h and lat_w == b_lat_w, (
            f"Latent grid mismatch: prog ({lat_h}, {lat_w}) vs "
            f"boundary ({b_lat_h}, {b_lat_w}). Check that patch_extent "
            f"divides both grids evenly."
        )

        prog_tokens = self._tokenize(
            prog, self.prog_proj, prog_ph, prog_pw, _PROG_STREAM
        )
        boundary_tokens = self._tokenize(
            boundary, self.boundary_proj, b_ph, b_pw, _BOUNDARY_STREAM
        )

        # Concat prog + boundary tokens along the sequence dim.  The Perceiver
        # cross-attends over the unified sequence, so prog and boundary mix
        # inside the latent set rather than being fused after the fact.
        seq = torch.cat([prog_tokens, boundary_tokens], dim=1)

        pooled = self.perceiver(seq)  # (B*lat_h*lat_w, latent_dim)
        x = self.fusion_proj(pooled)  # (B*lat_h*lat_w, out_channels)

        # --- Patch-level positional + scale encoding (Aurora) ---
        x = rearrange(x, "(b h w) l -> b (h w) l", h=lat_h, w=lat_w)
        lat, lon = ctx.input_resolution_cpu
        pos_encode, scale_encode = pos_scale_enc(
            self.out_channels,
            lat,
            lon,
            (prog_ph, prog_pw),
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

        x = rearrange(x, "b (h w) l -> b l h w", h=lat_h, w=lat_w)

        return x
