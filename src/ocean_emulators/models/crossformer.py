# This code is adapted from the WXFormer model, which is available at
# https://github.com/NCAR/miles-credit

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn

from ocean_emulators.config import CrossformerConfig
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.corrector import Corrector


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.output_channels = out_chans

        blk: list[nn.Module] = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class DynamicPositionBias(nn.Module):
    def __init__(self, dim):
        super(DynamicPositionBias, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            Rearrange("... () -> ..."),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0):
        super().__init__()
        assert attn_type in {
            "short",
            "long",
        }, "attention type must be one of local or distant"
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        # positions

        self.dpb = DynamicPositionBias(dim // 4)

        # calculate and store indices for retrieving bias

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        height, width = x.shape[-2:]
        heads = self.heads
        wsz = self.window_size
        device = x.device

        # prenorm
        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == "short":
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz, s2=wsz)
        elif self.attn_type == "long":
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz, l2=wsz)
            x = x.contiguous()

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v)
        )
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add dynamic positional bias

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rearrange(rel_pos, "c i j -> (i j) c")
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz, y=wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == "short":
            out = rearrange(
                out,
                "(b h w) d s1 s2 -> b d (h s1) (w s2)",
                h=height // wsz,
                w=width // wsz,
            )
        elif self.attn_type == "long":
            out = rearrange(
                out,
                "(b h w) d l1 l2 -> b d (l1 h) (l2 w)",
                h=height // wsz,
                w=width // wsz,
            )
        out = out.contiguous()

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth=4,
        dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            attn_type="short",
                            window_size=local_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                        Attention(
                            dim,
                            attn_type="long",
                            window_size=global_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


class TensorPadding:
    def __init__(self, mode="earth", pad_lat=(40, 40), pad_lon=(40, 40)):
        """
        Initialize the TensorPadding class with the specified mode and padding sizes.

        Args:
            mode (str): The padding mode, either 'mirror' or 'earth'.
            pad_lat (list[int]): Padding sizes for the North-South (latitude) dimension
                                [top, bottom].
            pad_lon (list[int]): Padding sizes for the West-East (longitude) dimension
                                [left, right].
        """
        self.mode = mode
        self.pad_NS = pad_lat
        self.pad_WE = pad_lon

    def pad(self, x):
        """
        Apply padding to the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The padded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_padding(x)
        elif self.mode == "earth":
            return self._earth_padding(x)

    def unpad(self, x):
        """
        Remove padding from the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Padded tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_unpad(x)
        elif self.mode == "earth":
            return self._earth_unpad(x)

    def _earth_padding(self, x):
        """
        Apply earth padding to the tensor (poles and circular padding).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        if any(p > 0 for p in self.pad_NS):
            # 180-degree shift using half the longitude size
            shift_size = int(x.shape[-1] // 2)
            xroll = torch.roll(x, shifts=shift_size, dims=-1)
            # pad poles
            xroll_flip_top = torch.flip(xroll[..., : self.pad_NS[0], :], (-2,))
            xroll_flip_bot = torch.flip(xroll[..., -self.pad_NS[1] :, :], (-2,))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular")

        return x

    def _earth_unpad(self, x):
        """
        Remove earth padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            start_NS = self.pad_NS[0]
            end_NS = -self.pad_NS[1] if self.pad_NS[1] > 0 else None
            x = x[..., start_NS:end_NS, :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            start_WE = self.pad_WE[0]
            end_WE = -self.pad_WE[1] if self.pad_WE[1] > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x

    def _mirror_padding(self, x):
        """
        Apply mirror padding to the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        # pad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            pad_lon_left, pad_lon_right = self.pad_WE
            x = F.pad(
                x, pad=(self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular"
            )

        # pad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = F.pad(
                x, pad=(0, 0, self.pad_NS[0], self.pad_NS[1], 0, 0), mode="reflect"
            )

        return x

    def _mirror_unpad(self, x):
        """
        Remove mirror padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = x[..., self.pad_NS[0] : -self.pad_NS[1], :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            x = x[..., :, self.pad_WE[0] : -self.pad_WE[1]]

        return x


# classes
class CrossFormer(BaseModel):
    def __init__(
        self,
        config: CrossformerConfig,
        wet: Optional[torch.Tensor] = None,
        hist: int = 0,
    ):
        """
        CrossFormer is the base architecture for the WXFormer model. It uses
        convolutions and long and short distance attention layers in the
        encoder layer and then uses strided transpose convolution blocks for
        the decoder layer.
        """
        super().__init__(
            ch_width=[config.n_in] + list(config.dims),
            n_out=config.n_out,
            wet=wet,
            hist=hist,
            pred_residuals=config.pred_residuals,
            last_kernel_size=-1,
            pad="",
        )

        dim = tuple(config.dims)
        depth = tuple(config.depth)
        global_window_size = tuple(config.global_window_size)
        cross_embed_kernel_sizes = tuple(
            [tuple(_) for _ in config.cross_embed_kernel_sizes]
        )
        cross_embed_strides = tuple(config.cross_embed_strides)

        self.image_height = config.image_height
        self.image_width = config.image_width
        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.use_spectral_norm = config.use_spectral_norm
        self.use_interp = config.interp
        if config.padding_conf is not None:
            self.use_padding = True
            self.padding_opt = TensorPadding(
                pad_lat=config.padding_conf["pad_lat"],
                pad_lon=config.padding_conf["pad_lon"],
            )

        # input channels
        self.input_channels = config.n_in

        # output channels
        self.output_channels = config.n_out

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size_tuple = cast_tuple(config.local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size_tuple) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # dimensions
        last_dim = dim[-1]
        first_dim = (
            self.input_channels
            if (config.patch_height == 1 and config.patch_width == 1)
            else dim[0]
        )
        dims = [first_dim, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # allocate cross embed layers
        self.layers = nn.ModuleList([])

        # loop through hyperparameters
        for (
            dim_in,
            dim_out,
        ), num_layers, global_wsize, local_wsize, kernel_sizes, stride in zip(
            dim_in_and_out,
            depth,
            global_window_size,
            local_window_size_tuple,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            # create CrossEmbedLayer
            cross_embed_layer = CrossEmbedLayer(
                dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernel_sizes, stride=stride
            )

            # create Transformer
            transformer_layer = Transformer(
                dim=dim_out,
                local_window_size=local_wsize,
                global_window_size=global_wsize,
                depth=num_layers,
                dim_head=config.dim_head,
                attn_dropout=config.attn_dropout,
                ff_dropout=config.ff_dropout,
            )

            # append everything
            self.layers.append(nn.ModuleList([cross_embed_layer, transformer_layer]))

        self.up_block1 = UpBlock(1 * last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlock(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlock(2 * (last_dim // 4), last_dim // 8, dim[0])
        self.up_block4 = nn.ConvTranspose2d(
            2 * (last_dim // 8),
            self.output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        if self.use_spectral_norm:
            logging.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        self.corrector = Corrector(config.corrector, hist)

    def forward_once(self, x):
        # x_copy = x.clone().detach()
        # print("Input shape:", x.shape)
        if self.use_padding:
            x = self.padding_opt.pad(x)

        # print("After padding:", x.shape)
        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            # print("After cross embed:", x.shape)
            x = transformer(x)
            # print("After transformer:", x.shape)
            encodings.append(x)

        x = self.up_block1(x)
        # print("After up block 1:", x.shape)
        x = torch.cat([x, encodings[2]], dim=1)
        # print("After cat 1:", x.shape)
        x = self.up_block2(x)
        # print("After up block 2:", x.shape)
        x = torch.cat([x, encodings[1]], dim=1)
        # print("After cat 2:", x.shape)
        x = self.up_block3(x)
        # print("After up block 3:", x.shape)
        x = torch.cat([x, encodings[0]], dim=1)
        # print("After cat 3:", x.shape)
        x = self.up_block4(x)
        # print("After up block 4:", x.shape)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        # print("After unpadding:", x.shape)

        if self.use_interp:
            x = F.interpolate(
                x, size=(self.image_height, self.image_width), mode="bilinear"
            )

        # print("After interpolation:", x.shape)

        x = self.corrector(x)

        # print("After corrector:", x.shape)

        return x
