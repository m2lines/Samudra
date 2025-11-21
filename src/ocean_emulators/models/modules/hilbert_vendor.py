# Code vendored from https://github.com/Yunge6666/Hilbert-Local-Attention
# ruff: noqa
# Minor modification by Ocean Emulators contributors.
"""
Neighborhood Attention Transformer.
To appear in CVPR 2023.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import natten
from natten import NeighborhoodAttention1D as NeighborhoodAttention

is_natten_post_017 = hasattr(natten, "context")


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int, result):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = sgn(ax), sgn(ay)
    dbx, dby = sgn(bx), sgn(by)

    if h == 1 or w == 1:
        if h == 1:
            for _ in range(w):
                result.append((x, y))
                x, y = x + dax, y + day
        elif w == 1:
            for _ in range(h):
                result.append((x, y))
                x, y = x + dbx, y + dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        generate2d(x, y, ax2, ay2, bx, by, result)
        generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, result)
    else:
        if h2 % 2 and h > 2:
            bx2, by2 = bx2 + dbx, by2 + dby
        generate2d(x, y, bx2, by2, ax2, ay2, result)
        generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, result)
        generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
            result,
        )


def gilbert2d(width, height) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    if width >= height:
        generate2d(0, 0, width, 0, 0, height, result)
    else:
        generate2d(0, 0, 0, height, width, 0, result)
    return result


class GilbertPathCache:
    def __init__(self):
        self.cache = {}
        self.device_index_cache = {}

    def get_or_create_path(self, H, W):
        key = (H, W)
        if key not in self.cache:
            path = gilbert2d(W, H)

            forward_map = torch.zeros((H, W), dtype=torch.long)
            reverse_map = torch.zeros((H * W, 2), dtype=torch.long)

            for idx, (x, y) in enumerate(path[: H * W]):
                if y < H and x < W:
                    forward_map[y, x] = idx
                    reverse_map[idx, 0] = y
                    reverse_map[idx, 1] = x

            self.cache[key] = {
                "path": path,
                "forward_map": forward_map,
                "reverse_map": reverse_map,
                "y_indices": reverse_map[:, 0].clone(),
                "x_indices": reverse_map[:, 1].clone(),
                "H": H,
                "W": W,
            }

        return self.cache[key]

    def get_indices_on_device(self, H, W, device):
        device_key = (H, W, str(device))
        if device_key in self.device_index_cache:
            return self.device_index_cache[device_key]
        info = self.get_or_create_path(H, W)
        y_dev = info["y_indices"].to(device)
        x_dev = info["x_indices"].to(device)
        self.device_index_cache[device_key] = (y_dev, x_dev)
        return y_dev, x_dev

    def precompute_paths(self, resolutions):
        for H, W in resolutions:
            self.get_or_create_path(H, W)

    def clear_cache(self):
        self.cache.clear()


_global_gilbert_cache = GilbertPathCache()


def tensor_to_gilbert_path(x, cache=None):
    """
    Args:
        x: Input tensor, shape (B, H, W, C)
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        Reordered tensor, shape (B, H*W, C)
    """
    B, H, W, C = x.shape
    device = x.device
    if cache is None:
        cache = _global_gilbert_cache

    y_indices, x_indices = cache.get_indices_on_device(H, W, device)
    gilbert_tensor = x[:, y_indices, x_indices, :]  # (B, H*W, C)

    return gilbert_tensor


def gilbert_tensor_to_2d(x, H, W, cache=None):
    """
    Args:
        x: Gilbert sequence tensor, shape (B, H*W, C)
        H: Target height
        W: Target width
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        2D layout tensor, shape (B, H, W, C)
    """
    B, N, C = x.shape
    device = x.device

    if cache is None:
        cache = _global_gilbert_cache

    output_2d = torch.zeros((B, H, W, C), dtype=x.dtype, device=device)

    valid_n = min(N, H * W)
    if valid_n > 0:
        y_all, x_all = cache.get_indices_on_device(H, W, device)
        y_indices = y_all[:valid_n]
        x_indices = x_all[:valid_n]

        output_2d[:, y_indices, x_indices, :] = x[:, :valid_n, :]

    return output_2d


model_urls = {
    "nat_mini_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_base.pth",
}


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        sequence_order: str = "row_major",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.sequence_order = sequence_order
        # use 1D neighborhood attention (input expected to be BLC). Here we do not pass extra parameters to be compatible with natten==0.14.4.
        extra_args: dict = {}
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            # attn_drop=attn_drop,  NB(alxmrs): Commented this out, not in current version 0.21.1
            proj_drop=drop,
            **extra_args,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            B, H, W, C = x.shape
            if self.sequence_order == "gilbert":
                x_seq = tensor_to_gilbert_path(x)
            else:
                x_seq = x.view(B, H * W, C).contiguous()
            x_seq = self.attn(x_seq)
            if self.sequence_order == "gilbert":
                x = gilbert_tensor_to_2d(x_seq, H, W)
            else:
                x = x_seq.view(B, H, W, C)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        B, H, W, C = x.shape
        if self.sequence_order == "gilbert":
            x_seq = tensor_to_gilbert_path(x)
        else:
            x_seq = x.view(B, H * W, C).contiguous()
        x_seq = self.attn(x_seq)
        if self.sequence_order == "gilbert":
            x = gilbert_tensor_to_2d(x_seq, H, W)
        else:
            x = x_seq.view(B, H, W, C)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        sequence_order: str = "row_major",
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    sequence_order=sequence_order,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)


class NAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        sequence_order: str = "row_major",
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                sequence_order=sequence_order,
            )
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def nat1D_mini(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=49,
        sequence_order="row_major",
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_mini_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat1D_tiny(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        embed_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat1D_small(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[3, 6, 12, 24],
        embed_dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_small_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat1D_base(pretrained=False, **kwargs):
    model = NAT(
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        embed_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        kernel_size=7,
        **kwargs,
    )
    if pretrained:
        url = model_urls["nat_base_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
