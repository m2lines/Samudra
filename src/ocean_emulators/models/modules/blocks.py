from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
import torch.utils.checkpoint
from jaxtyping import Float

from ocean_emulators.models.modules.activations import CappedGELU


class PointwiseLinear(torch.nn.Module):
    """A 1×1 convolution implemented as nn.Linear.

    Mathematically equivalent to Conv2d(kernel_size=1), but avoids the
    non-contiguous gradient strides that 1×1 convs produce, which cause
    DDP to copy gradients instead of using zero-copy views.

    This optimization is use in the official ConvNext implementation[0].

    [0]: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L18
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(
        self, x: Float[torch.Tensor, "B C_in H W"]
    ) -> Float[torch.Tensor, "B C_out H W"]:
        return self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LayerNorm2d(torch.nn.Module):
    """Channel-wise LayerNorm for NCHW tensors.

    Applies ``nn.LayerNorm`` along the channel axis at each spatial position,
    matching the canonical ConvNeXt convention. The permute trick keeps the
    underlying ``nn.LayerNorm`` operating on its natural last-dim layout.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps)

    def forward(
        self, x: Float[torch.Tensor, "B C H W"]
    ) -> Float[torch.Tensor, "B C H W"]:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TransposedConvUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
    ):
        super().__init__()
        upsampler: list[torch.nn.Module] = []
        # Upsample transpose conv
        upsampler.append(
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
            )
        )

        if activation is not None:
            upsampler.append(activation())
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class ZonallyPeriodicBilinearUpsample(torch.nn.Module):
    """Bilinear upsampling that enforces periodicity along the x/longitude axis."""

    def __init__(self, upsampling: int | tuple[int, int] = 2):
        super().__init__()
        if isinstance(upsampling, int):
            upsampling = (upsampling, upsampling)
        if tuple(upsampling) != (2, 2):
            raise ValueError(
                "ZonallyPeriodicBilinearUpsample only supports 2x upsampling"
            )
        self.scale_h, self.scale_w = upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample with periodic padding along longitude to avoid seams and
        # keep interpolation aligned with PyTorch's bilinear sampling grid.
        width = x.shape[-1]
        padded = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
        upsampled = torch.nn.functional.interpolate(
            padded,
            scale_factor=(self.scale_h, self.scale_w),
            mode="bilinear",
            align_corners=False,
        )
        # Crop out the extra padded columns (scaled by the upsampling factor).
        start = self.scale_w
        end = start + width * self.scale_w
        return upsampled[..., start:end]


class DropPath(torch.nn.Module):
    """Drop path dropout (for skip connections).

    During training, randomly drops entire samples' skip connections
    with probability ``drop_prob``, scaling survivors by 1/(1-p) to preserve
    expected values. Implemented via ``nn.Dropout`` applied to a per-sample
    mask of ones.

    References:
        [0]: Rethinking U-net Skip Connections for Biomedical Image Segmentation (https://arxiv.org/abs/2402.08276)
        [1]: Dropout Reduces Underfitting (https://arxiv.org/abs/2303.01500)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=drop_prob)

    def forward(
        self, skip_conn: Float[torch.Tensor, "B C H W"]
    ) -> Float[torch.Tensor, "B C H W"]:
        if not self.training or self.dropout.p == 0.0:
            return skip_conn
        # Per-sample mask: (B, 1, 1, 1) broadcasts over C, H, W.
        mask = self.dropout(
            torch.ones(
                skip_conn.shape[0],
                1,
                1,
                1,
                device=skip_conn.device,
                dtype=skip_conn.dtype,
            )
        )
        return skip_conn * mask


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class MaxPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.maxpool(x)


class CoreBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        pad: str,
        upscale_factor: int = 1,
        norm: str = "batch",
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad
        self.upscale_factor = upscale_factor
        self.norm = norm

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class ConvBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        pad="circular",
        checkpoint_simple: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)

        layers: list[torch.nn.Module] = []
        layers.append(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(
                torch.nn.Conv2d(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(activation())

        self.layers = nn.ModuleList(layers)
        self.checkpoint_simple = checkpoint_simple

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
                # conv2d layers are expensive so we save their activations,
                # other (simple) layers are cheap, so we don't save their activations.
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)
            else:
                fts = layer(fts)
        return fts


def _pointwise(use_linear: bool, in_ch: int, out_ch: int) -> torch.nn.Module:
    """Create a pointwise (1×1) channel-mixing layer.

    When ``use_linear`` is True, returns a :class:`PointwiseLinear` backed by
    ``nn.Linear``, which produces 2-D weight tensors and avoids the
    non-contiguous gradient strides that ``Conv2d(kernel_size=1)`` introduces
    for degenerate spatial dimensions.  This matters for DDP, which otherwise
    falls back to copying gradients instead of using zero-copy views.

    The ``nn.Linear`` approach is also used in the official ConvNeXt
    implementation, where it is noted to be "slightly faster in PyTorch" [0].

    [0]: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    if use_linear:
        return PointwiseLinear(in_ch, out_ch)
    return torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, padding="same")


class ConvNeXtBlock(CoreBlock):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix
    paper. Use of dilations here.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
        checkpoint_simple: bool = False,
        pointwise_linear: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
        assert n_layers == 1, "Can only use a single layer here!"

        # Instantiate pointwise linear to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = _pointwise(pointwise_linear, in_channels, out_channels)

        # Convolution block
        convblock: list[torch.nn.Module] = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm2d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation())
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm2d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation())
        # Linear postprocessing
        convblock.append(
            _pointwise(
                pointwise_linear, int(in_channels * upscale_factor), out_channels
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)
        self.checkpoint_simple = checkpoint_simple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return skip + x


class TrueConvNeXtBlock(CoreBlock):
    """Canonical ConvNeXt block with a depthwise spatial convolution.

    Structure (per Liu et al. 2022, ConvNeXt)[0]:

        x → pre_proj (1×1, in→out, identity if in==out)   # channel transition
          → [skip branches here]
          → dwconv (k×k, depthwise, groups=out)           # spatial mixing
          → norm
          → pw_expand (1×1, out → out·U)                  # channel expand
          → activation
          → pw_project (1×1, out·U → out)                 # channel project
          + skip

    Differs from ``ConvNeXtBlock`` (which is misnamed — its spatial convs are
    dense, not depthwise) by factoring the spatial mixing (depthwise k×k) from
    the channel mixing (1×1's). This decoupling makes large-kernel sweeps
    affordable: at canonical 4× expansion, kernel-size cost scales as ``k²``
    times channels, not ``k²`` times channels-squared.

    This standard recipe has been adapted for our codebase, including:

      - Circular-x / zero-y padding around the depthwise conv (zonal periodicity).
      - ``PointwiseLinear`` for 1×1's (DDP-friendly gradient strides).
      - Configurable ``norm`` (batch | instance | layer).
      - Configurable ``activation`` (defaults to ``CappedGELU``).
      - ``checkpoint_simple`` recomputes cheap layers on backward.

    References:
        [0]: Liu et al. 2022, "A ConvNet for the 2020s" (https://arxiv.org/abs/2201.03545)
        [1]: Ding et al. 2022, "Scaling Up Your Kernels to 31×31" (https://arxiv.org/abs/2203.06717)

    NOTE: ``dilation`` is asserted to be 1. Large depthwise kernels make
    dilation redundant; combining the two would push ``N_pad`` beyond the
    feature-map size at deep U-Net stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        pad: str = "circular",
        upscale_factor: int = 4,
        norm: str = "batch",
        checkpoint_simple: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
        assert n_layers == 1, "TrueConvNeXtBlock only supports n_layers=1"
        assert dilation == 1, (
            "TrueConvNeXtBlock requires dilation=1; large depthwise kernels "
            "make dilation redundant, and at deep U-Net stages the implied "
            "N_pad would exceed the feature-map size."
        )

        # 1×1 channel transition; identity when widths already match.
        if in_channels == out_channels:
            self.pre_proj: torch.nn.Module = torch.nn.Identity()
        else:
            self.pre_proj = PointwiseLinear(in_channels, out_channels)

        # Depthwise spatial conv at out_channels width.
        self.dwconv = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
        )

        # Note: ``CoreBlock`` already binds ``self.norm`` to the *string* name
        # of the norm. We use a distinct attribute for the actual module so the
        # types don't clash.
        match norm:
            case "batch":
                self.norm_layer: torch.nn.Module = torch.nn.BatchNorm2d(out_channels)
            case "instance":
                self.norm_layer = torch.nn.InstanceNorm2d(out_channels)
            case "layer":
                self.norm_layer = LayerNorm2d(out_channels)
            case _:
                raise NotImplementedError(f"Unknown norm: {norm!r}")

        expanded = int(out_channels * upscale_factor)
        self.pw_expand = PointwiseLinear(out_channels, expanded)
        self.act = activation()
        self.pw_project = PointwiseLinear(expanded, out_channels)

        self.checkpoint_simple = checkpoint_simple

    def _maybe_checkpoint(
        self,
        fn: torch.nn.Module,
        x: Float[torch.Tensor, "B C H W"],
    ) -> Float[torch.Tensor, "B C H W"]:
        if self.checkpoint_simple:
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(
        self, x: Float[torch.Tensor, "B C_in H W"]
    ) -> Float[torch.Tensor, "B C_out H W"]:
        x = self.pre_proj(x)
        skip = x

        # Spatial conv: circular-x / zero-y pad, then depthwise k×k.
        x = torch.nn.functional.pad(x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad)
        x = torch.nn.functional.pad(x, (0, 0, self.N_pad, self.N_pad), mode="constant")
        x = self.dwconv(x)

        x = self._maybe_checkpoint(self.norm_layer, x)
        x = self.pw_expand(x)
        x = self._maybe_checkpoint(self.act, x)
        x = self.pw_project(x)

        return skip + x


class RepConvNeXtBlock(TrueConvNeXtBlock):
    """``TrueConvNeXtBlock`` with structural re-parameterization.

    Adds a parallel small-kernel (3×3) depthwise branch alongside the main
    k×k branch, with a per-branch ``BatchNorm2d`` between each conv and the
    sum (matching the canonical RepLKNet recipe). The small branch gives the
    optimizer an "easy path" for fine-scale spatial info that pure-large
    kernels can wash out, and per-branch BN lets the optimizer learn an
    independent scale for each branch.

    Per Ding et al. 2022 (RepLKNet) §5: structural re-parameterization
    helps large depthwise kernels train stably and retain high-frequency
    features, addressing the surface-fidelity loss observed in our
    experiments with large kernels in standard ("True") ConvNeXt blocks.

    Spatial-conv structure (replacing the parent's single ``dwconv``):

        x → ┬─ pad(N_pad)        → dwconv (k×k, bias=False) → bn_large ─┐
            └─ pad(N_pad_small)  → dwconv_small (3×3,    "")  → bn_small ┘
                                                                          ├─→ Σ
                                                                          ┘

    The parent's ``norm_layer`` is replaced with ``Identity()`` because BN
    is now applied per branch. ``norm`` must therefore be ``"batch"``.

    Both branches share input padding mode (circular-x / zero-y), so
    ``fold_reparam()`` produces an exactly equivalent single conv: each
    branch's ``Conv+BN`` is fused into a single ``Conv (bias=True)`` via
    standard BN folding, then the small kernel weights are placed at the
    center of the large kernel and biases are summed.

    When ``kernel_size <= 3`` the small branch is skipped (a parallel 3×3
    next to a 3×3 main conv would duplicate weights).

    References:
        Ding et al. 2022, "Scaling Up Your Kernels to 31×31" §5
        (https://arxiv.org/abs/2203.06717)
        Reference impl:
        https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py
    """

    SMALL_KERNEL = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        pad: str = "circular",
        upscale_factor: int = 4,
        norm: str = "batch",
        checkpoint_simple: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            n_layers=n_layers,
            activation=activation,
            pad=pad,
            upscale_factor=upscale_factor,
            norm=norm,
            checkpoint_simple=checkpoint_simple,
        )
        # Canonical reparam: per-branch BN replaces the post-sum norm. The
        # BN-folding identity used at fold time only handles BatchNorm.
        assert norm == "batch", (
            "RepConvNeXtBlock requires norm='batch' for the per-branch BN "
            "folding identity to be valid."
        )

        # Replace the parent's bias-bearing dwconv with a bias-free variant
        # (BN's beta supplies the effective bias). This matches the reference
        # `conv_bn` building block.
        self.dwconv = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            bias=False,
        )
        self.bn_large: torch.nn.Module = torch.nn.BatchNorm2d(out_channels)

        self.dwconv_small: torch.nn.Module | None = None
        self.bn_small: torch.nn.Module | None = None
        if kernel_size > self.SMALL_KERNEL:
            self.dwconv_small = torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=self.SMALL_KERNEL,
                groups=out_channels,
                bias=False,
            )
            self.bn_small = torch.nn.BatchNorm2d(out_channels)
        self.N_pad_small = self.SMALL_KERNEL // 2

        # Parent's post-spatial norm is now redundant: per-branch BN does
        # the normalization. Replace with Identity so the parent's forward
        # template (called via super interfaces in tests) stays valid.
        self.norm_layer = torch.nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "B C_in H W"]
    ) -> Float[torch.Tensor, "B C_out H W"]:
        x = self.pre_proj(x)
        skip = x

        # Large kernel branch: pad → conv (no bias) → BN.
        x_large = torch.nn.functional.pad(
            x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
        )
        x_large = torch.nn.functional.pad(
            x_large, (0, 0, self.N_pad, self.N_pad), mode="constant"
        )
        out = self.bn_large(self.dwconv(x_large))

        # Parallel small-kernel branch: pad → conv → BN.
        if self.dwconv_small is not None:
            assert self.bn_small is not None
            x_small = torch.nn.functional.pad(
                x, (self.N_pad_small, self.N_pad_small, 0, 0), mode=self.pad
            )
            x_small = torch.nn.functional.pad(
                x_small, (0, 0, self.N_pad_small, self.N_pad_small), mode="constant"
            )
            out = out + self.bn_small(self.dwconv_small(x_small))

        # norm_layer is Identity here; kept in the call sequence for parity
        # with TrueConvNeXtBlock and to honor `checkpoint_simple`.
        x = self._maybe_checkpoint(self.norm_layer, out)
        x = self.pw_expand(x)
        x = self._maybe_checkpoint(self.act, x)
        x = self.pw_project(x)

        return skip + x

    @staticmethod
    def _fuse_conv_bn(
        conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard Conv+BN folding (Conv has no bias).

        Returns ``(weight, bias)`` for the equivalent single conv:
            W_eq = W * (γ / sqrt(σ² + ε)).reshape(-1, 1, 1, 1)
            b_eq = β - μ * γ / sqrt(σ² + ε)
        """
        # `running_var` / `running_mean` are non-None on BatchNorm2d after init;
        # `weight` / `bias` are non-None when affine=True (the default).
        running_var = bn.running_var
        running_mean = bn.running_mean
        gamma = bn.weight
        beta = bn.bias
        assert (
            running_var is not None
            and running_mean is not None
            and gamma is not None
            and beta is not None
        )
        std = (running_var + bn.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        eq_w = conv.weight * scale
        eq_b = beta - running_mean * gamma / std
        return eq_w, eq_b

    @torch.no_grad()
    def fold_reparam(self) -> None:
        """Fold each branch's ``Conv+BN`` into a single equivalent ``Conv``.

        Process:
        1. Fuse ``(dwconv, bn_large)`` → ``(eq_k, eq_b)`` via standard BN
           folding (Conv has bias=False, BN supplies the effective bias).
        2. Same for ``(dwconv_small, bn_small)`` if present.
        3. Sum biases; place small kernel at the centre of the large one.
        4. Replace ``self.dwconv`` with a single bias-bearing conv holding
           the merged weights. ``bn_large`` becomes ``Identity()``; the
           small branch and its BN are removed.

        Both branches share identical padding semantics (circular-x /
        zero-y), so the merged conv is mathematically exact.
        """
        # Already folded.
        if not isinstance(self.bn_large, torch.nn.BatchNorm2d):
            return

        eq_k, eq_b = self._fuse_conv_bn(self.dwconv, self.bn_large)
        if self.dwconv_small is not None:
            assert isinstance(self.dwconv_small, torch.nn.Conv2d)
            assert isinstance(self.bn_small, torch.nn.BatchNorm2d)
            small_k, small_b = self._fuse_conv_bn(self.dwconv_small, self.bn_small)
            eq_b = eq_b + small_b
            c = eq_k.shape[-1] // 2
            s = self.SMALL_KERNEL // 2
            eq_k[:, :, c - s : c + s + 1, c - s : c + s + 1].add_(small_k)

        # Replace dwconv with a bias-bearing Conv holding the merged params.
        # Conv2d's tuple-typed kernel_size/dilation need narrowing for mypy.
        kernel_hw = self.dwconv.kernel_size
        dilation_hw = self.dwconv.dilation
        merged = torch.nn.Conv2d(
            self.dwconv.in_channels,
            self.dwconv.out_channels,
            kernel_size=(kernel_hw[0], kernel_hw[1]),
            dilation=(dilation_hw[0], dilation_hw[1]),
            groups=self.dwconv.groups,
            bias=True,
        )
        merged.weight.data = eq_k
        assert merged.bias is not None
        merged.bias.data = eq_b
        self.dwconv = merged
        self.bn_large = torch.nn.Identity()
        self.dwconv_small = None
        self.bn_small = None


class CoreBlockBuilder(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        n_layers: int,
        pad: str,
        checkpoint_simple: bool,
    ) -> CoreBlock: ...


class UpsamplingBlockBuilder(Protocol):
    def __call__(
        self, in_channels: int, out_channels: int
    ) -> BilinearUpsample | TransposedConvUpsample: ...
