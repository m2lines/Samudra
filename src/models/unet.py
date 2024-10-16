import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils.climate_utils import pairwise
from .modules.blocks import CoreBlock, BilinearUpsample, BilinearUpsample3D, TransposedConvUpsample, ConvNeXtTemporalBlock
import torch.nn as nn
import numpy as np
from .base import BaseModel
from einops import rearrange

class UNet(BaseModel):
    def __init__(
        self,
        core_block: DictConfig,
        down_sampling_block: DictConfig,
        up_sampling_block: DictConfig,
        activation: DictConfig,
        ch_width,
        n_out,
        dilation,
        n_layers,
        wet,
        hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
        super().__init__(
            ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
        )

        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            layers.append(
                instantiate(
                    core_block,
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(instantiate(down_sampling_block))
        layers.append(
            instantiate(
                core_block,
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(instantiate(up_sampling_block, in_channels=b, out_channels=b))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                instantiate(
                    core_block,
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(instantiate(up_sampling_block, in_channels=b, out_channels=b))
        layers.append(
            instantiate(
                core_block,
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(torch.nn.Conv2d(b, n_out, last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def forward_once(self, fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = l(fts)
            if count < self.num_steps:
                if isinstance(l, CoreBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l, BilinearUpsample) or isinstance(
                    l, TransposedConvUpsample
                ):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        return torch.mul(fts, self.wet)

class UNet3D(BaseModel):
    def __init__(
        self,
        core_block: DictConfig,
        temporal_block: DictConfig,
        down_sampling_block: DictConfig,
        up_sampling_block: DictConfig,
        activation: DictConfig,
        ch_width,
        n_out,
        dilation,
        n_layers,
        wet,
        hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
        super().__init__(
            ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
        )

        # going down
        enc_layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            enc_layers.append(
                instantiate(
                    core_block,
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            enc_layers.append(instantiate(down_sampling_block))
        middle_layers = []
        middle_layers.append(
            instantiate(
                core_block,
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        for i, _ in enumerate(pairwise(ch_width)):
            middle_layers.append(
                instantiate(
                    temporal_block,
                    b * (hist + 1),
                    b * (hist + 1),
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
        dec_layers = []
        dec_layers.append(instantiate(up_sampling_block, in_channels=b, out_channels=b))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            dec_layers.append(
                instantiate(
                    core_block,
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            dec_layers.append(instantiate(up_sampling_block, in_channels=b, out_channels=b))
        dec_layers.append(
            instantiate(
                core_block,
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        dec_layers.append(torch.nn.Conv3d(b, n_out, (1, last_kernel_size, last_kernel_size)))

        self.enc_layers = nn.ModuleList(enc_layers)
        self.middle_layers = nn.ModuleList(middle_layers)
        self.dec_layers = nn.ModuleList(dec_layers)
        
        self.num_encoder_steps = int(len(ch_width) - 1)
        self.middle_temporal_steps = int(len(ch_width))

    def forward_once(self, fts):
        temp = []
        for i in range(self.num_encoder_steps):
            temp.append(None)
        count = 0
        for l in self.enc_layers:
            assert l.__class__.__name__ == "ConvNeXtBlock3D" or l.__class__.__name__ == "AvgPool3D", f"Encoder Layer cannot be {l.__class__.__name__}"
            fts = l(fts)
            if isinstance(l, CoreBlock):
                temp[count] = fts
                count+=1
        
        for l in self.middle_layers:
            assert l.__class__.__name__ == "ConvNeXtTemporalBlock" or l.__class__.__name__ == "ConvNeXtBlock3D", f"Temporal Layer cannot be {l.__class__.__name__}"
            fts += l(fts)
        
        for l in self.dec_layers:
            crop = fts.shape[2:]
            assert l.__class__.__name__ == "BilinearUpsample3D" or l.__class__.__name__ == "TransposedConvUpsample"\
                or l.__class__.__name__ == "ConvNeXtBlock3D" or l.__class__.__name__ == "Conv3d", f"Decoder Layer cannot be {l.__class__.__name__}"

            if isinstance(l, nn.Conv3d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad, 0, 0), mode="constant"
                )
                
            fts = l(fts)
            if isinstance(l, BilinearUpsample3D) or isinstance(
                l, TransposedConvUpsample
            ):
                crop = np.array(fts.shape[2:])
                shape = np.array(
                    temp[int(2 * self.num_encoder_steps - count - 1)].shape[2:]
                )
                pads = shape - crop
                pads = [
                    pads[2] // 2,
                    pads[2] - pads[2] // 2,
                    pads[1] // 2,
                    pads[1] - pads[1] // 2,
                    0, 
                    0
                ]
                fts = nn.functional.pad(fts, pads)
                fts += temp[int(2 * self.num_encoder_steps - count - 1)]
                count += 1

        fts = rearrange(fts, 'b c t h w -> b t c h w')
        fts = torch.mul(fts, self.wet)
        fts = rearrange(fts, 'b t c h w -> b c t h w')
        return fts