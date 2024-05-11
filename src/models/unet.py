import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils.climate_utils import pairwise
from .modules.blocks import CoreBlock, BilinearUpsample, TransposedConvUpsample
import torch.nn as nn
import numpy as np
from .base import BaseModel

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
            pred_residuals=False, 
            last_kernel_size = 3, 
            pad="circular"):
        super().__init__(ch_width, n_out, wet, pred_residuals, last_kernel_size, pad)

        # going down
        layers = []
        for i, (a,b) in enumerate(pairwise(ch_width)):
            layers.append(instantiate(
                        core_block,
                        a,
                        b, 
                        dilation=dilation[i], 
                        n_layers=n_layers[i], 
                        activation=activation, 
                        pad=pad)
                        )
            layers.append(instantiate(down_sampling_block))
        layers.append(instantiate(
                    core_block,
                    b,
                    b, 
                    dilation=dilation[i], 
                    n_layers=n_layers[i], 
                    activation=activation, 
                    pad=pad)
                    )
        layers.append(instantiate(up_sampling_block))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a,b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(instantiate(
                        core_block,
                        a,
                        b, 
                        dilation=dilation[i], 
                        n_layers=n_layers[i], 
                        activation=activation, 
                        pad=pad)
                        )
            layers.append(instantiate(up_sampling_block))
        layers.append(instantiate(
                    core_block,
                    b,
                    b, 
                    dilation=dilation[i], 
                    n_layers=n_layers[i], 
                    activation=activation, 
                    pad=pad)
                    )
        layers.append(torch.nn.Conv2d(b, n_out, last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)

        #self.layers = nn.ModuleList(layer)

    def forward_once(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l, nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l, CoreBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l, BilinearUpsample) or isinstance(l, TransposedConvUpsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return torch.mul(fts,self.wet)