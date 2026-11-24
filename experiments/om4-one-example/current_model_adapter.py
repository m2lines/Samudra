"""Use the unmodified pinned main Samudra model in the single-example harness."""
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from samudra.config import SamudraConfig
from samudra.utils.ctx import BatchGrid


class CurrentModel(torch.nn.Module):
    def __init__(self, wet):
        super().__init__()
        root = Path(os.environ['SAMUDRA_SOURCE'])
        config = SamudraConfig.model_validate(yaml.safe_load(
            (root/'src/samudra/configs/samudra_om4/model.yaml').read_text()))
        assert config.unet.core_block.norm == 'instance' and config.use_bfloat16
        self.model = config.build(prog_channels=154, boundary_channels=8,
                                  out_channels=154, input_steps=2,
                                  grid_sizes=[tuple(wet.shape[-2:])])
        self.register_buffer('wet', wet.repeat(2,1,1))
        xy = np.load(os.environ['OM4_COORDS'])
        self.coords = (torch.tensor(xy['y']),torch.tensor(xy['x']))

    def forward(self, x, b):
        ctx = BatchGrid(self.wet, self.coords, self.coords)
        return [self.model.forward_once(x, b, ctx)]
