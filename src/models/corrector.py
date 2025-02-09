from typing import List, Optional

import numpy as np
import torch

from constants import TensorMap


class Corrector(torch.nn.Module):
    def __init__(self, corrector_config):
        super().__init__()
        self.corrector_config = corrector_config
        self.non_negative_corrector_names: Optional[List[str]] = (
            corrector_config.non_negative_corrector_names
        )
        self.tensor_map: TensorMap = TensorMap.get_instance()
        if self.non_negative_corrector_names is not None:
            self.non_neg_indices = torch.cat(
                [
                    self.tensor_map.VAR_3D_IDX[name]
                    for name in self.non_negative_corrector_names
                ],
                dim=0,
            )
        else:
            self.non_neg_indices = torch.tensor(np.nan)

    def forward(self, fts):
        if torch.isnan(self.non_neg_indices).any():
            indices = self.non_neg_indices.to(fts.device)
            fts[:, indices, :, :] = torch.relu(fts[:, indices, :, :])

        return fts
