import numpy as np
import torch
from einops import rearrange

from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.data import Normalize


class Corrector(torch.nn.Module):
    def __init__(self, corrector_config, hist):
        super().__init__()
        self.corrector_config = corrector_config
        self.non_negative_corrector_names: list[str] | None = (
            corrector_config.non_negative_corrector_names
        )
        self.tensor_map: TensorMap = TensorMap.get_instance()
        self.normalize = Normalize.get_instance()
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
        self.hist = hist

    def forward(self, fts):
        if not torch.isnan(self.non_neg_indices).all():
            # Reshape for unnormalization
            fts_reshaped = rearrange(
                fts, "n (hist c) h w -> n hist c h w", hist=self.hist + 1
            )
            fts_reshaped = rearrange(fts_reshaped, "n hist c h w -> (n hist) c h w")

            # unnormalize
            unnormalized_fts = self.normalize.unnormalize_tensor_prognostic(
                fts_reshaped, fill_value=0.0
            )

            # apply relu
            indices = self.non_neg_indices.to(unnormalized_fts.device)
            # unnormalized_fts_cloned = unnormalized_fts.clone()
            unnormalized_fts[:, indices, :, :] = torch.relu(
                unnormalized_fts[:, indices, :, :]
            )
            # renormalize
            fts_reshaped_relued = self.normalize.normalize_tensor_prognostic(
                unnormalized_fts
            )
            fts_reshaped_relued = rearrange(
                fts_reshaped_relued,
                "(n hist) c h w -> n hist c h w",
                hist=self.hist + 1,
            )
            fts_relued = rearrange(
                fts_reshaped_relued, "n hist c h w -> n (hist c) h w"
            )

            # assert
            # fts_norm = self.normalize.normalize_tensor_prognostic(
            # unnormalized_fts_cloned
            # )
            # fts_norm = rearrange(
            #     fts_norm, "(n hist) c h w -> n hist c h w", hist=self.hist + 1
            # )
            # fts_norm = rearrange(fts_norm, "n hist c h w -> n (hist c) h w")
            # print("fts_norm max: ", fts_norm.max())
            # print("fts max: ", fts.max())
            # print("fts_norm min: ", fts_norm.min())
            # print("fts min: ", fts.min())
            # assert torch.allclose(
            #     fts_norm, fts, atol=1e-4
            # )  # TODO: 1e-5 triggers assertion

            fts = fts_relued
        return fts
