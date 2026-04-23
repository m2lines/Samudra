from typing import Literal

import torch
import torch.nn.functional as F

PadType = Literal["circular", "constant", "reflect", "replicate", "halo_sponge"]


def resolved_x_pad_mode(pad_mode: PadType | str) -> str:
    if pad_mode == "halo_sponge":
        return "replicate"
    return pad_mode


def apply_spatial_pad(
    tensor: torch.Tensor,
    n_pad: int,
    pad_mode: PadType | str,
) -> torch.Tensor:
    if n_pad == 0:
        return tensor

    if pad_mode == "halo_sponge":
        # Simple regional-context padding: extend the nearest boundary value
        # on all four sides instead of injecting zeros beyond the patch.
        return F.pad(tensor, (n_pad, n_pad, n_pad, n_pad), mode="replicate")

    tensor = F.pad(tensor, (n_pad, n_pad, 0, 0), mode=pad_mode)
    return F.pad(tensor, (0, 0, n_pad, n_pad), mode="constant")
