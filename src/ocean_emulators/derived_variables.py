from collections.abc import Callable

import torch
from jaxtyping import Float

from ocean_emulators.constants import CP_SW, RHO_0, Grid, TensorMap


def compute_ocean_heat_content(
    T: Float[Grid, "*batch depth"] | Float[Grid, "*batch hist depth"], dz: torch.Tensor
) -> Float[Grid, "*batch"] | Float[Grid, "*batch hist"]:
    """Compute the heat content of the ocean.

    Args:
        T: Temperature tensor
        dz: Depth tensor of shape (depth,)

    Returns:
        Heat content tensor
    """
    mask = torch.isnan(T).any()

    # Compute heat content per layer
    if T.ndim == 4:
        depth_dim = 1
        surface_wet_mask = ~T[:, 0].isnan()
    elif T.ndim == 5:
        depth_dim = 2
        surface_wet_mask = ~T[:, :, 0].isnan()
    else:
        raise ValueError(f"Invalid number of dimensions: {T.ndim}")

    view_shape = [1] * T.ndim
    view_shape[depth_dim] = -1

    # Compute heat content per layer
    HC_t = RHO_0 * CP_SW * T * dz.view(view_shape)

    # Column integrated heat content
    total_HC_t = torch.nansum(HC_t, dim=depth_dim)

    if mask:
        total_HC_t = torch.where(surface_wet_mask, total_HC_t, torch.nan)

    return total_HC_t


def compute_global_ocean_heat_content(
    T: Float[Grid, "*batch depth"] | Float[Grid, "*batch hist depth"],
    dz: torch.Tensor,
    area_weighted_func: Callable,
) -> torch.Tensor:
    """Compute the global heat content of the ocean.

    Args:
        T: Temperature tensor of shape
        dz: Depth tensor of shape (depth,)
        area_weighted_func: Area weighted function
    Returns:
        Global heat content tensor of shape (batch_size,)
    """
    total_HC_t = compute_ocean_heat_content(T, dz)

    # Area weighted sum
    global_HC_t = area_weighted_func(total_HC_t)  # (batch,) [J]

    return global_HC_t


def add_derived_variables(tensor_out: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Add derived variables to the output.
    """
    # Ocean heat content
    derived_vars = {}
    tensor_map = TensorMap.get_instance()
    dz = tensor_map.dz.to(tensor_out.device)
    thetao = tensor_out[:, :, tensor_map.VAR_3D_IDX["thetao"]]
    ohct = compute_ocean_heat_content(thetao, dz)
    derived_vars["ocean_heat_content"] = ohct

    return derived_vars
