from datetime import timedelta

import cftime
import torch

from constants import DEPTH_LEVELS


def extract_wet_mask(wet_zarr, outputs, hist):
    depth_ind = []
    for var_depth_i in outputs:
        ind = var_depth_i.split("_")[-1]
        if ind == "zos":
            depth_ind.append("0")
        else:
            depth_ind.append(ind)
    depths = [DEPTH_LEVELS[int(depth_i)] for depth_i in depth_ind]
    wet = wet_zarr.sel(lev=depths)
    wet = torch.from_numpy(wet.to_array().to_numpy().squeeze())
    wet = torch.concat([wet] * (hist + 1), dim=0)
    return wet


def get_time_slice(time_config, initial_cond=False, time_delta=5, hist=1):
    start_time_str = time_config.start_time
    start_year, start_month, start_day = start_time_str.split("-")
    start_time = cftime.DatetimeNoLeap(
        int(start_year), int(start_month), int(start_day), 0, 0, 0
    )

    end_time_str = time_config.end_time
    end_year, end_month, end_day = end_time_str.split("-")
    end_time = cftime.DatetimeNoLeap(
        int(end_year), int(end_month), int(end_day), 0, 0, 0
    )
    num_steps = (end_time - start_time).days // time_delta + 1

    if initial_cond:
        start_time = start_time - timedelta(
            days=time_delta * (hist + 1)
        )  # Prepending initial condition

    return slice(start_time, end_time), num_steps
