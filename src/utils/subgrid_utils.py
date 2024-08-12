import numpy as np
import xarray as xr
import gcm_filters as gcm
from constants import REGIONS
import torch

# import xesmf as xe


def T_adv_u(u, u_pad, dyu, grid):
    adv = (u * dyu + u_pad.roll(xu_ocean=1) * dyu.roll(xu_ocean=1)) / (
        dyu + dyu.roll(xu_ocean=1)
    )
    adv = adv.rename({"yu_ocean": "yt_ocean"})
    adv["yt_ocean"] = grid.yt_ocean
    return adv


def T_adv_v(v, v_pad, dxu, grid):
    adv = (v * dxu + v_pad.roll(yu_ocean=1) * dxu.roll(yu_ocean=1)) / (
        dxu + dxu.roll(yu_ocean=1)
    )
    adv = adv.rename({"xu_ocean": "xt_ocean"})
    adv["xt_ocean"] = grid.xt_ocean
    return adv


def T_flux_u(T, T_pad, adv_u, grid):
    T_temp = (T + T_pad.roll(xt_ocean=-1)) / 2
    T_temp = T_temp.rename({"xt_ocean": "xu_ocean"})
    T_temp["xu_ocean"] = grid.xu_ocean
    return T_temp * adv_u


def T_flux_v(T, T_pad, adv_v, grid):
    T_temp = (T + T_pad.roll(yt_ocean=-1)) / 2
    T_temp = T_temp.rename({"yt_ocean": "yu_ocean"})
    T_temp["yu_ocean"] = grid.yu_ocean
    return T_temp * adv_v


def T_div(flux_u, flux_v, grid):
    dx = grid["dxt"]
    dx = dx.rename({"xt_ocean": "xu_ocean"})
    dx["xu_ocean"] = grid.xu_ocean

    dy = grid["dyt"]
    dy = dy.rename({"yt_ocean": "yu_ocean"})
    dy["yu_ocean"] = grid.yu_ocean

    flux_u_pad = flux_u.fillna(0)
    flux_v_pad = flux_v.fillna(0)

    divx = (flux_u - flux_u_pad.roll(xu_ocean=1)) / dx
    divx = divx.rename({"xu_ocean": "xt_ocean"})
    divx["xt_ocean"] = grid.xt_ocean

    divy = (flux_v - flux_v_pad.roll(yu_ocean=1)) / dy
    divy = divy.rename({"yu_ocean": "yt_ocean"})
    divy["yt_ocean"] = grid.yt_ocean
    return divx + divy


def U_adv_u(u, u_pad, grid):
    dyu = grid.dyu
    dyu = dyu.rename({"yu_ocean": "yt_ocean"})
    dyu["yt_ocean"] = grid.yt_ocean

    dxt = grid.dxt
    dxt = dxt.rename({"xt_ocean": "xu_ocean"})
    dxt["xu_ocean"] = grid.xu_ocean

    dyu_s = grid.dyu_s
    dyu_s = dyu_s.rename({"yu_ocean": "yt_ocean"})
    dyu_s["yt_ocean"] = grid.yt_ocean

    dyu_n = grid.dyu_n
    dyu_n = dyu_n.rename({"yu_ocean": "yt_ocean"})
    dyu_n["yt_ocean"] = grid.yt_ocean

    dxu_w = grid.dxu_w
    dxu_w = dxu_w.rename({"yu_ocean": "yt_ocean"})
    dxu_w["yt_ocean"] = grid.yt_ocean

    dxu_e = grid.dxu_e
    dxu_e = dxu_e.rename({"yu_ocean": "yt_ocean"})
    dxu_e["yt_ocean"] = grid.yt_ocean

    adv = (
        (u * dyu_s + u_pad.roll(yt_ocean=-1) * dyu_n) * dxu_w.roll(xu_ocean=-1)
        + (
            u_pad.roll(xu_ocean=-1) * dyu_s
            + u_pad.roll(xu_ocean=-1, yt_ocean=-1) * dyu_n
        )
        * dxu_e
    ) / (dyu * dxt.roll(xu_ocean=-1))
    adv = adv.rename({"yt_ocean": "yu_ocean", "xu_ocean": "xt_ocean"})
    adv["xt_ocean"] = grid.xt_ocean
    adv["yu_ocean"] = grid.yu_ocean
    return adv


def U_adv_v(v, v_pad, grid):
    dyt = grid.dyt
    dyt = dyt.rename({"yt_ocean": "yu_ocean"})
    dyt["yu_ocean"] = grid.yu_ocean

    dxu = grid.dxu
    dxu = dxu.rename({"xu_ocean": "xt_ocean"})
    dxu["xt_ocean"] = grid.xt_ocean

    dyu_s = grid.dyu_s
    dyu_s = dyu_s.rename({"xu_ocean": "xt_ocean"})
    dyu_s["xt_ocean"] = grid.xt_ocean

    dyu_n = grid.dyu_n
    dyu_n = dyu_n.rename({"xu_ocean": "xt_ocean"})
    dyu_n["xt_ocean"] = grid.xt_ocean

    dxu_w = grid.dxu_w
    dxu_w = dxu_w.rename({"xu_ocean": "xt_ocean"})
    dxu_w["xt_ocean"] = grid.xt_ocean

    dxu_e = grid.dxu_e
    dxu_e = dxu_e.rename({"xu_ocean": "xt_ocean"})
    dxu_e["xt_ocean"] = grid.xt_ocean

    adv = (
        (v * dxu_w + v_pad.roll(xt_ocean=-1) * dxu_e) * dyu_s.roll(yu_ocean=-1)
        + (
            v_pad.roll(yu_ocean=-1) * dxu_w
            + v_pad.roll(xt_ocean=-1, yu_ocean=-1) * dxu_e
        )
        * dyu_n
    ) / (dxu * dyt.roll(yu_ocean=-1))
    adv = adv.rename({"yu_ocean": "yt_ocean", "xt_ocean": "xu_ocean"})
    adv["xu_ocean"] = grid.xu_ocean
    adv["yt_ocean"] = grid.yt_ocean
    return adv


def U_flux_u(u, u_pad, adv_u, grid):
    u_temp = (u + u_pad.roll(xu_ocean=-1)) / 2
    u_temp = u_temp.rename({"xu_ocean": "xt_ocean"})
    u_temp["xt_ocean"] = grid.xt_ocean
    return u_temp * adv_u


def U_flux_v(u, u_pad, adv_v, grid):
    u_temp = (u + u_pad.roll(yu_ocean=-1)) / 2
    u_temp = u_temp.rename({"yu_ocean": "yt_ocean"})
    u_temp["yt_ocean"] = grid.yt_ocean
    return u_temp * adv_v


def U_div(flux_u, flux_v, grid):
    dx = grid["dxu"]
    dx = dx.rename({"xu_ocean": "xt_ocean"})
    dx["xt_ocean"] = grid.xt_ocean

    dy = grid["dyu"]
    dy = dy.rename({"yu_ocean": "yt_ocean"})
    dy["yt_ocean"] = grid.yt_ocean

    flux_u_pad = flux_u.fillna(0)
    flux_v_pad = flux_v.fillna(0)

    divx = (flux_u - flux_u_pad.roll(xt_ocean=1)) / dx
    divx = divx.rename({"xt_ocean": "xu_ocean"})
    divx["xu_ocean"] = grid.xu_ocean

    divy = (flux_v - flux_v_pad.roll(yt_ocean=1)) / dy
    divy = divy.rename({"yt_ocean": "yu_ocean"})
    divy["yu_ocean"] = grid.yu_ocean
    return divx + divy


def regrid_init(grid_HR, grid_LR, weights_u, weights_T):
    grid_HR_T = xr.Dataset(
        data_vars=dict(mask=(["yt_ocean", "xt_ocean"], grid_HR.wet.data)),
        coords=dict(
            lat_b=(["y_u", "x_u"], grid_HR.y_vert_T.data),
            lon_b=(["y_u", "x_u"], grid_HR.x_vert_T.data),
            lat=(["yt_ocean", "xt_ocean"], grid_HR.y_T.data),
            lon=(["yt_ocean", "xt_ocean"], grid_HR.x_T.data),
        ),
    )

    grid_HR_u = xr.Dataset(
        data_vars=dict(mask=(["yu_ocean", "xu_ocean"], grid_HR.wet_c.data)),
        coords=dict(
            lat_b=(["y_t", "x_t"], grid_HR.y_vert_C.data),
            lon_b=(["y_t", "x_t"], grid_HR.x_vert_C.data),
            lat=(["yu_ocean", "xu_ocean"], grid_HR.y_C.data),
            lon=(["yu_ocean", "xu_ocean"], grid_HR.x_C.data),
        ),
    )

    grid_LR_T = xr.Dataset(
        data_vars=dict(mask=(["yt_ocean", "xt_ocean"], grid_LR.wet.data)),
        coords=dict(
            lat_b=(["y_u", "x_u"], grid_LR.y_vert_T.data),
            lon_b=(["y_u", "x_u"], grid_LR.x_vert_T.data),
            lat=(["yt_ocean", "xt_ocean"], grid_LR.y_T.data),
            lon=(["yt_ocean", "xt_ocean"], grid_LR.x_T.data),
        ),
    )

    grid_LR_u = xr.Dataset(
        data_vars=dict(mask=(["yu_ocean", "xu_ocean"], grid_LR.wet_c.data)),
        coords=dict(
            lat_b=(["y_t", "x_t"], grid_LR.y_vert_C.data),
            lon_b=(["y_t", "x_t"], grid_LR.x_vert_C.data),
            lat=(["yu_ocean", "xu_ocean"], grid_LR.y_C.data),
            lon=(["yu_ocean", "xu_ocean"], grid_LR.x_C.data),
        ),
    )

    regridder_T = xe.Regridder(
        grid_HR_T, grid_LR_T, "conservative_normed", periodic=True, weights=weights_T
    )
    regridder_u = xe.Regridder(
        grid_HR_u, grid_LR_u, "conservative_normed", periodic=True, weights=weights_u
    )

    return regridder_T, regridder_u


def coarse_field_grid(variable, grid, factor):
    var = grid[variable]
    if "dx" in variable or "dn" in variable:

        if var.dims[0] == "yt_ocean":
            var_coarse = var.coarsen(xt_ocean=factor, boundary="trim").sum()
            var_coarse = var_coarse.coarsen(yt_ocean=factor, boundary="trim").mean()

        else:
            var_coarse = var.coarsen(xu_ocean=factor, boundary="trim").sum()
            var_coarse = var_coarse.coarsen(yu_ocean=factor, boundary="trim").mean()

    elif "dy" in variable or "de" in variable:

        if var.dims[0] == "yt_ocean":
            var_coarse = var.coarsen(yt_ocean=factor, boundary="trim").sum()
            var_coarse = var_coarse.coarsen(xt_ocean=factor, boundary="trim").mean()

        else:
            var_coarse = var.coarsen(yu_ocean=factor, boundary="trim").sum()
            var_coarse = var_coarse.coarsen(xu_ocean=factor, boundary="trim").mean()

    elif "area" in variable:

        if var.dims[0] == "yt_ocean":
            #             var_weighted = var*grid.wet
            var_coarse = var.coarsen(
                xt_ocean=factor, yt_ocean=factor, boundary="trim"
            ).sum()
        else:
            #             var_weighted = var*grid.wet_c
            var_coarse = var.coarsen(
                xu_ocean=factor, yu_ocean=factor, boundary="trim"
            ).sum()
    else:

        if var.dims[0] == "yt_ocean":
            var_weighted = var * grid.area_T

            var_coarse = var_weighted.coarsen(
                xt_ocean=factor, yt_ocean=factor, boundary="trim"
            ).mean()
            var_coarse = (
                var_coarse
                / grid.area_T.coarsen(
                    xt_ocean=factor, yt_ocean=factor, boundary="trim"
                ).mean()
            )
        else:
            var_weighted = var * grid.area_C

            var_coarse = var_weighted.coarsen(
                xu_ocean=factor, yu_ocean=factor, boundary="trim"
            ).mean()
            var_coarse = (
                var_coarse
                / grid.area_C.coarsen(
                    xu_ocean=factor, yu_ocean=factor, boundary="trim"
                ).mean()
            )
    return var_coarse


def coarse_grid(grid, factor):
    grid_coarse = grid.coarsen(
        xu_ocean=factor,
        yu_ocean=factor,
        xt_ocean=factor,
        yt_ocean=factor,
        boundary="trim",
    ).mean()
    for var in grid.variables:
        if var == "xt_ocean" or var == "x_vert_T":
            break
        grid_coarse[var] = coarse_field_grid(var, grid, factor)
        if "wet" in var:
            wet_coarse = grid_coarse[var]
            wet_clip = wet_coarse.where(wet_coarse < 0.5, 1)
            wet_clip = wet_clip.where(wet_clip >= 0.5, 0)
            grid_coarse[var] = wet_clip
    return grid_coarse


def get_area_tensor(grid_path, region, device, factor=10):
    # Get region boundaries
    lat, lon = lat, lon = REGIONS[region]["lat"], REGIONS[region]["lon"]

    grids = xr.open_dataset(grid_path)
    if region == "global_25":
        grids = xr.open_dataset(grid_path)

    elif "global" in region:
        grids = coarse_grid(grids, factor)

    else:
        grids = grids.sel(
            {"yu_ocean": slice(lat[0], lat[1]), "xu_ocean": slice(lon[0], lon[1])}
        )

    area = torch.from_numpy(grids["area_C"].to_numpy()).to(device=device, non_blocking=True)
    return area


def coarse_field(var, grid, factor):

    if var.dims[1] == "yt_ocean":
        var_weighted = var * grid.area_T * grid.wet

        var_coarse = var_weighted.coarsen(
            xt_ocean=factor, yt_ocean=factor, boundary="trim"
        ).sum()
        var_coarse = (
            var_coarse
            / (grid.area_T * grid.wet)
            .coarsen(xt_ocean=factor, yt_ocean=factor, boundary="trim")
            .sum()
        )
    else:
        var_weighted = var * grid.area_C * grid.wet_c

        var_coarse = var_weighted.coarsen(
            xu_ocean=factor, yu_ocean=factor, boundary="trim"
        ).sum()
        var_coarse = (
            var_coarse
            / (grid.area_C * grid.wet_c)
            .coarsen(xu_ocean=factor, yu_ocean=factor, boundary="trim")
            .sum()
        )
    return var_coarse
