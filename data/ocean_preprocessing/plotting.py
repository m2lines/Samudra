# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from xarrayutils.plotting import linear_piecewise_scale


def qc_plots(ds: xr.Dataset):
    ## plot maps
    fig, axarr = plt.subplots(ncols=2, nrows=3, figsize=[15, 13])
    for var, ax in zip(ds.data_vars, axarr.flat):
        da = ds[var].isel(time=0, lev=0, missing_dims="ignore").load()
        kwargs = {"x": "x"}
        da.plot(ax=ax, **kwargs)
        ax.set_title("Surface Snapshot")

    plt.show()

    ## plot simple (non-weighted averages) over time (and potentially depth)
    fig, axarr = plt.subplots(ncols=2, nrows=3, figsize=[15, 18])
    for var, ax in zip(ds.data_vars, axarr.flat):
        da = ds[var].mean(["x", "y"]).load()
        kwargs = {"x": "time"}
        if "lev" in da.dims:
            kwargs["yincrease"] = False

        da.plot(ax=ax, **kwargs)

        if "lev" not in da.dims:
            da.rolling(time=12).mean().plot(
                ax=ax, **kwargs, label="12 month rolling mean"
            )
            ax.legend()
        else:
            linear_piecewise_scale(1000, 5, ax=ax)
            # indicate the point between the different scalings
            ax.axhline(1000, color="0.5", ls="--")
            # Rearange the yticks
            ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        ax.set_title("Unweighted global mean")

    plt.show()

    ### show stdv over time averaged over longitudes
    fig, axarr = plt.subplots(ncols=2, nrows=3, figsize=[15, 18])
    for var, ax in zip(ds.data_vars, axarr.flat):
        da = ds[var].mean("x").std("time").load()
        kwargs = {"x": "y"}
        if "lev" in da.dims:
            kwargs["yincrease"] = False
            kwargs["robust"] = True

        da.plot(ax=ax, **kwargs)
        if "lev" in da.dims:
            linear_piecewise_scale(1000, 5, ax=ax)
            # indicate the point between the different scalings
            ax.axhline(1000, color="0.5", ls="--")
            # Rearange the yticks
            ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        ax.set_title("Stdv in time of zonal unweighted mean")

    plt.show()


####### QC plotting for preprocessing #########
def rotated_vectors_qc_plots(u, v, u_rotated, v_rotated):
    roi = dict(lev=0, time=min(100, len(u.time) - 1), y=slice(900, None))
    kwargs = dict(x="lon", y="lat", robust=True, transform=ccrs.PlateCarree())

    fig, axarr = plt.subplots(
        ncols=2,
        nrows=2,
        subplot_kw=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
    )
    u.isel(**roi, missing_dims="warn").plot(ax=axarr.flat[0], **kwargs)
    v.isel(**roi, missing_dims="warn").plot(ax=axarr.flat[1], **kwargs)

    u_rotated.isel(**roi, missing_dims="warn").plot(ax=axarr.flat[2], **kwargs)
    v_rotated.isel(**roi, missing_dims="warn").plot(ax=axarr.flat[3], **kwargs)

    for title, ax in zip(["u", "v", "u rotated", "v rotated"], axarr.flat):
        ax.set_title(title)

    # compare 'seam line'
    fig, axarr = plt.subplots(ncols=2, nrows=2, figsize=[8, 6])

    def fold_and_compare(ax: matplotlib.axes._axes.Axes, seam_data: xr.DataArray):
        middle = len(seam_data.x) // 2
        left = seam_data.isel(y=-1, x=slice(0, middle)).data
        right_flipped = seam_data.isel(y=-1, x=slice(middle, None)).data[::-1]

        ax.plot(left)
        ax.plot(right_flipped, ls="--")

    ax = axarr.flat[0]
    fold_and_compare(ax, u.isel(**roi, missing_dims="warn").load())
    ax.set_ylabel("u")
    ax.set_title("Before rotation: \n should be mirrored")

    ax = axarr.flat[1]
    fold_and_compare(ax, v.isel(**roi, missing_dims="warn").load())
    ax.set_ylabel("v")
    ax.set_title("Before rotation: \n should be mirrored")

    ax = axarr.flat[2]
    fold_and_compare(ax, u_rotated.isel(**roi, missing_dims="warn").load())
    ax.set_ylabel("u")
    ax.set_title("After rotation: \n should be aligned")

    ax = axarr.flat[3]
    fold_and_compare(ax, v_rotated.isel(**roi, missing_dims="warn").load())
    ax.set_ylabel("v")
    ax.set_title("After rotation: \n should be aligned")

    fig.tight_layout()
