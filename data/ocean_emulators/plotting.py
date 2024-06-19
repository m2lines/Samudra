import matplotlib.pyplot as plt
from xarrayutils.plotting import linear_piecewise_scale
import xarray as xr


def qc_plots(ds: xr.Dataset):
    ds = ds.squeeze()
    ds = ds.drop(["variant_label", "sub_experiment_id"])

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
