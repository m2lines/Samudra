"""Preprocess arbitrary datasets to standardized naming, grids"""

from xgcm import Grid
import xarray as xr
import numpy as np
import cf_xarray
from ocean_emulators.utils import split_2d_3d
import gcm_filters

try:
    import xesmf as xe  # type: ignore
except ImportError:
    xe = None
# Could I replace this with the xarray logic I am using in the tests?


def manual_v0_fixes(ds_input: xr.Dataset) -> xr.Dataset:
    """Manual fixes for the already existing data (for now only v0.0). This should not be used in the future"""
    # fixes that should be checked and fixes on the input data
    area = xr.open_dataset(
        "gs://leap-persistent/sd5313/grids_CM2x.zarr", engine="zarr", chunks={}
    )["area_C"].rename({"xu_ocean": "x", "yu_ocean": "y"})
    # from https://github.com/m2lines/ocean_emulators/issues/17
    dz = xr.DataArray(
        [
            5,
            10,
            15,
            20,
            30,
            50,
            70,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            800,
            1000,
            1000,
            1000,
        ],
        dims=["lev"],
    )
    z = xr.DataArray(
        [
            2.5,
            10,
            22.5,
            40,
            65,
            105,
            165,
            250,
            375,
            550,
            775,
            1050,
            1400,
            1850,
            2400,
            3100,
            4000,
            5000,
            6000,
        ],
        dims="lev",
    )
    wetmask = ~np.isnan(ds_input.thetao.isel(time=0).reset_coords(drop=True)).load()
    lon = xr.ones_like(ds_input.y) * ds_input.x
    lat = ds_input.y * xr.ones_like(ds_input.x)
    ds_input = ds_input.assign_coords(
        areacello=area, dz=dz, lev=z, wetmask=wetmask, lon=lon, lat=lat
    )
    # give a dummy commit hash
    ds_input.attrs["m2lines/ocean-emulators_git_hash"] = "dummy"
    return ds_input


# i need to test 2d and 3d separately


# def rename(ds: xr.Dataset) -> xr.Dataset:
#     """Rename variables and dimensions to CMOR standard names"""
#     # TODO: how to detect non-CMIP datasets?
#     return combined_preprocessing(ds)


# def standardize_dataset(ds_ocean: xr.Dataset, ds_atmos: xr.Dataset) -> xr.Dataset:
#     """Full wrapper that does
#     1. Rename variables and dimensions to CMOR standard names
#     2. Combine varibles if necessary
#     3. Interpolate velocity to tracer cells
#     4. Optional Filter the data
#     5. Horizontal regridding
#     6. Add metadata and provenance info
#     """
#     # Rename variables and dimensions to CMOR standard names
#     ds_renamed = rename(ds)
#     # Combine varibles if necessary
#     # Interpolate velocity to tracer cells
#     # Optional Filter the data
#     # Horizontal regridding
#     # Add metadata and provenance info
#     return ds_renamed


#################### CMIP specific Code ###########################
def infer_vertical_cell_extent(ds: xr.Dataset, dz_name: str = "dz") -> xr.Dataset:
    """
    recomputes z* vertical cell extent according to

    thkcello is the nominal cell thickness in z* coordinates. The model actual thkcello is time-dependent and can be calculated as thkcello * ( deptho + zos ) / deptho
    """
    required_vars = ["thkcello", "deptho", "zos"]

    if not all(v in ds.variables for v in required_vars):
        raise ValueError(
            f"Could not find {set(required_vars)-set(ds.variables)} in datasset coords. Found {list(ds.coords)}"
        )

    ds = ds.assign_coords({dz_name: ds.thkcello * (ds.deptho + ds.zos) / ds.deptho})
    return ds


def cmip_vertical_outer_grid(ds: xr.Dataset) -> xr.Dataset:
    # TODO: Check if an outer grid position is already available (e.g. from combining tracer and vertical velocities in xmip.grids.something_staggered_grid

    # TODO: Ask alistair if it is ok to just use the nominal depth levels + extensive quantities?
    lev_outer = cf_xarray.bounds_to_vertices(ds["lev_bounds"], "bnds").rename(
        {"lev_vertices": "lev_outer"}
    )
    ds = ds.assign_coords({"lev_outer": lev_outer})
    # set up an xgcm grid
    # FIXME: This should work with metadata!
    grid = Grid(
        ds,
        coords={"Z": {"center": "lev", "outer": "lev_outer"}},
        boundary="fill",
        autoparse_metadata=False,
    )
    return grid, ds


##################### General Code #################
def rotate_vectors(u, v, angle):
    """rotates vector components u and v using `angle`
    (assumed to be defined in deg, and in the CCW direction)
    Currently only works when all components are on the same grid position
    """
    # angle should be a 2d array
    if not len(angle.dims) == 2:
        raise ValueError(f"Expected only two dimensions on `angle`. Got {angle.dims}")
    # assert that all components are on the same position
    if not (
        set(angle.dims).issubset(set(u.dims)) and set(angle.dims).issubset(set(v.dims))
    ):
        raise ValueError("`u` and `v` need to be on the same grid position as `angle`.")

    # rotate velocities
    theta = np.deg2rad(angle)
    vec = xr.concat([u, v], dim="dim_in")
    # construct rotation matrix
    rot = xr.concat(
        [
            xr.concat([np.cos(theta), np.sin(theta)], dim="dim_out"),
            xr.concat([-np.sin(theta), np.cos(theta)], dim="dim_out"),
        ],
        dim="dim_in",
    )
    rotated_vector = xr.dot(vec, rot, dim="dim_in", optimize=True)
    u_rotated = rotated_vector.isel(dim_out=0)
    v_rotated = rotated_vector.isel(dim_out=1)
    return u_rotated, v_rotated


def vertical_regrid(ds_raw: xr.Dataset, target_depth_bounds: np.ndarray) -> xr.Dataset:
    # reconstruct vertical bounds
    # TODO (this should be done outside to make this function more general)
    grid, ds = cmip_vertical_outer_grid(ds_raw)
    # split out the 2d variables
    ds_2d = xr.Dataset(
        {var: ds[var] for var in ds.data_vars if "lev" not in ds[var].dims}
    )
    ds = ds.drop_vars(list(ds_2d.data_vars))

    dz = ds["dz"]
    ds_extensive = ds * dz

    ds_extensive_regridded = xr.Dataset()
    for var in ds_extensive.data_vars:
        # TODO: assert that lev is actually on this variable, otherwise what?
        ds_extensive_regridded[var] = grid.transform(
            ds_extensive[var],
            "Z",
            target_depth_bounds,
            target_data=ds.lev_outer,
            method="conservative",
        )

    # by default this is named after the 'target_data', but for the purpose of simplicity, lets rename this here
    ds_extensive_regridded = ds_extensive_regridded.rename({"lev_outer": "lev"})

    # Calculate the cell thickness of the target grid.
    dz_regridded = xr.DataArray(
        np.diff(target_depth_bounds),
        dims=["lev"],
        coords={"lev": ds_extensive_regridded.thetao.lev},
    )

    ds_regridded = ds_extensive_regridded / dz_regridded
    ds_regridded = ds_regridded.assign_coords(dz=dz_regridded)
    for co_name, co in ds.coords.items():
        if "lev" not in co.dims:
            ds_regridded = ds_regridded.assign_coords({co_name: co})
    ds_regridded = ds_regridded.drop("lev_outer")
    # merge the 2d variables back in
    ds_regridded = xr.merge([ds_regridded, ds_2d])
    ds_regridded.attrs = ds_raw.attrs
    return ds_regridded


def spatially_filter(
    ds: xr.Dataset, w_mask, filter_scale=18, depth_dim="lev", y_dim="y", x_dim="x"
):
    """Applies a spatial filter with 3d/2d wetmask depending on the variable dimensions"""
    wmask_3d = (w_mask == 1).astype(int).reset_coords(drop=True)
    depth_indexer = {depth_dim: 0}
    wmask_2d = wmask_3d.isel(**depth_indexer).drop_vars(depth_dim)

    ds_2d, ds_3d = split_2d_3d(ds, depth_dim=depth_dim)
    ds_2d = ds_2d.reset_coords(drop=True)
    ds_3d = ds_3d.reset_coords(drop=True)

    filt_2d = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": wmask_2d},  # why can gcm filters not accept bool masks?
    )
    filt_3d = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": wmask_3d},  # why can gcm filters not accept bool masks?
    )
    datasets = [
        filt_2d.apply(ds_2d, dims=[y_dim, x_dim]),
        filt_3d.apply(ds_3d, dims=[y_dim, x_dim]),
    ]
    ds_filtered = xr.merge(datasets)
    # get attrs and coords back
    ds_filtered = ds_filtered.assign_coords({co: ds[co] for co in ds.coords})
    ds_filtered.attrs = ds.attrs
    return ds_filtered


def horizontal_regrid(ds, ds_target):
    """Regrid `ds` horizontally, and conserve the integral in space"""
    regridder_kwargs = dict(ignore_degenerate=True, periodic=True, unmapped_to_nan=True)

    # try to run this with higher precision (TODO: Test if this actually makes a difference).
    s = xr.Dataset(
        coords={
            co: ds[co].astype("float128") for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )
    t = xr.Dataset(
        coords={
            co: ds_target[co].astype("float128")
            for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )

    regridder = xe.Regridder(s, t, "conservative", **regridder_kwargs)
    ds_regridded = regridder(ds, skipna=True, na_thres=1)

    # get lon/lats from the target grid
    lon = ds_target.lon
    lat = ds_target.lat

    lon_b = ds_target.lon_b
    lat_b = ds_target.lat_b

    # get x and y values
    x = lon.isel(y=0)
    y = lat.isel(x=0)

    # calculate new area
    r_earth = 6356  # in km
    new_area = xe.util.cell_area(ds_target, r_earth) * 1e6

    ## calculate the wetmask afterwards...
    wetmask = ~np.isnan(ds_regridded.thetao.isel(time=0).drop_vars("time")).load()
    ocean_frac = regridder(ds.wetmask.astype("float64")).fillna(0.0)

    ds_regridded = ds_regridded.drop_vars(["lon_b", "lat_b"])
    ds_regridded = ds_regridded.assign_coords(
        lon=lon,
        lat=lat,
        lon_b=lon_b,
        lat_b=lat_b,
        areacello=new_area,
        x=x,
        y=y,
        wetmask=wetmask,
        ocean_fraction=ocean_frac,
    )
    ds_regridded.attrs = ds.attrs | ds_regridded.attrs

    return ds_regridded


def cmip_bounds_to_xesmf(ds: xr.Dataset, order=None):
    # the order is specific to the way I reorganized vertex order in xmip (if not passed we get the stripes in the regridded output!

    if not all(var in ds.variables for var in ["lon_b", "lat_b"]):
        ds = ds.assign_coords(
            lon_b=cf_xarray.bounds_to_vertices(
                ds.lon_verticies.load(), bounds_dim="vertex", order=order
            ),
            lat_b=cf_xarray.bounds_to_vertices(
                ds.lat_verticies.load(), bounds_dim="vertex", order=order
            ),
        )
    return ds


def test_vertex_order(ds):
    # pick a point in the southern hemisphere to avoid curving nonsense
    p = {"x": slice(20, 22), "y": slice(20, 22)}
    ds_p = ds.isel(**p).squeeze()
    # get rid of all the unneccesary variables
    for var in ds_p.variables:
        if (
            ("lev" in ds_p[var].dims)
            or ("time" in ds_p[var].dims)
            or (var in ["sub_experiment_label", "variant_label"])
        ):
            ds_p = ds_p.drop_vars(var)
    ds_p = cmip_bounds_to_xesmf(
        ds_p, order=None
    )  # woudld be nice if this could automatically get the settings provided to `cmip_bounds_to_xesmf`
    ds_p = ds_p.load().transpose(..., "x", "y", "vertex")
    if (
        not (ds_p.lon_b.diff("x_vertices") > 0).all()
        and (ds_p.lat_b.diff("y_vertices") > 0).all()
    ):
        raise ValueError("Test vertices not strictly monotinically increasing")
