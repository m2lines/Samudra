"""Preprocess arbitrary datasets to standardized naming, grids"""

from xgcm import Grid
import xarray as xr
import numpy as np
import cf_xarray

try:
    import xesmf as xe  # type: ignore
except ImportError:
    xe = None


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
def split_2d_3d(ds: xr.Dataset):
    ds_2d = xr.Dataset({v: ds[v] for v in ds.data_vars if "lev" not in ds[v].dims})
    ds_3d = xr.Dataset({v: ds[v] for v in ds.data_vars if "lev" in ds[v].dims})
    return ds_2d, ds_3d


def find_index_for_true(da_bool: xr.DataArray):
    """Find slices along all dimensions within a boolean array that have any True value"""
    # all_dims = da_bool.dims
    all_dims = [
        di for di in ["variable", "time"] if di in da_bool.dims
    ]  # all variables that should be checked for indexers
    # not necessary to check e.g. x,y, lev here
    true_found_index = {}
    for dim in all_dims:
        other_dims = [di for di in da_bool.dims if di != dim]
        test = da_bool.any(other_dims).load()
        index = da_bool[dim].isel({dim: test})
        true_found_index[dim] = index.data
    return true_found_index


def test_nan_consistency(ds: xr.Dataset, name="None"):
    """Test the consistency of nan values in the dataset across variables and time
    (compared to a reference at time=0)."""
    ds = ds.to_array()
    ref = ds.isel(time=0)
    # # make sure the ref data has nans in the same places for all variables
    a = (np.isnan(ref.isel(variable=0)) != np.isnan(ref)).all(["variable"])

    # find the index values for true values in b
    index = find_index_for_true(a)
    if not all(len(v) == 0 for v in index.values()):
        raise ValueError(
            "Found non-matching nan values between variables on the first time step."
        )

    ## make sure that the ref nan pattern is the same as every time step
    b = np.isnan(ref) != np.isnan(ds)

    # find the index values for true values in b
    index = find_index_for_true(b)

    # if they are all length 0 all is good, otherwise raise.
    if not all(len(v) == 0 for v in index.values()):
        raise ValueError(
            f"{name}:Found nonmatching nans compared to first time step in the following indexes {index}"
        )


def input_data_test_deep(ds_input: xr.Dataset):
    """Expensive tests that compute on the entire dataset"""
    ds_nan_test_2d, ds_nan_test_3d = split_2d_3d(ds_input)
    print("2D consistency check")
    test_nan_consistency(ds_nan_test_2d, "2D nan consistency check")

    print("3D consistency check")
    test_nan_consistency(ds_nan_test_3d, "3D nan consistency check")


def input_data_test(ds_input: xr.Dataset, deep=False):
    """Test function to assert the format of the input dataset.
    If `deep` is True, this will run expensive compuation across the entire dataset."""

    expected_data_vars = [
        "thetao",
        "so",
        "uo",
        "vo",
        "zos",
        "hfds",
        "tauvo",
        "tauuo",
        "sithick",
        "siconc",
    ]
    # add the derived mean/std variables
    expected_data_vars_full = []
    for v in expected_data_vars:
        expected_data_vars_full.extend([f"{v}_mean", f"{v}_std"])

    expected_coords = [
        "areacello",
        "dz",
        "x",
        "y",
        "time",
        "lev",
        "lon",
        "lat",
        "wetmask",
    ]
    if not set(ds_input.coords.keys()) == set(expected_coords):
        raise ValueError(
            f"Expected coords {set(expected_coords)} but found {list(set(ds_input.coords.keys()))}"
        )

    expected_sizes = {"x": 360, "y": 180, "lev": 19}
    for di, s in expected_sizes.items():
        if not ds_input.sizes[di] == s:
            raise ValueError(
                f"Expected size ({s}) for dimension {di}, but got {ds_input.sizes[di]}"
            )

    check_attrs = ["m2lines/ocean-emulators_git_hash"]
    for attr in check_attrs:
        if attr not in ds_input.attrs.keys():
            raise ValueError(f"Could not find {attr} in dataset attributes")

    # asser shape of coordinates
    dims_expected_on_coords = {
        "wetmask": ["x", "y", "lev"],
        "areacello": ["x", "y"],
        "lon": ["x", "y"],
        "lat": ["x", "y"],
        "dz": ["lev"],
    }
    for co, expected_dims in dims_expected_on_coords.items():
        if not set(expected_dims) == set(ds_input[co].dims):
            raise ValueError(
                f"Expected dimensions {set(expected_dims)} on {co}, but got {set(ds_input[co].dims)}"
            )

    if deep:
        input_data_test_deep(ds_input)


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
    ds = ds.assign_coords(
        {
            "lev_outer": lev_outer
            )
        }
    )
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


# test:
# - What about the coordinates after? Are the non-depth ones the same (not weirdly scaled?).


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


def spatially_regrid(
    ds_source: xr.Dataset,
    ds_target: xr.Dataset,
    method: str = "conservative",
    check=False,
) -> xr.Dataset:
    if check:
        for test_ds, name in [
            (ds_source, "source dataset"),
            (ds_target, "target dataset"),
        ]:
            try:
                test_vertex_order(test_ds)
            except ValueError:
                raise ValueError(
                    f"something is wrong with the vertex order of the {name}"
                )
    if xe is None:
        raise ImportError(
            "The spatial regridding requires xesmf. Install using `conda install xesmf`."
        )

    regridder = xe.Regridder(
        cmip_bounds_to_xesmf(ds_source),
        cmip_bounds_to_xesmf(ds_target),
        method,
        ignore_degenerate=True,
        unmapped_to_nan=True,
        periodic=True,
    )
    return regridder(ds_source)
