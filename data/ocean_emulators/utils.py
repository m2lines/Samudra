import subprocess

import numpy as np
import xarray as xr


def get_git_url_hash():
    github_server_url = "https://github.com"
    # Get the repository's remote origin URL
    try:
        repo_origin_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()

        # Extract the repository path from the remote URL
        repository_path = (
            repo_origin_url.replace("github.com/", "")
            .replace("git@github.com:", "")
            .replace(".git", "")
        )

        # Get the current commit SHA
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()

        # Construct the GitHub commit URL
        git_url_hash = f"{github_server_url}/{repository_path}/commit/{commit_sha}"
    except Exception as e:
        print(f"Getting git_url_hash failed with {e}")
        git_url_hash = "none"
        # Output the GitHub commit URL
    return git_url_hash


def _pick_first_element_of_missing_dims(mask: xr.DataArray, data: xr.DataArray):
    missing_dims = [di for di in mask.dims if di not in data.dims]
    if len(missing_dims) == 0:
        return mask
    else:
        return mask.isel({di: 0 for di in missing_dims})


def apply_mask(ds: xr.Dataset, mask: xr.DataArray):
    """Applies mask to same and lower dimensional data"""
    ds_out = xr.Dataset(attrs=ds.attrs)
    for var in ds.data_vars:
        data = ds[var]
        mask_pruned = _pick_first_element_of_missing_dims(mask, data)
        ds_out[var] = data.where(mask_pruned)
    return ds_out


def assert_mask_match(ds: xr.Dataset, mask: xr.DataArray):
    """Assert that nans at a sample time step are consistent with a mask (mask True or 1 indicates not nan)"""
    for var in ds.data_vars:
        data_test = ds[var]
        # make sure that 2d variables are only tested agains 2d wetmask
        mask_test = _pick_first_element_of_missing_dims(mask, data_test)
        if not (data_test.notnull() == mask_test).all():
            raise ValueError(
                f"Wetmask does not match between `ds` and `wetmask` for variable {var}!"
            )


def split_2d_3d(ds: xr.Dataset, depth_dim="lev"):
    ds_2d = xr.Dataset({v: ds[v] for v in ds.data_vars if depth_dim not in ds[v].dims})
    ds_3d = xr.Dataset({v: ds[v] for v in ds.data_vars if depth_dim in ds[v].dims})
    return ds_2d, ds_3d


def _find_index_for_true(da_bool: xr.DataArray, check_dims):
    """Find slices along all dimensions within a boolean array that have any True value"""
    # all_dims = da_bool.dims
    all_dims = [di for di in check_dims if di in da_bool.dims]

    true_found_index = {}
    for dim in all_dims:
        other_dims = [di for di in da_bool.dims if di != dim]
        test = da_bool.any(other_dims).load()
        index = da_bool[dim].isel({dim: test})
        true_found_index[dim] = index.data
    return true_found_index


def ensure_nan_consistency(ds: xr.Dataset, name="None"):
    """Test the consistency of nan values in the dataset across variables and time
    (compared to a reference at time=0).
    """
    ds = ds.to_array()
    ref = ds.isel(time=0)
    # # make sure the ref data has nans in the same places for all variables
    first_var = np.isnan(ref.isel(variable=0))
    all_var = np.isnan(ref).all(["variable"])

    a = first_var != all_var

    # find the index values for true values in b
    index = _find_index_for_true(a, check_dims=list(a.dims))
    print(index)
    if not all(len(v) == 0 for v in index.values()):
        raise ValueError(
            "Found non-matching nan values between variables on the first time step."
        )

    ## make sure that the ref nan pattern is the same as every time step
    b = np.isnan(ref) != np.isnan(ds)

    # find the index values for true values in b
    index = _find_index_for_true(b, check_dims=["variable", "time"])

    # if they are all length 0 all is good, otherwise raise.
    if not all(len(v) == 0 for v in index.values()):
        raise ValueError(
            f"{name}:Found nonmatching nans compared to first time step in the following indexes {index}"
        )
