#!/usr/bin/env python

# # Preprocessing data for 3D ocean emulation
#
# ## Outstanding issues for generalization
# - [ ] Rotate the velocities according to the grid file in the Arctic
# - [ ] Are the CMIP velocities really already on the tracer grid? [Issue](https://github.com/m2lines/ocean_emulators/issues/19#issue-2362588496)
# - TODO: Figure out the masking properly (https://xesmf.readthedocs.io/en/stable/user_api.html see `skipna` etc) -> See the antarctic 0 fields
# - [x] Add area and dz to the output dataset (this is tricky since it is affected by masking/spatial regridding)
# - [x] replace x/y with nominal lon/lat
# - [ ] Preserve units/long name for all variables
# - [ ] Come up with a more robust solution to the vertex order issue...for now I am building in a test. This is GNARLY! For now ill just run the permutation each time and then reorder I guess.
# - [ ] Reduce precision

# In[9]:


# ## relying on all the dangling branches 🙈
# !pip install git+https://github.com/jbusecke/xgcm.git@conservative_transform_w_nans


# In[10]:


# !pip install git+https://github.com/ocean-transport/scale-aware-air-sea@kwargs-for-open-close


# In[11]:


# !pip install -e /home/jovyan/PROJECTS/ocean_emulators/


# In[12]:


# !pip install git+https://github.com/jbusecke/xMIP.git


# In[1]:


import intake
import numpy as np
import xarray as xr
from scale_aware_air_sea.utils import to_zarr_split
from xmip.postprocessing import (
    _drop_duplicate_grid_labels,  # FIXME: Make this part of the public API (started https://github.com/jbusecke/xMIP/pull/356)
)
from xmip.preprocessing import combined_preprocessing

# In[1]:
from ocean_emulators.preprocessing import (
    infer_vertical_cell_extent,
    input_data_test,
    spatially_regrid,
    vertical_regrid,
)

to_ddict_kwargs = dict(aggregate=False, preprocess=combined_preprocessing)


# In[2]:


import os

os.environ["JUPYTER_IMAGE_SPEC"]


# In[3]:


from distributed import Client

client = Client()
client


# In[5]:


# From https://github.com/m2lines/ocean_emulators/issues/17
target_vertical_levels = np.array(
    [
        0,
        5,
        15,
        30,
        50,
        80,
        130,
        200,
        300,
        450,
        650,
        900,
        1200,
        1600,
        2100,
        2700,
        3500,
        4500,
        5500,
        6500,
    ]
)

source_ids = ["GFDL-CM4", "CESM2"]


# In[6]:


# uncomment/comment lines to swap catalogs
url = "https://storage.googleapis.com/cmip6/cmip6-pgf-ingestion-test/catalog/catalog.json"  # Only stores that pass current tests
col = intake.open_esm_datastore(url)


# In[7]:


# Quick workaround for https://github.com/leap-stc/cmip6-leap-feedstock/issues/146
url = "https://storage.googleapis.com/cmip6/cmip6-pgf-ingestion-test/catalog/catalog_noqc.json"  # Only stores that fail current tests
col_no_qc = intake.open_esm_datastore(url)


# In[8]:


cat_ice = col.search(
    source_id=source_ids,
    experiment_id="piControl",
    table_id="SImon",
    grid_label="gn",
    variable_id=["sithick", "siconc"],
)  # TODO: u and v are only on gn grid
ddict_ice = cat_ice.to_dataset_dict(**to_ddict_kwargs)


# In[9]:


cat_ocean = col.search(
    source_id=source_ids,
    experiment_id="piControl",
    table_id="Omon",
    variable_id=["so", "thetao", "uo", "vo", "zos", "hfds", "tauvo", "tauuo"],
)  # TODO: u and v are only on gn grid
ddict_ocean = cat_ocean.to_dataset_dict(**to_ddict_kwargs)

# some models (e.g. CM4 have zos not on the native grid. I think it is better to interpolate this, since we need it to infer the vertical thickness

ddict_ocean = _drop_duplicate_grid_labels(
    ddict_ocean, "gn"
)  # this is redundant to the call to 'interpolate_grid_label' below...but for now this should be fine
ddict_ice = _drop_duplicate_grid_labels(ddict_ice, "gn")
# include ice, since I think the gn is the ocean grid
ddict_ocean = ddict_ocean | ddict_ice


# ## Rechunk and save out/reload again to harmonize chunking structure and avoid rechunking on the fly
#

# In[10]:


# fs = gcsfs.GCSFileSystem()
# fs.rm("gs://leap-scratch/jbusecke/ocean-emulators/temp/ddict_ocean_GFDL-CM4.piControl.SImon.r1i1p1f1.siconc.siconc.zarr", recursive=True);


# In[11]:


import gcsfs
from scale_aware_air_sea.utils import maybe_save_and_reload

prefix = "gs://leap-scratch/jbusecke/ocean-emulators/temp"

# fs = gcsfs.GCSFileSystem()
# fs.rm(prefix, recursive=True)


def rechunk(ds: xr.Dataset) -> xr.Dataset:
    target_chunks = {"time": 1, "x": -1, "y": -1, "lev": -1, "bnds": 2, "vertex": 4}
    valid_target_chunks = {k: c for k, c in target_chunks.items() if k in ds.dims}
    ds = ds.chunk(valid_target_chunks)
    # this is dumb, but lets get this over with
    ds.encoding = {}
    for v in ds.variables:
        ds[v].encoding = {}
    return ds


ddict_ocean_reloaded = {
    k: maybe_save_and_reload(
        rechunk(ds),
        f"{prefix}/ddict_ocean_{k}.zarr",
        split=True,
        to_zarr_split_kwargs=dict(split_interval=600 if "lev" in ds.dims else 3000),
    )
    for k, ds in ddict_ocean.items()
}

# FIXME: Some coords are mislabeled as data_vars
ddict_reloaded = {
    k: ds.set_coords([v for v in ds.data_vars if v != ds.attrs["variable_id"]])
    for k, ds in ddict_ocean_reloaded.items()
}


# ## Fixing the vertex order to avoid those stripes (This is agony!)

# In[16]:


import itertools

from ocean_emulators.preprocessing import test_vertex_order


def reorder_vertex(ds, new_order):
    ds_wo_vertex = ds.drop_vars([va for va in ds.variables if "vertex" in ds[va].dims])
    ds_w_vertex = ds.drop_vars(
        [va for va in ds.variables if "vertex" not in ds[va].dims]
    )
    ds_w_vertex_reordered = xr.concat(
        [ds_w_vertex.isel(vertex=i) for i in new_order], dim="vertex"
    )
    return xr.merge([ds_w_vertex_reordered, ds_wo_vertex])


def get_order(ds):
    order = [0, 1, 2, 3]
    all_orders = itertools.permutations(order, len(order))
    for new_order in all_orders:
        ds_reordered = reorder_vertex(ds, new_order)
        try:
            test_vertex_order(ds_reordered)
            print(f"{new_order=} worked!")
            return new_order
        except:
            pass


from xmip.utils import cmip6_dataset_id


def test_and_reorder_vertex(ds):
    """This is an expensive check that tries every possible order of the vertex and confirms
    that we get strictly monontonic lon_b/lat_b coordinates for a test point.
    """
    new_order = get_order(ds)
    if new_order is None:
        # drop them, maybe another one works better? This is a nightmare TBH.
        ds_out = ds.drop_vars([va for va in ds.variables if "vertex" in ds[va].dims])
        print(f"Unable to find a vertex order for {cmip6_dataset_id(ds)}")
        # raise ValueError(f"Unable to find a vertex order for {cmip6_dataset_id(ds)}")
    else:
        print(f"Changing vertex order for {cmip6_dataset_id(ds)}")
        ds_out = reorder_vertex(ds, new_order)
    return ds_out


# In[17]:


# I guess ill run it over every goddamn dataset, since I have nothing better to do lol.
ddict_reloaded_fix_vertex = {
    k: test_and_reorder_vertex(ds) if "vertex" in ds.dims else ds
    for k, ds in ddict_reloaded.items()
}


# ## Interpolate the SSH onto the native grid
# Somehow the source dict still gets modified in place...not great

# In[18]:


### What I want here is to Group by source_id -> modify one dataset, but not combine -> recombine to dataset dict. This violates the assumptions behind `combine_dataset`
### I could use `interp_grid_label` but that leads to issues since e.g. the CESM data is on a staggered grid and needs to be interpolated with xgcm
### I could maybe modify the 'create staggered grid' thing in a way, but maybe for now a manual solution is easier to handle.
### TODO: Bring this modification upstream to xmip with good test coverage
from xmip.postprocessing import (
    EXACT_ATTRS,
    _key_from_attrs,
    _match_datasets,
    _prune_match_attrs_to_available,
    cmip6_dataset_id,
)


def combine_datasets_new(
    ds_dict,
    combine_func,
    combine_func_args=(),
    combine_func_kwargs={},
    match_attrs=EXACT_ATTRS,
):
    """General combination function to combine datasets within a dictionary according to their matching attributes.
    This function provides maximum flexibility, but can be somewhat complex to set up. The postprocessing module provided several
    convenience wrappers like `merge_variables`, `concat_members`, etc.

    Parameters
    ----------
    ds_dict : [type]
        [description]
    combine_func : [type]
        [description]
    combine_func_args : tuple, optional
        [description], by default ()
    combine_func_kwargs : dict, optional
        [description], by default {}
    match_attrs : [type], optional
        [description], by default exact_attrs

    Returns:
    -------
    [type]
        [description]
    """
    # make a copy of the input dict, so it is not modified outside of the function
    # ? Not sure this is always desired.
    ds_dict = {k: v for k, v in ds_dict.items()}
    ds_dict_combined = {}

    # Check each of the matching attributes. If attr is not present in any of
    # the input datasets drop from match_attrs and warn
    match_attrs = _prune_match_attrs_to_available(match_attrs, ds_dict)

    while len(ds_dict) > 0:
        # The order does not matter here, so just choose the first key
        k = list(ds_dict.keys())[0]
        ds = ds_dict.pop(k)
        matched_datasets = _match_datasets(ds, ds_dict, match_attrs, pop=True)

        # for now Ill hardcode the merging. I have been thinking if I should
        # just pass a general `func` to deal with a list of datasets. The user could pass custom stuff
        # And I think that way I can generalize all/most of the `member` level combine functions.
        # Well for now, lets do it the manual way.
        # try:
        combined = combine_func(
            matched_datasets, *combine_func_args, **combine_func_kwargs
        )
        if isinstance(combined, xr.Dataset):
            # create new dict key
            ds_combined = combined
            new_k = _key_from_attrs(
                ds_combined, match_attrs, sep="."
            )  # TODO: Does this still work as in the old case?
            ds_dict_combined[new_k] = ds_combined
        elif isinstance(combined, list):
            print("returning uncombined list of datasets")
            for ds_combined in combined:
                new_k = _key_from_attrs(
                    ds_combined, EXACT_ATTRS, sep="."
                )  # NOTE: Need to construct the "FULL" key here, otherwise keys are not unique!
                ds_dict_combined[new_k] = ds_combined
        # except Exception as e:
        #     warnings.warn(f"{cmip6_dataset_id(ds)} failed to combine with :{e}")

    return ds_dict_combined


def _interp_only_zos(ds_list: list[xr.Dataset]) -> list[xr.Dataset]:
    # Check that this only affects the zos field (otherwise I might have to rethink this and derive thicknesses separately?)
    check = [
        ds.attrs["variable_id"] for ds in ds_list if ds.attrs["grid_label"] != "gn"
    ]
    assert all("mon" in ds.attrs["table_id"] for ds in ds_list)
    if len(check) == 0:
        return ds_list
    else:
        assert set(["zos"]) == set(check)
        print(f"Regridding zos onto thetao for {cmip6_dataset_id(ds_list[0])}")
        # Interpolate the zos dataset, and modify metadata
        other_datasets = [
            ds for ds in ds_list if ds.attrs["variable_id"] not in ["zos", "thetao"]
        ]
        [ds_zos] = [ds for ds in ds_list if ds.attrs["variable_id"] == "zos"]
        [ds_thetao] = [ds for ds in ds_list if ds.attrs["variable_id"] == "thetao"]
        ds_zos_regridded = spatially_regrid(
            ds_zos, ds_thetao
        )  # bilinear causes the line again
        attrs = ds_zos.attrs
        # TODO somehow indicate that this was regridded
        attrs["grid_label"] = ds_thetao.attrs["grid_label"]
        ds_zos_regridded.attrs = attrs
        return other_datasets + [ds_zos_regridded, ds_thetao]


ddict_all = combine_datasets_new(
    ddict_reloaded_fix_vertex,
    _interp_only_zos,
    match_attrs=["source_id", "experiment_id", "variant_label"],
)


# In[19]:


ddict_all["GFDL-CM4.gn.piControl.Omon.r1i1p1f1.zos"].isel(time=0).zos.roll(x=700).plot()


# ## Add grid info

# In[20]:


cat_grid_fixed = col_no_qc.search(
    source_id=source_ids,
    experiment_id="piControl",
    variable_id=["deptho", "thkcello", "areacello", "volcello"],
    grid_label=["gr", "gn"],
)
ddict_grid = cat_grid_fixed.to_dataset_dict(**to_ddict_kwargs)


# In[21]:


from xmip.postprocessing import match_metrics


def match_3d_metric_wrapper(
    ddict: list[xr.Dataset], ddict_metric: list[xr.Dataset]
) -> list[xr.Dataset]:
    ddict_merged_out = {}

    # split into 3d and 2d Avoids [This](https://github.com/jbusecke/xMIP/issues/355) came up when I tried to add 3d metrics to 2d fields
    ddict_3d = {
        k: ds for k, ds in ddict.items() if "lev" in ds[ds.attrs["variable_id"]].dims
    }
    ddict_2d = {
        k: ds
        for k, ds in ddict.items()
        if "lev" not in ds[ds.attrs["variable_id"]].dims
    }

    # I do absolutely not understand why this works with more strict match_attrs (it fails when you take the defaults)
    ddict_w_metrics = match_metrics(
        ddict_3d, ddict_grid, ["deptho", "thkcello", "areacello", "volcello"]
    )

    # merge 2d variables back in
    for ddict_parse in [ddict_w_metrics, ddict_2d]:
        for k, ds in ddict_parse.items():
            ddict_merged_out[k] = ds

    return ddict_merged_out


ddict_matched_metrics = match_3d_metric_wrapper(ddict_all, ddict_grid)


# ## 👉 👉 👉 (For tomorrow) Interpolate velocities onto tracer grid before any of the vertical regridding
#
# TODO: check the implications for the wetmask here
# FIXME: This needs more upstream work, since all the values for CM4 are on the center coordinate, pass through here for now
#
# What I generally need here is a three step process
#
# - Group datasets that should be combined with `combine_datasets`
# - Find ref dataset, and detect lon/lat shifts for all other datasets
# - parse datasets something like this: combine(ds_ref, [(ds_a, {'X':'left', 'Y':'center'}), (ds_b, {'X':'center', 'Y':'left'})])
#
# Then i want to get a single datasets that has x, x_left, y, y_left coordinates and the data arrays from each one of them located appropriately within the output dataset
# Also make sure that dataset can be immediately be metadata parsed for xgcm
#

# In[22]:


# # This is a rewrite from xmip and needs to be upstreamed
# # from xmip.grids import combine_staggered_grid does not work anymore with modern xgcm version

# # What is new:
# # - no more attempting to recreate metrics, if they are on the datasets, then parse them, otherwise dont do any calculations
# # - no way to overwrite the inferred shifts! The only way to manually change the position is to manipulate the metadata!

# # Questions: What about the depth? Should I parse the outer coordinates here?

# t = ddict_ocean['CMIP.NOAA-GFDL.GFDL-CM4.piControl.r1i1p1f1.Omon.thetao.gn.none.r1i1p1f1.v20180701.gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Omon/thetao/gn/v20180701/']
# u = ddict_ocean['CMIP.NOAA-GFDL.GFDL-CM4.piControl.r1i1p1f1.Omon.uo.gn.none.r1i1p1f1.v20180701.gs://cmip6/cmip6-pgf-ingestion-test/zarr_stores/9555660527_1/CMIP6.CMIP.NOAA-GFDL.GFDL-CM4.piControl.r1i1p1f1.Omon.uo.gn.v20180701.zarr']
# v = ddict_ocean['CMIP.NOAA-GFDL.GFDL-CM4.piControl.r1i1p1f1.Omon.vo.gn.none.r1i1p1f1.v20180701.gs://cmip6/cmip6-pgf-ingestion-test/zarr_stores/9555660527_1/CMIP6.CMIP.NOAA-GFDL.GFDL-CM4.piControl.r1i1p1f1.Omon.vo.gn.v20180701.zarr']


# ds_base = t
# ds_other = [u,v]
# # just for testing (needs to be done with a lon/lat comparison in the full workflow)
# u_modified = u.copy()
# u_modified.x.attrs['c_grid_axis_shift'] = 1

# v_modified = v.copy()
# v_modified.y.attrs['c_grid_axis_shift'] = 1
# ds_other = [u_modified, v_modified]

# from xgcm.metadata_parsers import parse_metadata
# from typing import Dict, Tuple, Union, List

# def combine_staggered_grid(
#     ds_base, other_ds=None,
#     default_shift:Union[Dict[str, Tuple[str, str]], Tuple[str, str]]=('center', 'left'),
#     exclude_axes:List[str]=['T'],
# ):
#     """

#     """
#     # figure out the shifts from base, if no other position than center is detected use some default?
#     # Only datasets with parseable metadata are accepted (any manual modification needs to be made before passing the inputs.
#     base_w_metadata = parse_metadata(ds_base)
#     other_w_metadata = [parse_metadata(ds) for ds in other_ds]
#     #TODO: I am not sure what this would do if it failed. Need to check with a different source_id
#     # TODO: Raise a useful error message that displays the error and the iid of the dataset

#     # collect unique axes:
#     all_metadata = [m for ds,m in [base_w_metadata]+other_w_metadata]
#     #TODO test that there will always be one value per axis?

#     axes_w_positions = {}
#     for md in all_metadata:
#         for axis, p_dict in md['coords'].items():
#             print(axis, p_dict)
#             for dim, pos in p_dict.items():
#                 if not axis in exclude_axes:
#                     if axis not in axes_w_positions.keys():
#                         axes_w_positions[axis] = []
#                     axes_w_positions[axis].extend(positions.keys())

#     # make sure positions are unique
#     axes_w_positions = {axis:list(set(positions)) for axis,positions in axes_w_positions.items()}

#     # Maybe expand default shifts to each axis
#     if not isinstance(default_shift, dict):
#         default_shift = {ax:default_shift for ax in axes_w_positions.keys()}

#     for ax, pos in axes_w_positions.items():
#         if len(pos) > 2:
#             raise ValueError(f"Cannot create a staggered xgcm grid with more than 2 positions per axis. Got {pos} for axis {ax}")
#         elif len(pos) == 1:
#             fill = default_shift[ax]
#             fill = [f for f in fill if f not in pos][0]
#             print(f"Only found a single axis position for axis {ax}. Filling in default shift value {fill}")
#             axes_w_positions[ax].append(fill)

#     # create the new dimensions according to the shift (retain the naming of ref dataset, and add suffix for all others)

#     ds_base, metadata_base = base_w_metadata
#     for ax, pos in axes_w_positions.items():
#         # determine which position is on the ref dataset
#         ref_pos = metadata_base['coords'][ax].keys()[0]
#         # determine the name for the new dimension and add it to the dataset
#         # TODO: Should this have coordinate values?
#         new_pos = [pos for pos in axes_w_positions[ax] if pos != ref_pos][0]
#         print(new_pos)
#         new_dimension = 1
#         print(new_dimension)
#     #then loop over "other" datasets and rename+add them so they sit on the appropriate positions of the output
#     return axes_w_positions, metadata_base

# combine_staggered_grid(ds_base, ds_other)


# ## 🚨 This is certainly wrong for CESM, but for now I am just merging all the variables and coordinates brute force.

# In[23]:


from xmip.postprocessing import combine_datasets


def _custom_merge(ds_list: list[xr.Dataset]) -> xr.Dataset:
    """A custom merge that assumes that datasets can be merged along variables and table_ids exactly"""
    return xr.merge(ds_list, compat="override", join="override")


def merge_variable_id_table_id(ds_dict):
    return combine_datasets(
        ds_dict,
        _custom_merge,
        match_attrs=["source_id", "grid_label", "experiment_id", "variant_label"],
    )


ddict_combined_on_tracer = merge_variable_id_table_id(ddict_matched_metrics)


# ## A rough patch for CESM: Replacing recomputing thkcello from volcello
# Test this thoroughly!

# In[24]:


def patch_thkcello(ds: xr.Dataset):
    if "thkcello" not in ds and "volcello" in ds:
        ds = ds.assign_coords(thkcello=ds.volcello / ds.areacello)
        ds = ds.drop(["volcello"])
    return ds


ddict_combined_on_tracer = {
    k: patch_thkcello(ds) for k, ds in ddict_combined_on_tracer.items()
}


# ## Combine and prep for vertical regridding

# In[25]:


ddict_w_dz = {
    k: infer_vertical_cell_extent(ds)
    for k, ds in ddict_combined_on_tracer.items()
    if "thkcello" in ds
}


# In[26]:


ddict_vert_regridded = {
    k: vertical_regrid(ds, target_vertical_levels) for k, ds in ddict_w_dz.items()
}


# In[ ]:


# In[29]:


test = ddict_ocean_w_dz["GFDL-CM4.gn.piControl.r1i1p1f1"]


# In[156]:


test.isel(time=0, x=90).dz.squeeze().plot()


# Ah nice this actually seems to account for partial bottom cells!

# In[82]:


ddict_vert_regridded["CESM2.gn.piControl.r1i1p1f1"].zos


# In[83]:


k = "CESM2.gn.piControl.r1i1p1f1"

test = ddict_w_dz[k].isel(time=slice(0, 2))
test_regridded = ddict_vert_regridded[k].isel(time=slice(0, 2)).astype(np.float32)


# In[84]:


test.isel(time=0, y=90).thetao.squeeze().plot(x="x")


# In[64]:


test_regridded.isel(time=0, y=90).thetao.squeeze().plot(x="x")


# In[162]:


# int_test = (test.astype(np.float64)*test.dz).sum('lev').load()
# int_test_regridded = (test_regridded*test_regridded.dz).sum('lev').load()
# transpose_order=set(int_test.sizes.keys())
# int_test_regridded = int_test_regridded.transpose(*transpose_order)
# int_test = int_test.transpose(*transpose_order)
# still failing, but can prob get it to go with less tight tolerances...
# xr.testing.assert_allclose(int_test, int_test_regridded, rtol=1e-4, atol=1e-4) # depends on dtype of the output TODO: for more cmip models ill have to deal with this! I probably just have to cast things a bit smarter. Visually these have very small differences.


# In[163]:


# int_test.thetao.isel(time=0).plot(x='x')


# In[164]:


# int_test_regridded.thetao.isel(time=0).plot(x='x')


# In[165]:


# (int_test-int_test_regridded).thetao.isel(time=0).plot(robust=True, x='x')


# ## Regrid and combine all fields (if not all are on the same grid already)

# In[37]:


import xesmf as xe


def regrid_and_merge_regular_grid(ds_list: list[xr.Dataset]) -> xr.Dataset:
    target_ds = xe.util.grid_global(
        1,
        1,
        lon1=360,
    )
    datasets = []
    for ds in ds_list:
        # drop the precision before the regridding?
        ds = ds.astype(np.float32)
        ds_regridded = spatially_regrid(ds, target_ds, "conservative")
        ds_regridded.attrs = ds.attrs
        datasets.append(ds_regridded)
    return xr.merge(datasets, combine_attrs="drop_conflicts")


ddict_final = combine_datasets(
    ddict_vert_regridded,
    regrid_and_merge_regular_grid,
    match_attrs=["source_id", "experiment_id", "variant_label"],
)


# ## Uniform masking (Shortcut, needs revisiting)
#
# Get a uniform nan mask over all models (this might change in the future, so i need a better solution eventually.

# In[38]:


def get_model_mask(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    data_vars = [
        v for v in ds.data_vars if "si" not in v
    ]  # exclude the sea ice variables here
    vars_3d = [v for v in data_vars if "lev" in ds[v].dims]
    vars_2d = [v for v in data_vars if "lev" not in ds[v].dims]
    masks_3d = [np.isnan(ds[var].isel(time=0)) for var in vars_3d]
    masks_2d = [np.isnan(ds[var].isel(time=0)) for var in vars_2d] + [
        np.isnan(ds[var].isel(time=0, lev=0)) for var in vars_3d
    ]
    combined_mask_3d = sum(masks_3d).load()
    combined_mask_2d = sum(masks_2d).load()
    return combined_mask_3d, combined_mask_2d


def apply_masks(
    ds: xr.Dataset, mask_3d: xr.DataArray, mask_2d: xr.DataArray
) -> xr.Dataset:
    ds_masked = xr.Dataset(attrs=ds.attrs)
    for v in ds.data_vars:
        if "lev" in ds[v].dims:
            ds_masked[v] = ds[v].where(mask_3d == True)
        else:
            ds_masked[v] = ds[v].where(mask_2d == True)
    return ds_masked


mask_3d, mask_2d = zip(*[get_model_mask(ds) for ds in ddict_final.values()])
# combined masks should mask everything that is not ocean across all models
mask_3d_combo = sum(mask_3d) == 0
mask_2d_combo = sum(mask_2d) == 0

ddict_masked = {
    k: apply_masks(ds, mask_3d_combo, mask_2d_combo) for k, ds in ddict_final.items()
}


# ## Add area and Clean Up
# For now calculate the area manually

# In[39]:


ddict_masked["CESM2.piControl.r1i1p1f1"]


# In[40]:


# add reproducible attributes
import subprocess

import numpy as np


def get_git_hash():
    try:
        # Run the git command to get the current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Check if the command was successful
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise Exception(f"Git command failed with error: {result.stderr}")
    except Exception as e:
        return str(e)


def calc_area(ds_dict: dict[str, xr.Dataset]):
    # pick any dataset, they should all be on the same grid at this point
    ds = ds_dict[list(ds_dict.keys())[0]]  #
    # 1 deg is roughly 111 km
    dy = 111000
    dx = np.cos(np.deg2rad(ds.lat))
    area = dy * dx
    return {k: ds.assign_coords({"areacello": area}) for k, ds in ds_dict.items()}


ddict_area = calc_area(ddict_masked)


def clean_up(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.squeeze()
    # remove coordinates that are documented in the attrs
    remove_coords = ["sub_experiment_id", "variant_label"]
    for co in remove_coords:
        assert co in ds.attrs.keys()
        ds = ds.drop_vars(co)

    # add lon/lat values to x/y for easier plotting
    ds = ds.assign_coords(x=ds.lon.isel(y=0), y=ds.lat.isel(x=0))

    # bake in git hash
    ds.attrs["m2lines/ocean-emulators_git_hash"] = get_git_hash()
    return ds


ddict_clean = {k: clean_up(ds) for k, ds in ddict_area.items()}


# ## Quick Test output (mostly checking for that regridding 'seam')

# In[41]:


ddict_test = ddict_clean
for k, ds in ddict_test.items():
    input_data_test(ds)


# In[42]:


# produce some qc plots (TODO: make this a nice function like the evaluation)
import matplotlib.pyplot as plt

for k, ds in ddict_test.items():
    plt.figure(figsize=[24, 6])
    for i, v in enumerate(ds.data_vars):
        plt.subplot(2, 5, i + 1)  # , sharex=True, sharey=True)
        ds[v].isel(time=-1, lev=0, missing_dims="ignore").roll(x=120).plot()
        plt.title(f"{k}--{v}")
    plt.show()


# ## Time to Write them out
#
# - For now still writing compressed, expecting Surya to rewrite the store.

# In[43]:


import os

fs = gcsfs.GCSFileSystem()
prefix = "gs://leap-persistent/jbusecke/ocean-emulators"


# In[46]:


# fs.rm(prefix, recursive=True);


# In[47]:


for k, ds_out in ddict_clean.items():
    store_name = f"CMIP6_{k}_v0.1.zarr"
    path = os.path.join(prefix, store_name)
    mapper = fs.get_mapper(path)
    to_zarr_split(ds_out, mapper, split_interval=500)


# In[4]:


paths = [
    "gs://leap-persistent/jbusecke/ocean-emulators/CMIP6_GFDL-CM4.piControl.r1i1p1f1_v0.1.zarr",
    "gs://leap-persistent/jbusecke/ocean-emulators/CMIP6_CESM2.piControl.r1i1p1f1_v0.1.zarr",
]


# In[7]:


import matplotlib.pyplot as plt
import xarray as xr
from xarrayutils.plotting import linear_piecewise_scale
from xmip.utils import cmip6_dataset_id

get_ipython().run_line_magic("matplotlib", "inline")


def qc_plots(ds: xr.Dataset):
    ## plot maps
    fig, axarr = plt.subplots(ncols=2, nrows=5, figsize=[15, 22])
    for var, ax in zip(ds.data_vars, axarr.flat):
        da = ds[var].isel(time=0, lev=0, missing_dims="ignore").load()
        kwargs = {"x": "x"}
        da.roll(x=30).plot(ax=ax, **kwargs)
        ax.set_title("Surface Snapshot")
    fig.suptitle(f"{cmip6_dataset_id(ds)}")
    plt.show()

    ## plot simple (non-weighted averages) over time (and potentially depth)
    fig, axarr = plt.subplots(ncols=2, nrows=5, figsize=[15, 22])
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
    fig.suptitle(f"{cmip6_dataset_id(ds)}")
    plt.show()

    ### show stdv over time averaged over longitudes
    fig, axarr = plt.subplots(ncols=2, nrows=5, figsize=[15, 22])
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
    fig.suptitle(f"{cmip6_dataset_id(ds)}")

    plt.show()


# In[10]:


ds


# In[9]:


# from ocean_emulators.plotting import qc_plots
ddict_reloaded = {}
for path in paths:
    ds = xr.open_dataset(path, engine="zarr", chunks={})
    print(cmip6_dataset_id(ds))
    qc_plots(ds)


# In[ ]:
