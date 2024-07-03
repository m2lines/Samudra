from ocean_emulators.preprocessing import standardize_dataset
import pytest
import xarray as xr
from xgcm import Grid
from data import input_data


################################## TODO:rework these tests once the preprocessing is more mature
# def _get_software_version():
#     pass


# def test_rename(input_data):
#     # test that the rename function renames the variables
#     expected_variables = {"thetao", "so", "zos", "uo", "vo", "hfs", "tauuo", "tauvo"}
#     expected_coords = {"wet_mask"}
#     expected_dims = {"time", "lev", "y", "x"}
#     ds_renamed = standardize_dataset(input_data)
#     assert set(ds_renamed.variables) == expected_variables
#     assert set(ds_renamed.dims) == expected_dims
#     assert set(ds_renamed.coords) == expected_coords


# def test_provenance(input_data):
#     # test that the provenance information is added
#     ds_standardized = standardize_dataset(input_data)
#     assert "provenance" in ds_standardized.attrs
#     current_version = (
#         _get_software_version()
#     )  # TODO: how do I get the software version?
#     assert current_version in ds_standardized.attrs["ocean_emulators_version"]
#     # TODO: How do we enter the info where the data actually came from?
#     # TODO: If we are using optional filterig we should track that option in the attributes too!


# ### End to End Tests ###


# def conserve_integral(input_data):
#     # test that the integral of the dataset is conserved
#     # TODO: Is this appropriate for velocities?
#     ds_standardized = standardize_dataset(input_data)
#     # test that the integral of the dataset is conserved
#     grid = Grid(input_data)
#     grid_standardized = Grid(ds_standardized)
#     xr.testing.assert_isclose(
#         grid.integrate(input_data), grid_standardized.integrate(ds_standardized)
#     )


# def conserve_land_fraction(test_dataset):
#     # TODO ask adam what the best way to test this is?
#     ds_standardized = standardize_dataset(test_dataset)
#     land_fraction_expected = (
#         (test_dataset["wet_mask"].isel(lev=0) == 0).sum()
#         / len(test_dataset.x)
#         * len(test_dataset.y)
#     )
#     land_fraction_standardized = (
#         (ds_standardized["wet_mask"].isel(lev=0) == 0).sum()
#         / len(input_data.x)
#         * len(input_data.y)
#     )
#     assert land_fraction_expected == land_fraction_standardized


# def test_filtering():
#     pass


#############################
def test_infer_vertical_cell_extent_missing(input_data):
    ds = input_data
    ds = ds.drop("zos")
    # TODO: Test that we get a message that *only* asks for zos (not the ones that are already on the dataset)
