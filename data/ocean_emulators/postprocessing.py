import xarray as xr


def post_processor(ds: xr.Dataset) -> xr.Dataset:
    """Converts the prediction output to an xarray dataset with the same dimensions/variables as input"""
    da = ds["__xarray_dataarray_variable__"]
    n_lev = 19
    variables = ["uo", "vo", "thetao", "so"]
    slices = [slice(i, i + n_lev) for i in range(0, len(variables) * n_lev, n_lev)]
    var_slices = {k: sl for k, sl in zip(variables, slices)}
    variables = {
        k: da.isel(var=sl).rename({"var": "lev"}) for k, sl in var_slices.items()
    }
    variables["zos"] = da.isel(var=-1).squeeze()
    return xr.Dataset(variables)


def prediction_data_test(ds_prediction: xr.Dataset, ds_input):
    """Testfunction to check post-processed prediction output for format"""
    # TODO: Run the test for the preprocessing data here and warn only if it fails
    # That data should have been checked before training and here we only strictly enforce that things reflect the state of the input data.

    expected_sizes = {"x": 360, "y": 180, "lev": 19}
    given_sizes = ds_prediction.sizes
    compare_dims = list(
        set(list(expected_sizes.keys()) + list(given_sizes.keys())) - set(["time"])
    )
    if any(expected_sizes[dim] != given_sizes[dim] for dim in compare_dims):
        raise ValueError(
            f"Input dataset does not have the right sizes. Expected{expected_sizes}, got {given_sizes}"
        )

    # ensure all dimensions have coordinate values
    dims_without_coords = [
        di for di in ds_prediction.dims if di not in ds_prediction.coords
    ]
    if len(dims_without_coords) > 0:
        raise ValueError(
            f"Found the following dimensions without coordinates: {dims_without_coords}"
        )

    # ensure the attributes are the same on both datasets
    if not ds_prediction.attrs == ds_input.attrs:
        raise ValueError(
            "Prediction and Input datasets do not have matching attributes"
        )

    # TODO: ensure that both arrays have the same coordinates
