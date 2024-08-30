from .gfdl_om4 import om4_preprocessing


def cm4_preprocessing(om_zarr_path, sis_zarr_path, nc_grid_path, nc_mosaic_path):
    """CM4 specific preprocessing

    Args:
        om_zarr_path (str): path to the ocean model output
        sis_zarr_path (str): path to the sea ice model output
        nc_grid_path (str): path to the grid file
        nc_mosaic_path (str): path to the mosaic file
    """
    ds_om = om4_preprocessing(
        zarr_data_path=om_zarr_path,
        nc_grid_path=nc_grid_path,
        nc_mosaic_path=nc_mosaic_path,
        vertical_dim="z_l",
    )

    return ds_om
