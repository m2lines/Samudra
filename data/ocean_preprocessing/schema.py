# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from xarrera import CoordsSchema, DataArraySchema, DatasetSchema  # noqa: E402

OM4_3D_VARS = ("so", "thetao", "uo", "vo")
OM4_REQUIRED_2D_VARS = ("hfds", "tauuo", "tauvo", "zos")
# Legacy averaged OM4 and CM4 datasets do not contain freshwater flux. Keep wfo
# optional in shared validators; the OM4 publication CLI enforces it separately.
OM4_OPTIONAL_2D_VARS = ("wfo",)


### Preprocessing Stage
ds_processed_coords_schema = CoordsSchema(
    {
        "wetmask": DataArraySchema(dtype=bool, dims=["lev", "y", "x"]),
        "lon_b": DataArraySchema(
            dtype="float64", shape=(1081, 1441), dims=["y_b", "x_b"]
        ),
        "lat_b": DataArraySchema(
            dtype="float64", shape=(1081, 1441), dims=["y_b", "x_b"]
        ),
        "lon": DataArraySchema(dtype="float64", shape=(1080, 1440), dims=["y", "x"]),
        "lat": DataArraySchema(dtype="float64", shape=(1080, 1440), dims=["y", "x"]),
        "angle": DataArraySchema(dtype="float64", shape=(1080, 1440), dims=["y", "x"]),
        "areacello": DataArraySchema(
            dtype="float64", shape=(1080, 1440), dims=["y", "x"]
        ),
        "dz": DataArraySchema(dtype="float64", shape=(19,), dims=["lev"]),
        "lev": DataArraySchema(dtype="float64", shape=(19,), dims=["lev"]),
        "ilev": DataArraySchema(dtype="float64", shape=(20,), dims=["ilev"]),
        "x": DataArraySchema(dtype="float64", shape=(1440,), dims=["x"]),
        "y": DataArraySchema(dtype="float64", shape=(1080,), dims=["y"]),
        "time": DataArraySchema(
            dims=["time"]
        ),  # can I check that this is actually cftime?
    }
)
ds_processed_schema = DatasetSchema(
    {
        k: DataArraySchema(dtype="float32", dims=["time", "y", "x"], name=k)
        for k in OM4_REQUIRED_2D_VARS
    }
    | {
        k: DataArraySchema(dtype="float32", dims=["time", "lev", "y", "x"], name=k)
        for k in OM4_3D_VARS
    }
)

### Prediction
ds_prediction_schema = DatasetSchema()
ds_prediction_coords_schema = CoordsSchema({"time": DataArraySchema(dims=["time"])})
