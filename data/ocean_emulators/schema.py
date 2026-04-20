import warnings

from xarrera import CoordsSchema, DataArraySchema, DatasetSchema  # noqa: E402

### Preprocessing Stage
vars_3d = ["so", "thetao", "uo", "vo"]
vars_2d = ["hfds", "tauuo", "tauvo", "zos"]
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
        for k in vars_2d
    }
    | {
        k: DataArraySchema(dtype="float32", dims=["time", "lev", "y", "x"], name=k)
        for k in vars_3d
    }
)

### Input Stage
vars_3d = ["so", "thetao", "uo", "vo"]
vars_2d = ["hfds", "tauuo", "tauvo", "zos"]
# TODO: add flux components + ice variables         "sithick","siconc",
ds_input_coords_schema = CoordsSchema(
    {
        "wetmask": DataArraySchema(dtype=bool, dims=["lev", "y", "x"]),
        "ocean_fraction": DataArraySchema(dtype="float64", dims=["lev", "y", "x"]),
        "lon_b": DataArraySchema(
            dtype="float64", shape=(181, 361), dims=["y_b", "x_b"]
        ),
        "lat_b": DataArraySchema(
            dtype="float64", shape=(181, 361), dims=["y_b", "x_b"]
        ),
        "lon": DataArraySchema(dtype="float64", shape=(180, 360), dims=["y", "x"]),
        "lat": DataArraySchema(dtype="float64", shape=(180, 360), dims=["y", "x"]),
        "areacello": DataArraySchema(
            dtype="float64", shape=(180, 360), dims=["y", "x"]
        ),
        "dz": DataArraySchema(dtype="int64", shape=(19,), dims=["lev"]),
        "lev": DataArraySchema(dtype="float64", shape=(19,), dims=["lev"]),
        "x": DataArraySchema(dtype="float64", shape=(360,), dims=["x"]),
        "y": DataArraySchema(dtype="float64", shape=(180,), dims=["y"]),
        "time": DataArraySchema(
            dims=["time"]
        ),  # can I check that this is actually cftime?
    }
)

# ds_input_attrs_schema = AttrsSchema({"m2lines/ocean-emulators_git_hash":'dummy'})

ds_input_schema = DatasetSchema(
    {
        k: DataArraySchema(dtype="float32", dims=["time", "y", "x"], name=k)
        for k in vars_2d
    }
    | {
        k: DataArraySchema(dtype="float32", dims=["time", "lev", "y", "x"], name=k)
        for k in vars_3d
    },
)

### Prediction
ds_prediction_schema = DatasetSchema()
ds_prediction_coords_schema = CoordsSchema({"time": DataArraySchema(dims=["time"])})
