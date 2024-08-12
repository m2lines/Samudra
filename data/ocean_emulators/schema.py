from xarray_schema import DataArraySchema, DatasetSchema, CoordsSchema

vars_3d = ['so', 'thetao', 'uo', 'vo']
vars_2d = ['hfds', 'tauuo', 'tauvo', 'zos']
ds_processed_coords_schema = CoordsSchema(
    {
        'wetmask':DataArraySchema(dtype=bool, dims=['lev', 'y', 'x']),
        'lon_b':DataArraySchema(dtype='float64', shape=(1081, 1441), dims=['y_b', 'x_b']),
        'lat_b':DataArraySchema(dtype='float64', shape=(1081, 1441), dims=['y_b', 'x_b']),
        'lon':DataArraySchema(dtype='float64', shape=(1080, 1440), dims=['y', 'x']),
        'lat':DataArraySchema(dtype='float64', shape=(1080, 1440), dims=['y', 'x']),
        'angle':DataArraySchema(dtype='float64', shape=(1080, 1440), dims=['y', 'x']),
        'areacello':DataArraySchema(dtype='float32', shape=(1080, 1440), dims=['y', 'x']),
        'dz':DataArraySchema(dtype='int64', shape=(19,), dims=['lev']),
        'lev':DataArraySchema(dtype='float64', shape=(19,), dims=['lev']),
        'x':DataArraySchema(dtype='float64', shape=(1440,), dims=['x']),
        'y':DataArraySchema(dtype='float64', shape=(1080,), dims=['y']),
        'time': DataArraySchema(dims=['time']), # can I check that this is actually cftime?
    })
ds_processed_schema = DatasetSchema(
    {
        k: DataArraySchema(dtype='float32', dims=['time', 'y', 'x'], name=k) for k in vars_2d
    }|{
        k: DataArraySchema(dtype='float32', dims=['time', 'lev', 'y', 'x'], name=k) for k in vars_3d
})