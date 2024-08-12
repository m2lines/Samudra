import pytest
import xarray as xr
import numpy as np
from ocean_emulators.preprocessing import rotate_vectors
from tests.data import input_data  # noqa # Might want to put these in conftest.py (see https://stackoverflow.com/questions/73191533/using-conftest-py-vs-importing-fixtures-from-dedicate-modules)

#############################
def test_infer_vertical_cell_extent_missing(input_data):
    ds = input_data
    ds = ds.drop("zos")
    # TODO: Test that we get a message that *only* asks for zos (not the ones that are already on the dataset)


def test_rotate_vectors_angle_wrong_dims():
    # create dummy data
    dims = {'x':3, 'y':2, 'time':5}
    sizes = list(dims.values())
    u = xr.DataArray(np.random.rand(*sizes), dims=dims.keys())
    v = xr.DataArray(np.random.rand(*sizes), dims=dims.keys())
    # 3d angle should raise error
    angle = xr.DataArray(np.random.rand(*sizes), dims=list(dims.keys()))
    print(angle)
    with pytest.raises(ValueError, match="Expected only two dimensions on `angle`. Got*"):
        rotate_vectors(u,v,angle)

def test_rotate_vectors_non_matching_pos():
    # create dummy data
    dims = {'x':3, 'y':2, 'time':5}
    sizes = list(dims.values())
    u = xr.DataArray(np.random.rand(*sizes), dims=dims.keys())
    v = xr.DataArray(np.random.rand(*sizes), dims=dims.keys())
    v = v.rename({'x':'other_x'})
    # 3d angle should raise error
    angle = xr.DataArray(np.random.rand(*sizes[0:2]), dims=list(dims.keys())[0:2])
    with pytest.raises(ValueError, match="`u` and `v` need to be on the same grid position as `angle`."):
        rotate_vectors(u,v,angle)
    
    
@pytest.mark.parametrize('dims',[{'x':2,'y':3}, {'xx':3, 'yy':2}, {'x':3, 'y':2, 'time':5}])
def test_rotate_vectors(dims):
    sizes = list(dims.values())
    names = list(dims.keys())
    # create dummy data
    u = xr.DataArray(np.random.rand(*sizes), dims=names)-0.5
    v = xr.DataArray(np.random.rand(*sizes), dims=names)-0.5
    angle = (xr.DataArray(np.random.rand(*sizes[0:2]), dims=names[0:2])-0.5)*180
    
    u_rotated, v_rotated = rotate_vectors(u,v,angle)
    
    # assert the shape is unaltered
    assert u.shape == u_rotated.shape
    assert u.dims == u_rotated.dims

    assert v.shape == v_rotated.shape
    assert v.dims == v_rotated.dims

    # check every position against expectation
    # but only for the first time step
    if 'time' in u:
        u = u.isel(time=0)
        v = v.isel(time=0)
        for i in len(u[names[0]]):
            for j in len(u[names[1]]):
                raw_u = u.isel({names[0]:i, names[1]:j}).data
                raw_v = v.isel({names[0]:i, names[1]:j}).data
                raw_a = angle.isel({names[0]:i, names[1]:j}).data
                expected_u = u_rotated.isel({names[0]:i, names[1]:j}).data
                expected_v = v_rotated.isel({names[0]:i, names[1]:j}).data
                
                uu = raw_u * np.cos(np.deg2rad(raw_a)) - raw_v * np.sin(np.deg2rad(raw_a))
                vv = raw_u * np.sin(np.deg2rad(raw_a)) + raw_v * np.cos(np.deg2rad(raw_a))
                assert np.isclose(uu, expected_u)
                assert np.isclose(vv,expected_v)
        
    
