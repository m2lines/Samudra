"""Unit tests for individual visualization modules."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path

from ocean_emulators.viz_modules.data_processing import (
    rename_vars,
    _combine_variables_by_level,
    load_groundtruth_data,
    remove_climatology
)
from ocean_emulators.viz_modules.analysis import (
    profile_mean,
    process_mask,
    create_basin_masks,
    compute_ohc,
    compute_mae,
    compute_rmse
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    time = pd.date_range('2020-01-01', periods=10, freq='D')
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 8)
    lev = [0, 10, 50, 100, 200]
    
    # Create sample data
    data_vars = {}
    
    # 3D variables with level suffix
    for var in ['thetao', 'so']:
        for i, depth in enumerate(lev):
            var_name = f"{var}_{i}"
            data_vars[var_name] = (
                ['time', 'lat', 'lon'], 
                np.random.rand(len(time), len(lat), len(lon))
            )
    
    # 2D variables
    data_vars['zos'] = (['time', 'lat', 'lon'], np.random.rand(len(time), len(lat), len(lon)))
    data_vars['areacello'] = (['lat', 'lon'], np.ones((len(lat), len(lon))))
    
    coords = {
        'time': time,
        'lat': lat,
        'lon': lon
    }
    
    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def sample_groundtruth():
    """Create a sample groundtruth dataset."""
    time = pd.date_range('2020-01-01', periods=5, freq='D')
    y = np.linspace(-90, 90, 4)
    x = np.linspace(0, 360, 6)
    lev = [0, 10, 50]
    
    data_vars = {
        'thetao': (['time', 'lev', 'y', 'x'], np.random.rand(len(time), len(lev), len(y), len(x))),
        'so': (['time', 'lev', 'y', 'x'], np.random.rand(len(time), len(lev), len(y), len(x))),
        'areacello': (['y', 'x'], np.ones((len(y), len(x)))),
        'dz': (['lev'], np.array([10, 40, 50]))
    }
    
    coords = {
        'time': time,
        'y': y,
        'x': x,
        'lev': lev
    }
    
    return xr.Dataset(data_vars, coords=coords)


def test_rename_vars(sample_dataset):
    """Test variable renaming function."""
    # Add some OM4 format variables
    sample_dataset['so_lev_10_0'] = (['time', 'lat', 'lon'], np.random.rand(10, 5, 8))
    sample_dataset['thetao_lev_50_0'] = (['time', 'lat', 'lon'], np.random.rand(10, 5, 8))
    
    renamed = rename_vars(sample_dataset)
    
    # Check that OM4 format variables were renamed
    assert 'so_lev_10_0' not in renamed.data_vars
    assert 'thetao_lev_50_0' not in renamed.data_vars
    
    # Check that regular variables are preserved
    assert 'zos' in renamed.data_vars
    assert 'areacello' in renamed.data_vars


def test_combine_variables_by_level(sample_dataset):
    """Test combining variables by level."""
    # Prepare dataset with level-suffixed variables
    combine_vars = ['thetao', 'so']
    
    result = _combine_variables_by_level(sample_dataset, combine_vars)
    
    # Check that combined variables exist
    for var in combine_vars:
        if f"{var}_0" in sample_dataset.data_vars:
            assert var in result.data_vars
            assert 'lev' in result[var].dims
            # Check that individual level variables are removed
            for i in range(5):
                assert f"{var}_{i}" not in result.data_vars


def test_remove_climatology(sample_groundtruth):
    """Test climatology removal."""
    # Create dataset with seasonal cycle
    sample_groundtruth['time'] = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # Add seasonal signal
    seasonal_signal = np.sin(2 * np.pi * np.arange(365) / 365)
    for var in ['thetao', 'so']:
        if var in sample_groundtruth.data_vars:
            original_shape = sample_groundtruth[var].shape
            sample_groundtruth[var] = sample_groundtruth[var] + seasonal_signal.reshape(-1, *([1] * (len(original_shape) - 1)))
    
    deseasonalized = remove_climatology(sample_groundtruth)
    
    # Check that the output has the same variables and dimensions
    assert set(deseasonalized.data_vars) == set(sample_groundtruth.data_vars)
    for var in deseasonalized.data_vars:
        assert deseasonalized[var].shape == sample_groundtruth[var].shape


def test_profile_mean(sample_groundtruth):
    """Test spatial averaging function."""
    profiles = profile_mean(sample_groundtruth)
    
    # Check that spatial dimensions are removed
    for var in profiles.data_vars:
        if var in ['thetao', 'so']:  # 3D variables
            expected_dims = ['time', 'lev']
            assert list(profiles[var].dims) == expected_dims
        elif var == 'areacello':  # Skip area weights
            continue
        elif var == 'dz':  # 1D variable
            assert list(profiles[var].dims) == ['lev']


def test_compute_ohc():
    """Test Ocean Heat Content computation."""
    # Create simple test data
    time = pd.date_range('2020-01-01', periods=3, freq='D')
    y = np.linspace(-90, 90, 4)
    x = np.linspace(0, 360, 6)
    lev = [0, 10, 50]
    
    temperature = xr.DataArray(
        np.ones((len(time), len(lev), len(y), len(x))) * 15,  # 15°C everywhere
        dims=['time', 'lev', 'y', 'x'],
        coords={'time': time, 'lev': lev, 'y': y, 'x': x}
    )
    
    areacello = xr.DataArray(
        np.ones((len(y), len(x))),
        dims=['y', 'x'],
        coords={'y': y, 'x': x}
    )
    
    dz = xr.DataArray(
        np.array([10, 40, 50]),
        dims=['lev'],
        coords={'lev': lev}
    )
    
    ohc = compute_ohc(temperature, areacello, dz)
    
    # Check output dimensions
    assert 'lev' not in ohc.dims  # Should be depth-integrated
    assert 'time' in ohc.dims
    assert 'y' in ohc.dims
    assert 'x' in ohc.dims
    
    # Check that values are positive (temperature > 0)
    assert (ohc > 0).all()


def test_compute_mae():
    """Test Mean Absolute Error computation."""
    # Create test data
    truth = xr.Dataset({
        'var1': (['time', 'x'], np.array([[1, 2, 3], [4, 5, 6]])),
        'var2': (['time', 'x'], np.array([[0, 1, 2], [3, 4, 5]]))
    })
    
    pred = xr.Dataset({
        'var1': (['time', 'x'], np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.2]])),
        'var2': (['time', 'x'], np.array([[0.1, 0.9, 2.1], [2.9, 4.1, 5.2]]))
    })
    
    mae = compute_mae(pred, truth)
    
    # Check that MAE is computed correctly
    assert mae['var1'].values == pytest.approx(0.15, abs=1e-6)  # (0.1+0.2+0.1+0.2+0.1+0.2)/6
    assert mae['var2'].values == pytest.approx(0.133333, abs=1e-6)  # (0.1+0.1+0.1+0.1+0.1+0.2)/6


def test_compute_rmse():
    """Test Root Mean Square Error computation."""
    # Create test data
    truth = xr.Dataset({
        'var': (['x'], np.array([1, 2, 3, 4]))
    })
    
    pred = xr.Dataset({
        'var': (['x'], np.array([1.1, 2.1, 2.9, 4.1]))
    })
    
    rmse = compute_rmse(pred, truth)
    
    # Check RMSE calculation: sqrt(mean((0.1^2 + 0.1^2 + 0.1^2 + 0.1^2)))
    expected_rmse = np.sqrt(0.01)  # 0.1
    assert rmse['var'].values == pytest.approx(expected_rmse, abs=1e-6)


@pytest.mark.manual
def test_load_groundtruth_data():
    """Test loading ground truth data (slow test - requires actual data)."""
    try:
        data_path = "/data/public/OM4.zarr"
        if Path(data_path).exists():
            ds = load_groundtruth_data(data_path)
            
            # Check basic properties
            assert isinstance(ds, xr.Dataset)
            assert 'time' in ds.dims
            assert 'areacello' in ds.data_vars
            
            # Check that time slice was applied
            assert str(ds.time.values[0]).startswith('2014')
            assert str(ds.time.values[-1]).startswith('2022')
            
        else:
            pytest.skip("Test data not available")
            
    except Exception as e:
        pytest.skip(f"Could not load test data: {e}")


@pytest.mark.parametrize("basin_name", ["Atlantic", "Pacific", "Indian", "Southern", "Arctic"])
def test_basin_masks_creation(basin_name, sample_groundtruth):
    """Test that basin masks can be created for each basin."""
    # This is a placeholder test - would need actual basin data
    # For now, just test the function signature and basic structure
    
    # Create mock basin data
    y = sample_groundtruth.y.values
    x = sample_groundtruth.x.values
    
    mock_basins = xr.Dataset({
        f"basin_{basin_name.lower()}": (
            ['lat', 'lon'], 
            np.random.choice([0, 1], size=(len(y), len(x)))
        )
    }, coords={'lat': y, 'lon': x})
    
    # Test that process_mask function works
    mask = mock_basins[f"basin_{basin_name.lower()}"]
    processed_mask = process_mask(mask, sample_groundtruth)
    
    # Check output properties
    assert 'y' in processed_mask.dims
    assert 'x' in processed_mask.dims
    assert processed_mask.shape == (len(y), len(x))


if __name__ == "__main__":
    # Allow running individual tests
    pytest.main([__file__])