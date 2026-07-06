# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import numpy as np
import torch
import xarray as xr

from samudra.constants import TensorMap, build_om4_spec
from samudra.utils.data import Normalize
from samudra.utils.writer import ZarrWriter
from tests.conftest import TEST_FULL_DATASET_SPEC

# write() never touches normalization (record_batch does), so these tests drive
# the writer directly with a buffer and omit a real Normalize.
_NO_NORMALIZE = cast(Normalize, None)


def _source_coords(ny, nx):
    """Coords as `get_coords_dict` should be returned: 1D lat/lon dims,
    plus the grid metadata that survives `with_lat_lon_coords` (areacello, dz,
    lev, ocean_fraction)."""
    spec = TEST_FULL_DATASET_SPEC
    n_lev = spec.num_prognostic_depth_levels
    lat = xr.DataArray(np.linspace(-89, 89, ny), dims="lat")
    lon = xr.DataArray(np.linspace(0, 359, nx), dims="lon")
    return {
        "lat": lat,
        "lon": lon,
        "lev": xr.DataArray(list(spec.depth_levels), dims="lev"),
        "dz": xr.DataArray(list(spec.depth_thickness), dims="lev"),
        "areacello": xr.DataArray(np.ones((ny, nx)), dims=["lat", "lon"]),
        "ocean_fraction": xr.DataArray(
            np.ones((n_lev, ny, nx)), dims=["lev", "lat", "lon"]
        ),
        # cell bounds survive on y_b/x_b dims (kept by with_lat_lon_coords)
        "lat_b": xr.DataArray(np.zeros((ny + 1, nx + 1)), dims=["y_b", "x_b"]),
        "lon_b": xr.DataArray(np.zeros((ny + 1, nx + 1)), dims=["y_b", "x_b"]),
    }


def test_writer_output_is_analysis_ready(tmp_path):
    """The eval writer emits depth-stacked vars on y/x dims with grid metadata."""
    spec = TEST_FULL_DATASET_SPEC
    tensor_map = TensorMap(dataset_spec=spec)
    names = list(tensor_map.prognostic_var_names)
    n_channels, n_lev = len(names), spec.num_prognostic_depth_levels
    nt, ny, nx = 2, 3, 4

    coords = _source_coords(ny, nx)
    writer = ZarrWriter(
        tmp_path,
        coords=coords,
        hist=0,
        model_path="dummy.ckpt",
        time_chunk_size=4,
        normalize=_NO_NORMALIZE,
        tensor_map=tensor_map,
    )

    # buffer[t, c] is uniformly the channel index c, so each reassembled level can
    # be checked against the channel it must have come from.
    buffer = np.broadcast_to(
        np.arange(n_channels)[None, :, None, None],
        (nt, n_channels, ny, nx),
    ).astype("float32")
    writer.buffer = torch.from_numpy(buffer.copy())
    writer.time_buffer = xr.DataArray(np.arange(nt), dims="time")

    writer.write()
    out = xr.open_zarr(writer.pred_path)

    # (#1) 3D vars are depth-stacked; (#2) on y/x dims; zos stays 2D.
    assert set(out.data_vars) == {"thetao", "so", "uo", "vo", "zos"}
    assert out["thetao"].dims == ("time", "lev", "y", "x")
    assert out["zos"].dims == ("time", "y", "x")
    assert not any(f"thetao_{i}" in out.variables for i in range(n_lev))

    # Each level holds exactly the channel it was flattened from (no transposition).
    for base in ["thetao", "so", "uo", "vo"]:
        for i in range(n_lev):
            assert (out[base].isel(lev=i).values == names.index(f"{base}_{i}")).all()
    assert (out["zos"].values == names.index("zos")).all()

    # (#3) grid metadata propagated, with horizontal dims renamed lat/lon -> y/x.
    np.testing.assert_array_equal(out["lev"].values, spec.depth_levels)
    np.testing.assert_array_equal(out["dz"].values, spec.depth_thickness)
    assert out["areacello"].dims == ("y", "x")
    assert out["ocean_fraction"].dims == ("lev", "y", "x")
    # cell bounds propagate unchanged (enables dx/dy in analysis).
    assert out["lat_b"].dims == ("y_b", "x_b")
    assert out["lon_b"].dims == ("y_b", "x_b")

    # 2D lat/lon reconstructed to match the truth layout (broadcast of 1D y/x).
    assert out["lat"].dims == ("y", "x") and out["lon"].dims == ("y", "x")
    np.testing.assert_allclose(out["lat"].isel(x=0).values, coords["lat"].values)
    np.testing.assert_allclose(out["lon"].isel(y=0).values, coords["lon"].values)


def test_writer_appends_along_time(tmp_path):
    """A second write extends the time axis without disturbing other coords."""
    spec = TEST_FULL_DATASET_SPEC
    tensor_map = TensorMap(dataset_spec=spec)
    n_channels = len(tensor_map.prognostic_var_names)
    ny, nx = 3, 4

    coords = _source_coords(ny, nx)
    writer = ZarrWriter(
        tmp_path,
        coords=coords,
        hist=0,
        model_path="dummy.ckpt",
        time_chunk_size=4,
        normalize=_NO_NORMALIZE,
        tensor_map=tensor_map,
    )

    def _write(times):
        writer.buffer = torch.zeros(len(times), n_channels, ny, nx)
        writer.time_buffer = xr.DataArray(np.array(times), dims="time")
        writer.write()

    _write([0, 1])
    _write([2, 3, 4])

    out = xr.open_zarr(writer.pred_path)
    np.testing.assert_array_equal(out["time"].values, [0, 1, 2, 3, 4])
    assert out["thetao"].sizes["time"] == 5
    assert out["areacello"].dims == ("y", "x")  # static coord unaffected


def test_writer_shallow_spec_slices_depth_metadata(tmp_path):
    """A shallow model spec must not conflict with full-depth source coords.

    A backfilled source carries 19-level `dz`/`ocean_fraction`, but a shallow spec
    (e.g. thermo_dynamic_5) emits fewer levels. The writer slices the depth-resolved
    coords to the emitted level count instead of raising on a conflicting `lev` dim.
    """
    spec = build_om4_spec(
        prognostic_vars_key="thermo_dynamic_5", boundary_vars_key="tau_hfds"
    )
    tensor_map = TensorMap(dataset_spec=spec)
    n_prog = spec.num_prognostic_depth_levels  # 5, fewer than the source's 19
    n_channels = len(tensor_map.prognostic_var_names)
    ny, nx, nt = 3, 4, 1

    # Source coords carry the FULL 19-level depth metadata.
    coords = {
        "lat": xr.DataArray(np.linspace(-89, 89, ny), dims="lat"),
        "lon": xr.DataArray(np.linspace(0, 359, nx), dims="lon"),
        "lev": xr.DataArray(list(spec.depth_levels), dims="lev"),
        "dz": xr.DataArray(list(spec.depth_thickness), dims="lev"),
        "ocean_fraction": xr.DataArray(
            np.ones((19, ny, nx)), dims=["lev", "lat", "lon"]
        ),
        "areacello": xr.DataArray(np.ones((ny, nx)), dims=["lat", "lon"]),
    }
    writer = ZarrWriter(
        tmp_path,
        coords=coords,
        hist=0,
        model_path="dummy.ckpt",
        time_chunk_size=4,
        normalize=_NO_NORMALIZE,
        tensor_map=tensor_map,
    )
    writer.buffer = torch.zeros(nt, n_channels, ny, nx)
    writer.time_buffer = xr.DataArray(np.arange(nt), dims="time")

    writer.write()  # must not raise on conflicting lev sizes
    out = xr.open_zarr(writer.pred_path)

    # Depth axis and all depth-resolved coords are sliced to the emitted levels.
    assert out.sizes["lev"] == n_prog
    assert out["thetao"].sizes["lev"] == n_prog
    assert out["dz"].sizes["lev"] == n_prog
    assert out["ocean_fraction"].sizes["lev"] == n_prog
    np.testing.assert_array_equal(out["lev"].values, spec.depth_levels[:n_prog])
    np.testing.assert_array_equal(out["dz"].values, spec.depth_thickness[:n_prog])
