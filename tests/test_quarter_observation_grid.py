# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import zarr  # type: ignore[import-untyped]
from scipy.special import roots_legendre

from samudra.constants import build_om4_layout
from scripts.prepare_quarter_observation_grid import (
    StaticOnlyStore,
    publish_grid,
    read_grid,
    validate_grid,
)


def compact_fixture(root):
    layout = build_om4_layout()
    data = zarr.open_group(str(root / "OM4.zarr"), mode="w")

    def array(group, name, dims, values=None, shape=None, dtype="f4"):
        result = group.create_dataset(
            name, data=values, shape=shape, dtype=dtype, fill_value=None
        )
        result.attrs["_ARRAY_DIMENSIONS"] = dims
        return result

    array(data, "time", ["time"], values=np.array([0, 5]))
    array(data, "lev", ["lev"], values=np.array(layout.depth_levels), dtype="f8")
    array(data, "y", ["y"], values=np.array([-45.0, 45.0]), dtype="f8")
    array(data, "x", ["x"], values=np.arange(45.0, 360, 90), dtype="f8")
    for name in ("uo", "vo", "thetao", "so"):
        array(data, name, ["time", "lev", "y", "x"], shape=(2, 19, 2, 4))
    for name in ("zos", "tauuo", "tauvo", "hfds"):
        array(data, name, ["time", "y", "x"], shape=(2, 2, 4))
    for depth in range(19):
        mask = np.ones((2, 4), dtype=bool)
        mask[0, depth % 4] = False
        array(data, f"mask_{depth}", ["y", "x"], values=mask, dtype="bool")
    array(data, "areacello", ["y", "x"], values=np.ones((2, 4)), dtype="f8")
    data.attrs["grid_type"] = "gaussian"
    data.attrs["m2lines/samudra_git_hash"] = "fixture-producer"
    zarr.consolidate_metadata(str(root / "OM4.zarr"))
    for store, offset in (("OM4_means.zarr", 0), ("OM4_stds.zarr", 1)):
        group = zarr.open_group(str(root / store), mode="w")
        for index, name in enumerate(
            (*layout.prognostic_var_names, *layout.boundary_var_names)
        ):
            array(group, name, [], values=np.array(index + offset, dtype="f4"))
        zarr.consolidate_metadata(str(root / store))


def test_grid_export_uses_canonical_mapping_without_temporal_payloads(tmp_path):
    compact_fixture(tmp_path)
    grid, evidence = read_grid(tmp_path)
    assert grid["mask"].shape == (77, 2, 4)
    np.testing.assert_array_equal(
        grid["names"], build_om4_layout().prognostic_var_names
    )
    np.testing.assert_array_equal(grid["mean"], np.arange(77))
    np.testing.assert_array_equal(grid["std"], np.arange(77) + 1)
    # The shared canonical mask implementation supplies the surface mask for SSH.
    np.testing.assert_array_equal(grid["mask"][76], grid["mask"][0])
    np.testing.assert_array_equal(grid["mask"][38:57], grid["mask"][:19])
    assert evidence["temporal_field_payloads_read"] == 0
    assert not {"uo", "vo", "thetao", "so", "zos", "hfds"}.intersection(
        evidence["payload_arrays_read"]["OM4.zarr"]
    )
    guarded = StaticOnlyStore(tmp_path / "OM4.zarr")
    with pytest.raises(RuntimeError, match="Forbidden temporal"):
        guarded["thetao/0.0.0.0"]
    with pytest.raises(TypeError, match="read-only"):
        guarded["new"] = b"bad"


def test_grid_reuse_checks_static_values_and_refuses_changed_artifacts(tmp_path):
    arrays = {"mask": np.array([[True, False]]), "mean": np.array([1.0], dtype="f4")}
    evidence = {"metadata_sha256": {"OM4.zarr": "metadata"}}
    first = publish_grid(tmp_path, arrays, evidence)
    assert publish_grid(tmp_path, arrays, evidence) == first
    # Simulate changed source payloads without changed metadata: the reuse
    # must compare current static values, not merely trust metadata hashes.
    changed = dict(arrays, mean=np.array([2.0], dtype="f4"))
    with pytest.raises(AssertionError, match="Source static values changed"):
        publish_grid(tmp_path, changed, evidence)
    with pytest.raises(ValueError, match="provenance"):
        publish_grid(tmp_path, arrays, {"metadata_sha256": "changed"})
    (tmp_path / "grid.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content hash"):
        publish_grid(tmp_path, arrays, evidence)
    assert json.loads((tmp_path / "GRID_READY.json").read_text()) == first


def test_grid_validation_rejects_regular_latitudes_and_bad_statistics(tmp_path):
    compact_fixture(tmp_path)
    arrays, _ = read_grid(tmp_path)
    arrays.update(
        lat=np.rad2deg(np.arcsin(roots_legendre(720)[0])),
        lon=(np.arange(1440) + 0.5) / 4,
        mask=np.broadcast_to(True, (77, 720, 1440)),
        areacello=np.broadcast_to(1.0, (720, 1440)),
    )
    validate_grid(arrays)
    with pytest.raises(AssertionError):
        validate_grid(dict(arrays, lat=np.linspace(-89.875, 89.875, 720)))
    with pytest.raises(ValueError, match="positive"):
        validate_grid(dict(arrays, std=np.zeros(77)))
    with pytest.raises(ValueError, match="channel order"):
        validate_grid(dict(arrays, names=arrays["names"][::-1]))
