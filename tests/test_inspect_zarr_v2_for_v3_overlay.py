import json
import subprocess
import sys
from pathlib import Path

import xarray as xr
import zarr

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "inspect_zarr_v2_for_v3_overlay.py"
)


def run_script(store_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(store_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_reports_group_store_with_consolidated_metadata(tmp_path: Path):
    store_path = tmp_path / "dataset.zarr"
    ds = xr.Dataset(
        {
            "thetao": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "mask": (("x",), [True, False]),
        }
    ).chunk({"time": 1, "x": 2})
    ds.to_zarr(store_path, zarr_format=2)
    zarr.consolidate_metadata(store_path)

    report = run_script(store_path)

    assert report["metadata_source"] == "consolidated"
    assert report["root_kind"] == "group"
    assert report["feasibility"]["verdict"] == "likely_feasible"
    assert report["summary"]["array_count"] == 2
    assert [array["path"] for array in report["arrays"]] == ["mask", "thetao"]
    assert report["summary"]["dimension_separators"] == ["."]


def test_flags_root_array_store_for_manual_review(tmp_path: Path):
    store_path = tmp_path / "array.zarr"
    array = zarr.open(
        store_path, mode="w", shape=(8,), chunks=(4,), dtype="i4", zarr_format=2
    )
    array[:] = range(8)

    report = run_script(store_path)

    assert report["root_kind"] == "array"
    assert report["feasibility"]["verdict"] == "needs_manual_review"
    assert any(
        "root is an array" in reason for reason in report["feasibility"]["reasons"]
    )
