import json
import subprocess
import sys
from pathlib import Path

import xarray as xr

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_zarr_v3_overlay.py"
)


def run_script(source_path: Path, output_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(source_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_overlay_builder_preserves_xarray_dimension_names(tmp_path: Path):
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "overlay.zarr"
    ds = xr.Dataset(
        {
            "thetao": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "mask": (("x",), [True, False]),
        }
    ).chunk({"time": 1, "x": 2})
    ds.to_zarr(source_path, zarr_format=2)

    run_script(source_path, output_path)

    metadata = json.loads((output_path / "zarr.json").read_text())
    consolidated = metadata["consolidated_metadata"]["metadata"]

    assert consolidated["thetao"]["dimension_names"] == ["time", "x"]
    assert consolidated["mask"]["dimension_names"] == ["x"]
    assert (output_path / "thetao").is_symlink()
    assert (output_path / "mask").is_symlink()
