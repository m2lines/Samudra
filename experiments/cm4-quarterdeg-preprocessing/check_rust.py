# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native compact Zarr reads with Xarray for every pilot channel."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import samudra_rust_loader as native  # type: ignore[import-not-found]
import xarray as xr


def main() -> None:
    root = Path("/scratch/jr7309/cm4-quarterdeg-20261023")
    ds = xr.open_zarr(root / "pilot-data/CM4.zarr", consolidated=True)
    selectors: list[tuple[str, int | None]] = [
        (name, level) for name in ("thetao", "so", "uo", "vo") for level in range(19)
    ] + [(name, None) for name in ("zos", "hfds", "tauuo", "tauvo", "wfo")]
    assert len(selectors) == 81
    pool = native.ZarrReadPool(4)
    reader = native.CompactOm4Reader(root / "pilot-data/CM4.zarr", selectors, pool)
    positions = [9, 0, 1, 1]
    output = np.empty((4, 81, 720, 1440), dtype=np.float32)
    started = time.perf_counter()
    reader.read_into(positions, selectors, output)
    native_seconds = time.perf_counter() - started
    channel = 0
    for name in ("thetao", "so", "uo", "vo", "zos", "hfds", "tauuo", "tauvo", "wfo"):
        expected = ds[name].isel(time=positions).values
        for level in range(19) if "lev" in ds[name].dims else [None]:
            actual_reference = expected if level is None else expected[:, level]
            np.testing.assert_array_equal(
                output[:, channel], actual_reference, err_msg=f"{name}/{level}"
            )
            channel += 1
    assert channel == 81
    audit = json.loads((root / "records/pilot-audit.json").read_text())
    result = {
        "passed": True,
        "source": audit["source"],
        "processing_commit": (root / "commit.txt").read_text().strip(),
        "rust_loader_matches_xarray_exactly": True,
        "channels": channel,
        "positions": positions,
        "native_seconds": native_seconds,
        "rust_source_sha256": hashlib.sha256(
            (root / "source/rust/loader/src/lib.rs").read_bytes()
        ).hexdigest(),
        "pilot_stored_bytes": audit["stored_bytes"],
        "estimated_full_stored_bytes": audit["stored_bytes"] * 1460,
    }
    (root / "records/pilot-qualified.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("PILOT_QUALIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
