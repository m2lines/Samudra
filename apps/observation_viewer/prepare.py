# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage verified report exports as memory-mapped arrays for the Panel viewer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {
    "obs00050": "Mixed · 50 observation updates",
    "obs00500": "Mixed · 500 observation updates",
    "obs02000": "Mixed · 2,000 observation updates",
    "obs08000": "Mixed · 8,000 observation updates",
    "scratch08000": "Observations only · 8,000 updates",
    "scratch16000": "Observations only · 16,000 updates",
}
ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]
REPORT = "https://github.com/m2lines/Samudra/blob/codex/d-observation-pilot/docs/experiments/observation-d/global-physical-day30-2026-09-30.md"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(compact, interiors, references, output):
    """Verify source receipts, preserve values/masks, and write a portable catalog."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "Choose an empty output directory; completed data is immutable"
        )
    receipts = {
        root: json.loads((root / "COMPLETE.json").read_text())
        for root in [compact, *interiors]
    }
    sources, files = {}, {}

    def read(root, name):
        path = root / name
        actual = digest(path)
        if root in receipts and actual != receipts[root]["files"][name]["sha256"]:
            raise ValueError(f"Report receipt mismatch: {path}")
        sources[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)

    def write(name, values):
        path = output / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        # Read back the entire written payload, including NaNs and masks.
        stored = np.load(path, mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(stored, values)
        files[path.name] = {
            "sha256": digest(path),
            "shape": list(stored.shape),
            "dtype": str(stored.dtype),
            "bytes": path.stat().st_size,
        }
        return path.name

    grid = read(compact, "grid.npz")
    interior_grid = read(interiors[0], "grid.npz")
    for axis in ("lat", "lon"):
        np.testing.assert_array_equal(grid[axis], interior_grid[axis])
    for root in interiors[1:]:
        other = read(root, "grid.npz")
        for key, values in interior_grid.items():
            np.testing.assert_array_equal(other[key], values)
    interior_roots = {
        model: root for root in interiors for model in receipts[root]["stages"]
    }
    models = {
        key: {
            "label": label,
            "lineage": receipts[compact]["checkpoints"][key]["lineage"],
        }
        for key, label in LABELS.items()
        if key in receipts[compact]["checkpoints"] and key in interior_roots
    }
    if not models:
        raise ValueError("No matched surface/interior models found")
    surface_mask = grid["mask"][[0, 6]]
    interior_mask = interior_grid["mask"][:, :14].transpose(1, 0, 2, 3)
    # Full-column support, including the layers straddling 700 and 2000 m.
    heat_mask = np.stack([interior_mask[:11, 0].all(0), interior_mask[10:14, 0].all(0)])
    climate = read(interiors[0], "december-climatology.npz")["ts"]
    for root in interiors[1:]:
        np.testing.assert_array_equal(
            read(root, "december-climatology.npz")["ts"], climate
        )
    catalog = {
        "schema_version": 1,
        "report_url": REPORT,
        "models": models,
        "origins": ORIGINS,
        "lat": grid["lat"].tolist(),
        "lon": grid["lon"].tolist(),
        "depths": interior_grid["depths"][:14].tolist(),
        "surface_mask": write("surface-mask", surface_mask),
        "interior_mask": write("interior-mask", interior_mask),
        "heat_mask": write("heat-mask", heat_mask),
        "heat_unit": "GJ/m²",
        "heat_input_unit": "J/m²",
        "heat_depth_bounds_m": [[0, 700], [700, 2000]],
        "climatology": write(
            "december-climatology",
            np.where(interior_mask, climate.transpose(1, 0, 2, 3), np.nan),
        ),
        "records": {},
    }
    for origin in ORIGINS:
        month = f"{int(origin[:4]) - 1}-12"
        reference = read(references, f"{month}.npz")
        np.testing.assert_array_equal(reference["depth"], catalog["depths"])
        if str(reference["time"]) != month:
            raise ValueError("Interior reference month does not match forecast origin")
        record = {
            "context_month": month,
            "interior_reference": write(
                f"{origin}-interior-reference",
                np.where(
                    interior_mask, reference["values"].transpose(1, 0, 2, 3), np.nan
                ),
            ),
            "models": {},
        }
        shared_reference = None
        shared_heat_reference = None
        for model in models:
            values = read(compact, f"{model}-{origin}.npz")
            surface = np.where(surface_mask, values["surface"], np.nan)
            if surface.shape != (73, 2, len(grid["lat"]), len(grid["lon"])):
                raise ValueError(f"Unexpected annual shape for {model}/{origin}")
            observed = np.where(surface_mask, values["reference"][:, :2], np.nan)
            if shared_reference is None:
                shared_reference = observed
                record["surface_reference"] = write(
                    f"{origin}-surface-reference", observed
                )
            else:
                np.testing.assert_array_equal(observed, shared_reference)
            months = values["months"].tolist()
            expected_months = [f"{origin[:4]}-{month:02d}" for month in range(1, 13)]
            if months != expected_months:
                raise ValueError(f"Unexpected heat-content months for {model}/{origin}")
            record["heat_months"] = months
            heat_values = {}
            for key in ["predicted_ohc", "reference_ohc"]:
                raw = values[key]
                if raw.shape != (12, 2, len(grid["lat"]), len(grid["lon"])):
                    raise ValueError(f"Unexpected {key} shape for {model}/{origin}")
                if np.isfinite(raw[:, ~heat_mask]).any():
                    raise ValueError(
                        f"Unexpected {key} values outside full-column mask"
                    )
                heat_values[key] = np.where(
                    heat_mask, raw.astype(np.float64) / 1e9, np.nan
                )
            heat_reference = heat_values["reference_ohc"]
            if shared_heat_reference is None:
                shared_heat_reference = heat_reference
                record["heat_reference"] = write(
                    f"{origin}-heat-reference", heat_reference
                )
            else:
                np.testing.assert_array_equal(heat_reference, shared_heat_reference)
            ts = read(interior_roots[model], f"{model}-{origin}.npz")["ts"]
            interior = np.where(interior_mask, ts[:, :14].transpose(1, 0, 2, 3), np.nan)
            # The compact and full-depth exports must describe the same initializer.
            for compact_channel, variable, depth in [
                (0, 0, 0),
                (1, 0, 9),
                (2, 1, 0),
                (3, 1, 9),
            ]:
                expected = np.where(
                    interior_mask[depth, variable],
                    values["initial"][compact_channel],
                    np.nan,
                )
                np.testing.assert_array_equal(interior[depth, variable], expected)
            record["models"][model] = {
                "surface": write(f"{model}-{origin}-surface", surface),
                "interior": write(f"{model}-{origin}-interior", interior),
                "heat": write(f"{model}-{origin}-heat", heat_values["predicted_ohc"]),
            }
            print(f"Verified {model} / {origin}", flush=True)
        catalog["records"][origin] = record
    catalog["sources"] = sources
    catalog["files"] = files
    # Preserve full original receipts alongside the new catalog.
    for i, (root, receipt) in enumerate(receipts.items()):
        catalog.setdefault("receipts", {})[str(root)] = f"source-receipt-{i}.json"
        (output / f"source-receipt-{i}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--interior", type=Path, action="append", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.compact, args.interior, args.references, args.output)
