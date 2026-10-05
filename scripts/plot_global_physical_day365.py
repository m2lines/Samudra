#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Day365-only regional spectra for saved annual global physical-only rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from global_physical_spectra import (  # type: ignore[import-not-found]
    REGIONS,
    regional_spectra,
)
from global_physical_spectra_plotting import (  # type: ignore[import-not-found]
    plot_regional_spectra,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--om4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.arrays / "COMPLETE.json").read_text())
    native_meta = json.loads((args.native / "NATIVE_COMPLETE.json").read_text())
    om4_meta = json.loads((args.om4 / "COMPLETE.json").read_text())
    grid = dict(np.load(args.arrays / "grid.npz"))
    lat, lon, wet = grid["lat"], grid["lon"], grid["mask"]
    origins = ["2015-01-01", "2018-01-01", "2021-01-01"]
    stages = [
        ("obs00050", "Early: 50 obs"),
        ("obs00500", "Developing: 500 obs"),
        ("obs02000", "Middle: 2,000 obs"),
        ("obs08000", "Final: 8,000 obs"),
        ("scratch08000", "Obs-only: 8,000 obs"),
        ("scratch16000", "Obs-only: 16,000 obs"),
    ]
    endpoints = ["obs08000", "scratch08000", "scratch16000"]
    arrays: dict = {}
    sources = {}
    for key, _ in stages:
        arrays[key] = []
        for origin in origins:
            path = args.arrays / (key + "-" + origin + ".npz")
            record = meta["files"][path.name]
            assert (
                digest(path) == record["sha256"]
                and path.stat().st_size == record["bytes"]
            )
            data = dict(np.load(path))
            assert len(data["surface"]) == len(data["reference"]) == 73
            np.testing.assert_array_equal(
                data["surface"][-1], [data["state"][-1, 0], data["state"][-1, 6]]
            )
            arrays[key].append(data)
            sources[str(path)] = record
    predictions = {
        key: np.stack([value["surface"][72] for value in values])
        for key, values in arrays.items()
    }
    reference = np.stack([value["reference"][72, :2] for value in arrays["obs08000"]])
    for values in arrays.values():
        np.testing.assert_allclose(
            np.stack([v["reference"][72, :2] for v in values]),
            reference,
            atol=0,
            rtol=0,
            equal_nan=True,
        )
    om4_path = args.om4 / "om4-day365.npz"
    record = om4_meta["files"][om4_path.name]
    assert (
        digest(om4_path) == record["sha256"]
        and om4_path.stat().st_size == record["bytes"]
    )
    om4 = dict(np.load(om4_path))
    np.testing.assert_array_equal(om4["origins"], origins)
    np.testing.assert_array_equal(om4["lat"], lat)
    np.testing.assert_array_equal(om4["lon"], lon)
    native_grids = {}
    native = {}
    for channel, product in enumerate(["oisst", "duacs"]):
        assert (
            native_meta[product]["lead_days"] == 365
            and native_meta[product]["origins"] == origins
        )
        grid_path = args.native / (product + "-grid.npz")
        native_grids[product] = dict(np.load(grid_path))
        sources[str(grid_path)] = {
            "sha256": digest(grid_path),
            "bytes": grid_path.stat().st_size,
        }
        frames = []
        for i, origin in enumerate(origins):
            path = args.native / (product + "-" + origin + ".npz")
            assert digest(path) == native_meta[product]["files"][path.name]
            data = dict(np.load(path))
            interval = om4_meta["alignment"][i]
            assert (
                str(data["start"]) == interval["start"]
                and str(data["end"]) == interval["end"]
            )
            assert native_meta[product]["dates"][i] == [
                interval["start"],
                interval["end"],
            ]
            np.testing.assert_array_equal(
                np.isfinite(data["coarse"]), np.isfinite(reference[i, channel])
            )
            np.testing.assert_allclose(
                data["coarse"],
                reference[i, channel],
                atol=2e-6,
                rtol=1e-6,
                equal_nan=True,
            )
            frames.append(data["native"])
            sources[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
        native[product] = np.stack(frames)
    spectra = regional_spectra(
        predictions, reference, om4["surface"], lat, lon, wet, native, native_grids
    )
    plot_regional_spectra(
        spectra,
        stages,
        endpoints,
        args.output,
        "mean of three day-365 spatial power spectra; January starts 2015 / 2018 / 2021",
        prefix="day365-",
    )
    distances: dict = {}
    for variable, regions in spectra.items():
        distances[variable] = {}
        for key, _ in stages:
            distances[variable][key] = {}
            for baseline in ["Observations (1°)", "OM4 (1°)"]:
                means = []
                for region, _, _ in REGIONS[:4]:
                    a, b = regions[region][key], regions[region][baseline]
                    np.testing.assert_array_equal(a["k_rad_km"], b["k_rad_km"])
                    keep = np.array(a["k_rad_km"]) <= 0.5 * min(
                        a["nyquist_rad_km"], b["nyquist_rad_km"]
                    )
                    means.append(
                        float(
                            np.mean(
                                np.abs(
                                    np.log10(np.array(a["power"])[keep])
                                    - np.log10(np.array(b["power"])[keep])
                                )
                            )
                        )
                    )
                distances[variable][key][baseline] = float(np.mean(means))
    result = {
        "lead_days": 365,
        "origins": origins,
        "checkpoints": {key: meta["checkpoints"][key]["lineage"] for key, _ in stages},
        "definition": "Each spectrum uses only the final five-day output of a 73-step annual autoregression. Mean of three individual snapshot powers, not an average over rollout leads. Three January origins differ from the twelve monthly 2015 origins of day30 figures. Exact matching native/coarse/OM4 reference intervals verified. Same transform, regions, masking, native geographic support and geostrophic exclusions as day30. Diagnostic only; selection unchanged.",
        "spectra": spectra,
        "spectral_distance": distances,
        "sources": sources,
        "om4": om4_meta,
        "native": native_meta,
        "verification": [
            "Annual surface endpoints exactly equal saved day365 map states for all six checkpoints and three origins",
            "Exact checkpoint-independent coarse observations",
            "Native daily-to-coarse and five-day means reproduce annual reference masks/values",
            "Native and OM4 intervals equal final forecast bin for each origin",
            "Archive/member hashes and fixed checkpoint lineage verified",
        ],
        "figures": {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in args.output.glob("*.png")
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DAY365_SPECTRA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
