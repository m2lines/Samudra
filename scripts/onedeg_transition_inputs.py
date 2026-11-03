# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Freeze small benchmark inputs from the previously verified original-data cache."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.onedeg_weight_edit import save_json, sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-times", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path("experiments/onedeg-weight-edits/transition_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    old_manifest = json.loads((args.cache / "data_manifest.json").read_text())
    anomalies = np.load(args.cache / "hfds_anomalies.npy", mmap_mode="r")
    dates = json.loads(args.source_times.read_text())
    assert len(dates) == len(anomalies)
    for entry in old_manifest["windows"]:
        start = entry["start_index"]
        assert [str(t) for t in dates[start : start + len(entry["times"])]] == entry[
            "times"
        ]
    manifest = {
        "original_data_root": old_manifest["data_root"],
        "forcing_transform": old_manifest["anomaly_transform"],
        "forcing_cache_sha256": sha256(args.cache / "hfds_anomalies.npy"),
        "windows": [],
    }
    for split, year in protocol["years"].items():
        for month in protocol["months"]:
            start = next(
                i for i, t in enumerate(dates) if t.startswith(f"{year}-{month:02d}-")
            )
            name = f"{split}_{year}_{month:02d}"
            manifest["windows"].append(
                {
                    "name": name,
                    "split": split,
                    "start_index": start,
                    "times": [str(t) for t in dates[start : start + 14]],
                }
            )
            np.save(args.output / f"{name}_anomaly.npy", anomalies[start : start + 14])
    states = np.load(args.cache / "discovery_2015_01.npz")["states"]
    manifest["original_cached_states_sha256"] = hashlib.sha256(
        states.tobytes()
    ).hexdigest()
    save_json(args.output / "manifest.json", manifest)
    shutil.copyfile(protocol_path, args.output / protocol_path.name)
    for name in ("static.npz", "forcing_stats.npz", "carry_basis.pt"):
        shutil.copyfile(args.cache / name, args.output / name)
    shutil.copyfile(
        args.cache / "discovery_2015_01_baseline.npy",
        args.output / "original_reference_prediction.npy",
    )


if __name__ == "__main__":
    main()
