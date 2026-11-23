# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Create the portable viewer catalog after full checksum and cohort checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assemble(root):
    models = {}
    grids: dict = {}
    shared: dict[str, Path] = {}
    for arm, label in [("mixed", "LLC + OBS"), ("obs-only", "OBS only")]:
        complete = json.loads((root / arm / "COMPLETE.json").read_text())
        assert set(complete["checkpoints"]) == {"10000", "11000"}
        for update, checkpoint in complete["checkpoints"].items():
            assert (
                checkpoint["training_comparison"]
                and checkpoint["identity"]["hours"] == 720
            )
            assert (
                checkpoint["identity"]["origins"] == 8
                and len(checkpoint["cases"]) == 16
            )
            assert checkpoint["identity"]["manifest"]["config"]["pretrain_mix"] == (
                "llc-obs" if arm == "mixed" else "obs"
            )
            cases = {}
            for name, entry in checkpoint["cases"].items():
                receipt = root / arm / entry["receipt"]
                assert sha(receipt) == entry["sha256"]
                proof = json.loads(receipt.read_text())
                assert proof["identity"] == checkpoint["identity"]
                prefix = Path(arm) / f"update-{update}"
                for path, metadata in proof["files"].items():
                    target = root / prefix / path
                    assert sha(target) == metadata["sha256"]
                    # References are identical across checkpoints. Hard links keep the
                    # transferred bundle compact while preserving every receipt path.
                    if "/prediction-" not in path:
                        digest = metadata["sha256"]
                        if digest in shared and not os.path.samefile(
                            target, shared[digest]
                        ):
                            temp = target.with_suffix(".link")
                            os.link(shared[digest], temp)
                            temp.replace(target)
                        shared[digest] = target
                grid_path = receipt.parent / "grid.npz"
                assert sha(grid_path) == proof["grid_sha256"]
                grid = dict(np.load(grid_path, allow_pickle=False))
                if grids:
                    for key, value in grid.items():
                        np.testing.assert_array_equal(value, grids[key])
                else:
                    grids = grid
                case = proof["case"]
                for key in (
                    "history",
                    "observed_initial",
                    "observed_surface",
                    "llc_initial",
                    "llc_interior",
                    "llc_surface",
                ):
                    case[key] = str(prefix / case[key])
                case["prediction"] = {
                    k: str(prefix / v) for k, v in case["prediction"].items()
                }
                cases[name] = case
            models[f"{arm}-{update}"] = dict(
                label=f"{label} · EMA {int(update):,}"
                + (" · OBS cooldown" if update == "11000" else " · before cooldown"),
                identity=checkpoint["identity"],
                cases=cases,
            )
    keys = [set(model["cases"]) for model in models.values()]
    assert all(k == keys[0] for k in keys)
    reference = next(iter(models.values()))["cases"]
    for model in models.values():
        for name, case in model["cases"].items():
            for key in (
                "source",
                "origin",
                "times",
                "initial_times",
                "saved_hours",
                "saved_times",
                "history_times",
            ):
                assert case[key] == reference[name][key], "Mismatched evaluation cohort"
    np.savez_compressed(root / "grid.npz", **grids)
    catalog = dict(schema="llc-six-hour-viewer-v1", grid="grid.npz", models=models)
    temp = root / "catalog.tmp"
    temp.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    temp.replace(root / "catalog.json")
    print(
        json.dumps(
            dict(
                catalog_sha256=sha(root / "catalog.json"),
                grid_sha256=sha(root / "grid.npz"),
                models=list(models),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    assemble(p.parse_args().root)
