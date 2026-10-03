#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the fixed January/July day-30 cases from completed extent runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(methods, data, output):
    cases = ["2022-01", "2022-07"]
    provenance = dict(
        scope="Saved day-30 five-day mean forecasts; no new inference or training",
        cases=cases,
        case_choice="January and July of the final held-out year, inherited from the original report",
        lead_index=5,
        grid_sha256=digest(data / "grid.npz"),
        script_sha256=digest(Path(__file__)),
        sources=[],
        case_times={},
    )
    with np.load(data / "grid.npz") as grid:
        arrays = {key: grid[key] for key in ("lat", "lon")}
        arrays["mask"] = grid["mask"][0]
    all_methods = methods + [
        ["Training seasonal climatology", methods[0][1], "seasonal-climatology.npz"]
    ]
    predictions, reference = [], None
    for label, directory, filename in all_methods:
        run = Path(directory)
        folder = run / "test-selected"
        complete = json.loads((folder / "COMPLETE.json").read_text())
        selection = json.loads((run / "best.json").read_text())
        training = json.loads((run / "TRAIN_COMPLETE.json").read_text())
        fingerprint = json.loads((folder / "evaluation-input.json").read_text())
        checksum = digest(run / "best.pt")
        if complete["origins"] != 96 or {
            complete["selected_sha256"],
            selection["checkpoint_sha256"],
            training["best_checkpoint_sha256"],
            fingerprint["checkpoint_sha256"],
        } != {checksum}:
            raise ValueError("Evaluation/selection/training checkpoint lineage differs")
        if not fingerprint["global_observations"] or fingerprint["split"] != "test":
            raise ValueError("Map source is not the global test evaluation")
        path = folder / filename
        with np.load(path) as export:
            origins = export["origins"].tolist()
            if len(origins) != 96 or len(set(origins)) != 96:
                raise ValueError("Export has incomplete or duplicate origins")
            indices = [origins.index(case) for case in cases]
            prediction = export["prediction"][indices, 5, :2]
            truth = export["reference"][indices, 5, :2]
        if reference is None:
            reference = truth
        elif not np.array_equal(reference, truth, equal_nan=True):
            raise ValueError("Map candidates have different observations")
        if label == "Training seasonal climatology":
            arrays["climatology"] = prediction
        else:
            predictions.append(prediction)
        provenance["sources"].append(
            dict(
                method=label,
                path=str(path),
                sha256=digest(path),
                selected_checkpoint_sha256=checksum,
                source_checkpoint_sha256=None,
            )
        )
    for case in cases:
        with np.load(data / "test" / (case + ".npz")) as sample:
            provenance["case_times"][case] = dict(
                last_history_midpoint=str(sample["midpoints"][18]),
                day30_bin_midpoint=str(sample["midpoints"][24]),
            )
    arrays["reference"] = reference
    arrays["prediction"] = np.array(predictions)
    arrays["provenance"] = np.array(json.dumps(provenance))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(
        json.dumps(
            dict(bundle=str(output), sha256=digest(output), bytes=output.stat().st_size)
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        type=Path,
        required=True,
        help="JSON [literal label, run directory, export filename] rows",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(json.loads(args.methods.read_text()), args.data, args.output)


if __name__ == "__main__":
    main()
