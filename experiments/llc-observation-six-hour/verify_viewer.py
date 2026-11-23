# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify exported arrays and reproduce the frozen training validation metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_scores(validation, channels):
    if set(validation) != {"24", "48", "72"}:
        raise ValueError("Validation requires all three daily leads")
    for methods in validation.values():
        if set(methods) != {"model", "persistence"}:
            raise ValueError("Validation requires model and persistence")
        for scores in methods.values():
            for metric in ("mse_normalized", "mse_physical", "cells"):
                values = np.asarray(scores[metric])
                if values.shape != (channels,) or not np.isfinite(values).all():
                    raise ValueError("Incomplete or nonfinite channel metrics")
                if (values < 0).any() or (metric == "cells" and (values <= 0).any()):
                    raise ValueError("Invalid errors or empty scored support")


def verify(run_root, root, updates, origins, hours):
    cfg = yaml.safe_load((run_root / "train.yaml").read_text())
    train = Path(cfg["experiment"]["base_output_dir"]) / cfg["experiment"]["name"]
    logged = [
        json.loads(line) for line in (train / "metrics.jsonl").read_text().splitlines()
    ]
    checkpoints = {}
    for update in updates:
        directory = root / f"update-{update}"
        receipts = sorted(directory.glob("*/COMPLETE.json"))
        if len(receipts) != 2 * origins:
            raise ValueError("Incomplete export cohort")
        cases, identities = {}, []
        metrics: dict[str, list[float]] = {}
        for receipt in receipts:
            proof = json.loads(receipt.read_text())
            ident, case = proof["identity"], proof["case"]
            assert (
                ident["updates"] == update
                and ident["hours"] == hours
                and ident["origins"] == origins
            )
            identities.append(ident)
            required = {
                case[k]
                for k in (
                    "history",
                    "observed_initial",
                    "observed_surface",
                    "llc_initial",
                    "llc_interior",
                    "llc_surface",
                )
            } | set(case["prediction"].values())
            if len(required) != 9 or set(proof["files"]) != required:
                raise ValueError("Missing exported fields")
            assert case["id"] == receipt.parent.name
            for name, info in proof["files"].items():
                path = directory / name
                if sha(path) != info["sha256"]:
                    raise ValueError("Export checksum differs: " + name)
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                assert (
                    list(array.shape) == info["shape"]
                    and str(array.dtype) == info["dtype"]
                )
            grid_path = receipt.parent / "grid.npz"
            assert sha(grid_path) == proof["grid_sha256"]
            grid = dict(np.load(grid_path, allow_pickle=False))
            names = (
                grid["names"].tolist()
                if case["source"] == "llc"
                else ["thetao_0", "zos"]
            )
            check_scores(case["validation"], len(names))
            for hour, methods in case["validation"].items():
                for method, scores in methods.items():
                    for metric in ("mse_normalized", "mse_physical"):
                        for name, value in zip(names, scores[metric], strict=True):
                            key = f"snapshot_val/{case['source']}/{method}/{hour}h/{name}/{metric}"
                            metrics.setdefault(key, []).append(value)
            cases[case["id"]] = dict(
                receipt=str(receipt.relative_to(root)), sha256=sha(receipt)
            )
        assert all(identity == identities[0] for identity in identities)
        for source in ("llc", "obs"):
            assert sum(key.startswith(source + "-") for key in cases) == origins
        assert all(len(values) == origins for values in metrics.values())
        reduced = {k: float(np.mean(v)) for k, v in metrics.items()}
        for key in list(reduced):
            if key.endswith("/mse_physical"):
                reduced[key.removesuffix("mse_physical") + "rmse"] = (
                    reduced.pop(key) ** 0.5
                )
        reference = [
            row
            for row in logged
            if row.get("updates") == update
            and "snapshot_val/obs/normalized_rmse/channel_mean" in row
        ]
        # A short qualification uses a subset; production must reproduce all eight origins.
        comparison = None
        if origins == cfg["validation_examples"]:
            if not reference:
                raise ValueError("No saved training validation for exported checkpoint")
            for key, value in reduced.items():
                np.testing.assert_allclose(
                    value, reference[-1][key], rtol=2e-4, atol=2e-7, err_msg=key
                )
            comparison = dict(
                metrics=len(reduced),
                rtol=2e-4,
                atol=2e-7,
                training_metrics_sha256=sha(train / "metrics.jsonl"),
            )
        checkpoints[str(update)] = dict(
            identity=identities[0],
            cases=cases,
            validation=reduced,
            training_comparison=comparison,
        )
    value = dict(
        schema_version=1,
        checkpoints=checkpoints,
        verification_script_sha256=sha(__file__),
    )
    temp = root / "COMPLETE.tmp"
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(root / "COMPLETE.json")
    print(json.dumps(dict(event="verified", checkpoints=list(checkpoints))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--export-root", type=Path, required=True)
    p.add_argument("--updates", type=int, nargs="+", default=[10000, 11000])
    p.add_argument("--origins", type=int, default=8)
    p.add_argument("--hours", type=int, default=720)
    a = p.parse_args()
    verify(a.run_root, a.export_root, a.updates, a.origins, a.hours)
