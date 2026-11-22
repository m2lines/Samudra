# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU matched-map comparison of the small regressor and existing classifiers.

Depends on the sibling soft-priors-25k renderer and its pre-spike-noise helpers.
Inputs are completed physical-field NPZ maps; this script performs no inference.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "soft-priors-25k/render_maps.py"
_spec = importlib.util.spec_from_file_location("soft_prior_map_renderer", HELPER)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
PRIOR = Path(
    "/home/jder/.local/state/codex-async-followup/workflows/"
    "skyline-soft-priors-25k-artifacts/maps"
)
base.LABELS.update(
    {
        "new-regression": "Small all-fields regression",
        "new-all-fields": "Small all-fields classifier",
        "new-temperature": "Small T-only classifier",
    }
)
TEMPERATURE = (
    "gold",
    "new-regression",
    "new-all-fields",
    "new-temperature",
    "original-temperature-regression",
    "original-all-fields-regression",
)
CLASSIFIERS = (
    "gold",
    "new-regression",
    "new-all-fields",
    "new-temperature",
    "original-temperature-classifier",
    "original-all-fields-classifier",
)
AUXILIARY = (
    "gold",
    "new-regression",
    "new-all-fields",
    "original-all-fields-regression",
    "persistence",
)


def load_patches(root, prior, cache, regression):
    """Join exactly the prior patches, retaining their spectral tile selection."""
    receipt = json.loads((root / "new-regression-receipt.json").read_text())
    if (receipt["checkpoint_step"], receipt["weights_kind"], receipt["parameters"]) != (
        25000,
        "ema",
        14321665,
    ):
        raise ValueError("Expected final EMA 25k small regression checkpoint")
    if receipt["architecture"]["objective"] != "regression":
        raise ValueError("Expected continuous regression objective")
    groups = base.load_patches(prior, cache, regression)
    tasks = {p["task"] for group in groups for p in group}
    expected = {f"new-regression-{task}.npz" for task in tasks}
    found = {p.name for p in root.glob("new-regression-*.npz")}
    if found != expected or len(expected) != 11:
        raise ValueError(
            f"Expected 11 regressor maps; missing={expected - found}, extra={found - expected}"
        )
    for group in groups:
        for patch in group:
            path = root / f"new-regression-{patch['task']}.npz"
            values = base.load_npz(path)
            base.validate_map(values, patch["field"], path)
            base.same(
                values["checkpoint_sha256"],
                receipt["checkpoint_sha256"],
                f"{path}/checkpoint",
            )
            base.same(values["crop"], 17, f"{path}/crop")
            base.matched(patch["ref"], values, patch["field"], str(path))
            patch["data"]["new-regression"] = values[f"{patch['field']}_prediction"]
    if sum(map(len, groups)) != 15:
        raise ValueError("Expected 15 matched field grids")
    return groups


def prediction_kind(model):
    if model == "gold":
        return "OM4 target"
    if model == "persistence":
        return "persistence"
    if model in {"new-all-fields", "new-temperature"} or model.endswith("classifier"):
        return "discrete argmax-bin center"
    return (
        "scalar regression (BF16 output)"
        if model == "new-regression"
        else "scalar regression (historical)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-maps", type=Path, default=PRIOR)
    parser.add_argument("--classifier-cache", type=Path, default=base.CACHE)
    parser.add_argument("--regression-archive", type=Path, default=base.REGRESSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = load_patches(
        args.root, args.prior_maps, args.classifier_cache, args.regression_archive
    )
    patches = [p for group in groups for p in group]
    metrics = base.analyze(patches)
    for row in metrics:
        row["point_prediction_kind"] = prediction_kind(row["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    base.save_sources(patches, args.output)
    with (args.output / "map-metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for group, names, stem in (
        (groups[0], TEMPERATURE, "temperature-regression"),
        (groups[0], CLASSIFIERS, "temperature-classifier"),
        (groups[1], AUXILIARY, "velocity"),
        (groups[2], AUXILIARY, "salinity"),
    ):
        for errors in (False, True):
            base.draw(group, names, metrics, args.output, stem, errors)
    # Hash all newly inferred maps and inherited input artifacts used by the join.
    inputs = sorted(args.root.glob("new-regression-*.npz"))
    inputs += sorted(args.prior_maps.glob("*.npz"))
    inputs += sorted(args.classifier_cache.glob("points-4273-*.npz"))
    inputs += [
        args.regression_archive,
        HELPER,
        HERE.parent / "pre-spike-noise/render.py",
    ]
    receipt = {
        "new_map_files": 11,
        "new_field_grids": 15,
        "metric_rows": len(metrics),
        "inputs_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "regression_receipt": json.loads(
            (args.root / "new-regression-receipt.json").read_text()
        ),
        "gold_atol_physical": 2e-5,
        "maps": "Native 64x64 pixels, region-major; common row limits; context boxes 17 or 33 cells.",
        "spectra": "Identical inherited fully wet 64/32 tiles, planar detrend, periodic Hann, summed radial power 0.25..0.5 cycles/cell. Excess power does not identify noise causally.",
        "categorical_readout": "Argmax-bin center only; no continuous expectation or regression NLL.",
        "baseline_limits": "Original models differ in size, context and training. Cached Samudra is T only, included in metrics but excluded from panels and color ranges.",
    }
    (args.output / "map-render-receipt.json.gz").write_bytes(
        gzip.compress((json.dumps(receipt, indent=2) + "\n").encode(), mtime=0)
    )


if __name__ == "__main__":
    main()
