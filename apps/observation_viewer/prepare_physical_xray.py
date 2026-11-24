# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable Xray arrays from completed physical-state evaluations."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(roots, output, base=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite nonempty bundle")
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    provenance: list[dict] = []
    grid = None
    if base is not None:
        base = Path(base)
        previous = json.loads((base / "COMPLETE.json").read_text())
        if previous["schema"] != "physical-xray-v1":
            raise ValueError("Unsupported base bundle")
        for name, expected in previous["sha256"].items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or digest(path) != expected:
                raise ValueError(f"Invalid base artifact: {name}")
            shutil.copyfile(path, output / name)
        for case in previous["cases"]:
            if (
                case["reference_label"] == "Observed surface input"
                and "copied where observed" not in case["reference_note"]
            ):
                case = dict(case)
                case["reference_note"] = (
                    "Initial surface values are copied where observed and generated outside observational coverage. "
                    + case["reference_note"]
                )
            cases.append(case)
        provenance.extend(previous["provenance"])
        with np.load(output / "grid.npz") as z:
            grid = dict(z)
    for root in map(Path, roots):
        proof = json.loads((root / "COMPLETE.json").read_text())
        for name, expected in proof["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError(f"Invalid evaluation artifact: {name}")
        provenance.append(
            {
                "evaluation": str(root),
                "receipt_sha256": digest(root / "COMPLETE.json"),
                "inputs": proof["inputs"],
            }
        )
        paths = sorted((root / "annual").glob("full-state-*.npz")) + sorted(
            (root / "om4").glob("full-state-*.npz")
        )
        if proof["inputs"].get("source") == "OM4":
            paths += sorted(root.glob("full-state-*.npz"))
        for path in paths:
            origin = path.stem.removeprefix("full-state-")
            with np.load(path) as z:
                data = dict(z)
            is_om4 = proof["inputs"].get("source") == "OM4" or path.parent.name == "om4"
            if is_om4:
                reference = data["gold"]
                initial_reference = data["gold_initial"]
            else:
                with np.load(path.parent / f"{origin}.npz") as z:
                    reference = z["reference"]
                with np.load(path.parent / f"maps-{origin}.npz") as z:
                    initial_reference = z["initial_surface_reference"]
            new_grid = {
                "names": data["channel_names"],
                "lat": data["lat"],
                "lon": data["lon"],
                "mean": data["normalization_mean"],
                "std": data["normalization_std"],
                "wet_mask": data["mask"],
            }
            if grid is None:
                grid = new_grid
                np.savez_compressed(output / "grid.npz", **grid)
            elif any(
                not np.array_equal(grid[k], v, equal_nan=k not in ["names"])
                for k, v in new_grid.items()
            ):
                raise ValueError("Evaluation grids or normalization differ")
            for index, lead in [(-1, 0)] + list(enumerate(data["leads_days"])):
                if (
                    not is_om4
                    and lead == 0
                    and any(
                        origin in c["label"]
                        and c["reference_label"] == "Observed surface input"
                        for c in cases
                    )
                ):
                    # Keep the already published initialization; the small first-step
                    # export can differ slightly across GPU architectures.
                    if proof["inputs"].get("checkpoint_sha256") not in {
                        item["inputs"].get("checkpoint_sha256")
                        for item in provenance[:-1]
                    }:
                        raise ValueError("Duplicate origin has different checkpoint")
                    continue
                stem = f"case-{len(cases):03d}.npz"
                prefix = "initial_" if lead == 0 else ""
                arrays = {
                    target: data[prefix + source][index]
                    for target, source in [
                        ("prediction_mean", "mean"),
                        ("prediction_member0", "draw1"),
                        ("spread", "spread"),
                    ]
                }
                gold = np.full_like(arrays["prediction_mean"], np.nan)
                if is_om4:
                    gold = initial_reference[-1] if lead == 0 else reference[index]
                else:
                    gold[[38, 76]] = (
                        initial_reference
                        if lead == 0
                        else reference[int(lead) // 5 - 1, :2]
                    )
                arrays["gold"] = gold
                arrays["gold_support"] = np.isfinite(gold) & grid["wet_mask"].astype(
                    bool
                )
                for key in ["prediction_mean", "prediction_member0", "gold"]:
                    arrays[key] = (arrays[key] - grid["mean"][:, None, None]) / grid[
                        "std"
                    ][:, None, None]
                arrays["spread"] /= grid["std"][:, None, None]
                np.savez_compressed(output / stem, **arrays)
                initialization_note = (
                    "Initial surface values are copied where observed and generated outside observational coverage. "
                    if lead == 0 and not is_om4
                    else ""
                )
                cases.append(
                    {
                        "file": stem,
                        "label": f"{'OM4' if is_om4 else proof['inputs'].get('annual_split', proof['inputs'].get('source', 'evaluation'))} · {origin} · {'initialization' if lead == 0 else f'day {lead}'}",
                        "reference_label": "OM4 aligned full-state target"
                        if is_om4
                        else (
                            "Observed surface input"
                            if lead == 0
                            else "Observed five-day surface mean"
                        ),
                        "reference_note": initialization_note
                        + (
                            f"OM4 surface-history initialization, with aligned native five-day OM4 interior targets at all depths. These are model-data targets, not ocean observations. Initialized SSH and SST are copied surface inputs, not reconstruction skill. Displayed state is the five-day bin from day {int(lead) - 5} to day {lead}. "
                            if is_om4
                            else "Surface references are aligned five-day observations. No instantaneous interior reference is available; interior errors remain unavailable. "
                        )
                        + "Member 1 feeds its own sampled physical state forward. This does not establish temporal realism: the pilot has excessive short-term member variability.",
                    }
                )
    if not cases:
        raise ValueError("No completed annual full-state exports found")
    manifest = {
        "schema": "physical-xray-v1",
        "cases": cases,
        "provenance": provenance,
        "model_description": "Two Samudra ConvNeXt U-Nets diffuse the initial and next ocean states. Each member's sampled physical state feeds its next processor step; there is no diffusion decoder or deterministic latent trajectory. Shared ERA5 forcing, 77 physical channels, effective batch 8. This pilot has 600 OM4 and 600 observation updates, including a final 100-update observation-only cooldown. Displayed fields are unsmoothed. Full evaluation checkpoint, EMA selection, ensemble size, and sampling settings are recorded below. Initialization is conditioned on surface history. Available surface inputs are copied into the initialized surface state; unobserved wet cells are generated by the initializer, including the northernmost SSH rows absent from observational coverage. OM4 training supplies SSH over every wet surface cell and excludes those known cells from the initializer denoising loss; it does not directly train missing-SSH reconstruction for the observational coverage gap. Forecast states are sampled by the processor. An interior field's existence does not imply observational supervision or aligned interior truth.",
        "sha256": {p.name: digest(p) for p in output.glob("*.npz")},
    }
    (output / "COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    prepare(args.evaluation, args.output, args.base)
