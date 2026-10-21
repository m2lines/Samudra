#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify model-only exposure and compare monthly adaptation curves."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    normalization = json.loads(args.normalization.read_text())["normalization"]
    names = dict(
        A="P-OM4-surface",
        B="P-OM4-multitask",
        C="P-mix-surface",
        D="P-mix-multitask",
    )
    curves, endpoints, checksums = {}, {}, {}
    origins = None
    initial_hashes = set()

    def read(path):
        checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for original, arms in [(True, "ABCDE"), (False, "ABCD")]:
        for arm in arms:
            name = arm if original else names[arm]
            directory = (
                args.original / arm if original else args.root / "production" / arm
            )
            event_path = directory / "events.jsonl"
            checksums[str(event_path)] = hashlib.sha256(
                event_path.read_bytes()
            ).hexdigest()
            events = [json.loads(s) for s in event_path.read_text().splitlines()]
            if [e["step"] for e in events] != list(range(1, 3001)):
                raise ValueError("Incomplete optimizer stream: " + name)
            if not all(
                np.isfinite(e["loss"]) and np.isfinite(e["gradient_norm"])
                for e in events
            ):
                raise ValueError("Nonfinite training: " + name)
            complete = read(directory / "TRAIN_COMPLETE.json")
            if complete["step"] != 3000 or complete["producer"] != (
                "1c9817d44" if original else "a1e74e188"
            ):
                raise ValueError("Wrong endpoint producer")
            if not original:
                counts = Counter(e["source"] for e in events)
                expected = dict(
                    obs=1000,
                    om4=2000 if arm in "AB" else 1000,
                    cm4=0 if arm in "AB" else 1000,
                )
                if counts != Counter(expected) or complete["counts"] != expected:
                    raise ValueError("Wrong model/observation exposure")
                if any(e["source"] == "obs" for e in events[:2000]) or any(
                    e["source"] != "obs" for e in events[2000:]
                ):
                    raise ValueError("Observations appeared outside cooldown")
                proof = read(args.root / "qualification" / arm / "QUALIFIED.json")
                if (
                    proof["producer"] != complete["producer"]
                    or not proof["exact_observation_cooldown_resume"]
                ):
                    raise ValueError("Missing adaptation qualification")
                initial_hashes.add(proof["initial_backbone_sha256"])
            records = []
            for event in events:
                if "validation_rmse_blend" not in event:
                    continue
                payload = read(directory / f"validation-{event['step']:05d}.json")
                if payload["members"] != 8 or payload["sampler_steps"] != 32:
                    raise ValueError("Changed diffusion readout")
                if origins is None:
                    origins = payload["origins"]
                if payload["origins"] != origins or len(origins) != 9:
                    raise ValueError("Changed validation cohort")
                metrics = payload["metrics"]
                blend = np.mean([metrics[k] / v for k, v in normalization.items()])
                if not np.isclose(
                    blend, event["validation_rmse_blend"], atol=1e-12, rtol=0
                ):
                    raise ValueError("Changed metric normalization")
                records.append(
                    dict(
                        step=event["step"],
                        blend=float(blend),
                        surface_blend=float(
                            np.mean(
                                [
                                    metrics[k] / normalization[k]
                                    for k in ["sst_rmse", "velocity_rmse", "eke_rmse"]
                                ]
                            )
                        ),
                        metrics=metrics,
                    )
                )
            if len(records) != 12:
                raise ValueError("Incomplete validation curve")
            eligible = records if original else [r for r in records if r["step"] > 2000]
            selected = min(eligible, key=lambda r: r["blend"])
            if not np.isclose(selected["blend"], complete["best"], atol=1e-12, rtol=0):
                raise ValueError("Wrong checkpoint-selection scope")
            raw_path = directory / "validation-raw-03000.json"
            raw = read(raw_path)["metrics"] if raw_path.exists() else None
            if raw is None and not original:
                raise ValueError("Missing new-run raw endpoint evaluation")
            curves[name] = records
            endpoints[name] = dict(
                **records[-1],
                counts=complete["counts"],
                selected_step=selected["step"],
                selected_blend=selected["blend"],
                raw_blend=(
                    float(np.mean([raw[k] / v for k, v in normalization.items()]))
                    if raw is not None
                    else None
                ),
                wandb=(directory / "wandb-url.txt").read_text().strip(),
            )
    if len(initial_hashes) != 1:
        raise ValueError("Different random backbone initializations")
    colors = dict(zip("ABCDE", ["C0", "C1", "C2", "C3", "C4"], strict=True))
    fields = ["blend", "surface_blend", "ohc_0_700_rmse", "ohc_700_2000_rmse"]
    titles = [
        "Five-component RMSE blend",
        "Surface-only RMSE blend",
        "Monthly OHC 0–700 m RMSE (GJ/m²)",
        "Monthly OHC 700–2,000 m RMSE (GJ/m²)",
    ]
    for filename, start in [
        ("validation-full.png", 250),
        ("validation-cooldown.png", 2250),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, field, title in zip(axes.flat, fields, titles, strict=True):
            for original, arms in [(True, "ABCDE"), (False, "ABCD")]:
                for arm in arms:
                    name = arm if original else names[arm]
                    records = [r for r in curves[name] if r["step"] >= start]
                    ax.plot(
                        [r["step"] for r in records],
                        [
                            r[field] if field in r else r["metrics"][field] / 1e9
                            for r in records
                        ],
                        color=colors[arm],
                        label="Joint " + arm
                        if original and arm != "E"
                        else ("Obs-only E" if original else name),
                        linestyle="--" if original else "-",
                        marker=".",
                    )
            ax.set_title(title)
            ax.set_xlabel("Optimizer updates; obs-only cooldown starts at 2k")
            ax.set_xlim(start, 3000)
            ax.axvline(2000, color="gray", linewidth=0.7)
            if start == 250:
                ax.set_yscale("log")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        target = args.out / filename
        fig.savefig(target, dpi=140)
        plt.close(fig)
        with Image.open(target) as saved:
            saved.convert("RGB").quantize(colors=256).save(target, optimize=True)
    checksums.update(read(args.root / "ENDPOINT_SHA256.json"))
    checksums["common initial backbone tensor digest"] = initial_hashes.pop()
    (args.out / "checksums.txt").write_text(
        "".join(
            f"{sha}  {path}  # pragma: allowlist secret\n"
            for path, sha in sorted(checksums.items())
        )
    )
    result = dict(
        normalization=normalization,
        origins=origins,
        initial_backbone_identity_verified=True,
        curves=curves,
        endpoints=endpoints,
        qualification_gpu_hours=1445 / 3600,
        production_gpu_hours=36173 * 4 / 3600,
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, endpoint in endpoints.items():
        print(name, json.dumps(endpoint))


if __name__ == "__main__":
    main()
