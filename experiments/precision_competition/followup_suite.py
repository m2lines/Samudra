# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Predeclared qualification selection and bounded competition/recovery manifests."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .benchmark import Fields, atomic_json
from .followup import FollowupConfig

SEEDS = (11, 22, 33)


def entry(config):
    return {"id": config.run_id, "config": asdict(config)}


def scales():
    """Fixed CPU calibration stream, disjoint from training/validation/test."""
    torch.set_num_threads(4)
    result = {}
    for task in ("transport", "evolution"):
        source = Fields(FollowupConfig(auxiliary=task), "cpu")
        generator = source.generator(660001)
        total = 0.0
        for _ in range(128):
            total += source.draw(64, generator)["auxiliary"].square().sum().item()
        result[task] = total / 8192
    result["main"] = source.scale**2
    return result


def qualification():
    scale = scales()
    base = FollowupConfig(steps=12000, eval_every=100, train_seconds=900)
    entries = []
    for seed in SEEDS:
        for lr in (3e-4, 3e-5):
            for schedule in ("constant", "cosine"):
                entries.append(
                    entry(replace(base, seed=seed, lr=lr, schedule=schedule))
                )
        for auxiliary in ("transport", "evolution"):
            for schedule in ("constant", "cosine"):
                for normalized in (False, True):
                    entries.append(
                        entry(
                            replace(
                                base,
                                seed=seed,
                                auxiliary=auxiliary,
                                schedule=schedule,
                                main_weight=0.0,
                                auxiliary_weight=1.0,
                                aux_loss_scale=scale[auxiliary] if normalized else 1.0,
                            )
                        )
                    )
    return entries, scale


def load_records(manifest, results):
    records = []
    for row in json.loads(Path(manifest).read_text()):
        folder = Path(results) / row["id"]
        value = json.loads((folder / "result.json").read_text())
        if value["step"] != value["config"]["steps"]:
            raise ValueError(f"Incomplete training budget: {row['id']}")
        if value["config"] != row["config"]:
            raise ValueError("Result does not match manifest")
        history = json.loads((folder / "history.json").read_text())
        records.append((value, history, folder))
    return records


def select_recipe(records, auxiliary=False):
    groups: dict[tuple, list[tuple[float, Any, Any, Any]]] = {}
    key: tuple
    for value, history, folder in records:
        c = FollowupConfig(**value["config"])
        if auxiliary:
            if c.main_weight or c.auxiliary != "transport":
                continue
            score = max(row["auxiliary_normalized_rmse"] for row in history[-3:])
            key = (c.lr, c.schedule, c.aux_loss_scale)
        else:
            if not c.main_weight or c.auxiliary_weight:
                continue
            score = -min(row["bits"] for row in history[-3:])
            key = (c.lr, c.schedule)
        groups.setdefault(key, []).append((score, value, history, folder))
    eligible = [
        rows
        for rows in groups.values()
        if sorted(v[1]["config"]["seed"] for v in rows) == list(SEEDS)
    ]
    if not eligible:
        raise ValueError("Missing seed-matched qualification recipes")
    # Optimize worst seed's last-three-validation score. No test/EMA selection.
    return min(
        eligible,
        key=lambda rows: (max(row[0] for row in rows), str(rows[0][1]["config"])),
    )


def checkpoint_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def competition(manifest, results, calibration):
    records = load_records(manifest, results)
    solo = select_recipe(records)
    aux = select_recipe(records, auxiliary=True)
    decision = {
        "selected_solo": [r[1]["id"] for r in solo],
        "selected_auxiliary": [r[1]["id"] for r in aux],
        "worst_solo_bits_last_three": -max(r[0] for r in solo),
        "worst_aux_nrmse_last_three": max(r[0] for r in aux),
        "selection": "worst-seed last-three raw validation; test and EMA excluded",
    }
    if (
        decision["worst_solo_bits_last_three"] < 6
        or decision["worst_aux_nrmse_last_three"] >= 0.8
    ):
        return [], {**decision, "gate": "failed; no competition submitted"}
    aux_scale = aux[0][1]["config"]["aux_loss_scale"]
    main_scale = calibration["main"] if aux_scale != 1 else 1.0
    entries = []
    available = []
    for bits in (4, 6, 8):
        if not all((r[3] / f"bits-{bits}.pt").exists() for r in solo):
            continue
        available.append(bits)
        for _, value, _, folder in solo:
            path = folder / f"bits-{bits}.pt"
            base = replace(
                FollowupConfig(**value["config"]),
                stage="competition",
                steps=3000,
                train_seconds=300,
                lr=3e-5,
                schedule="cosine",
                parent_checkpoint=str(path.resolve()),
                parent_sha256=checkpoint_digest(path),
                main_loss_scale=main_scale,
                aux_loss_scale=aux_scale,
                pair_id=f"bits{bits}-seed{value['config']['seed']}",
            )
            entries.append(entry(base))
            for relation in ("independent", "dependent"):
                for weight in (0.1, 1.0, 10.0):
                    entries.append(
                        entry(replace(base, relation=relation, auxiliary_weight=weight))
                    )
    return entries, {
        **decision,
        "gate": "passed",
        "precision_levels": available,
        "count": len(entries),
        "main_loss_scale": main_scale,
        "aux_loss_scale": aux_scale,
    }


def recovery(manifest, results):
    records = load_records(manifest, results)
    entries = []
    for value, _, folder in records:
        c = FollowupConfig(**value["config"])
        if c.auxiliary_weight not in (0.0, 1.0):
            continue
        path = folder / "latest.pt"
        base = replace(
            c,
            stage="recovery",
            parent_checkpoint=str(path.resolve()),
            parent_sha256=checkpoint_digest(path),
            pair_id=c.pair_id + f"-from-{c.relation}-{c.auxiliary_weight}",
        )
        entries.append(entry(base))
        if c.auxiliary_weight:
            entries.append(entry(replace(base, auxiliary_weight=0.0)))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["qualification", "competition", "recovery"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "qualification":
        entries, calibration = qualification()
        atomic_json(args.output.with_suffix(".scales.json"), calibration)
    elif args.stage == "competition":
        entries, decision = competition(
            args.manifest, args.results, json.loads(args.calibration.read_text())
        )
        atomic_json(args.output.with_suffix(".decision.json"), decision)
    else:
        entries = recovery(args.manifest, args.results)
    atomic_json(args.output, entries)
    print(
        json.dumps(
            {"stage": args.stage, "runs": len(entries), "manifest": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
