#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit, qualify and evaluate frozen pilot endpoints for continuous years."""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.monthly_annual import (
    fixed_ohc_support,
    lead_metrics,
    monthly_ohc_metrics,
    rollout,
)
from samudra.experiments.monthly_forcing_adapter import ModelForcingAdapter
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_annual import read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def audit(config, root):
    contracts = {
        str(p): digest(p)
        for p in [
            Path(config["samples"]) / n
            for n in ["grid.npz", "statistics.npz", "SHA256SUMS"]
        ]
    }
    annual = Path(config["annual"])
    for origin in config["origins"]:
        directory = annual / "test" / origin
        desc, raw = read_origin(directory)
        if not np.isfinite(raw["atmosphere"]).all():
            raise ValueError("Nonfinite annual atmosphere")
        with np.load(
            Path(config["samples"]) / "test" / (origin[:7] + ".npz")
        ) as monthly:
            n = len(monthly["month_weights"]) + 19
            for k in ["surface", "velocity", "atmosphere"]:
                np.testing.assert_array_equal(raw[k][:n], monthly[k])
            np.testing.assert_array_equal(raw["midpoint"][:n], monthly["midpoints"])
        for record in desc["records"] + desc["monthly_interiors"]:
            path = directory / record["file"]
            if digest(path) != record["sha256"]:
                raise ValueError("Annual payload changed")
            if record in desc["monthly_interiors"]:
                with (
                    np.load(path) as z,
                    np.load(
                        Path(config["samples"]) / "test" / (path.name[:7] + ".npz")
                    ) as m,
                ):
                    np.testing.assert_array_equal(z["ohc"], m["ohc"])
        contracts[str(directory / "COMPLETE.json")] = digest(
            directory / "COMPLETE.json"
        )
    for row in config["models"]:
        path = Path(row["checkpoint"])
        contracts[str(path)] = digest(path)
        complete = json.loads((path.parent / "TRAIN_COMPLETE.json").read_text())
        if complete["step"] != 3000:
            raise ValueError("Model incomplete")
    if config.get("prior_audit"):
        prior = json.loads(Path(config["prior_audit"]).read_text())
        if prior["contracts"] != contracts:
            raise ValueError(
                "Recovery source/data hashes differ from the failed attempt"
            )
    reuse = config.get("rollout_reuse")
    if reuse:
        prior_root = Path(reuse["root"])
        for name, expected in reuse["metadata"].items():
            if digest(prior_root / name) != expected:
                raise ValueError("Reused rollout provenance changed")
        qualified = json.loads((prior_root / "QUALIFIED.json").read_text())
        if (
            qualified["producer"] != reuse["producer"]
            or qualified["first_month_readout_exact"] is not True
            or qualified["audit_sha256"] != digest(prior_root / "AUDITED.json")
        ):
            raise ValueError("Reused producer was not qualified")
        prior_config = json.loads((prior_root / "paths.json").read_text())
        if (
            prior_config["models"] != config["models"]
            or prior_config["origins"] != config["origins"]
        ):
            raise ValueError("Reused model/origin cohort differs")
        for name, expected in reuse["files"].items():
            if digest(prior_root / name) != expected:
                raise ValueError("Reused rollout hash changed")
    atomic_json(
        dict(config_sha256=digest(root / "paths.json"), contracts=contracts),
        root / "AUDITED.json",
    )


def load_model(row):
    saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    if saved["step"] != 3000:
        raise ValueError("Expected fixed cooled 3k endpoint")
    model = MonthlyLatentOcean().cuda().eval()
    if row["adapter"]:
        model.add_module("model_forcing_adapter", ModelForcingAdapter().cuda())
    model.load_state_dict(saved["ema"], strict=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=["audit", "qualify", "evaluate"], required=True)
    p.add_argument("--index", type=int)
    a = p.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    if os.environ["SAMUDRA_CODE_COMMIT"] != config["producer"]:
        raise ValueError("Producer mismatch")
    if a.stage == "audit":
        audit(config, a.root)
        return
    proof = json.loads((a.root / "AUDITED.json").read_text())
    if proof["config_sha256"] != digest(a.root / "paths.json"):
        raise ValueError("Configuration changed after audit")
    for path, sha in proof["contracts"].items():
        if (
            path.endswith(("grid.npz", "statistics.npz", "SHA256SUMS", "COMPLETE.json"))
            and digest(path) != sha
        ):
            raise ValueError("Data contract changed")
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=32,
    )
    data.cache_prepared = True
    if a.stage == "qualify":
        model = load_model(config["models"][5])
        paths = [data.root / "test" / (o[:7] + ".npz") for o in config["origins"]]
        evaluate(
            model, data, paths, prediction_path=a.root / "qualification-reference.npz"
        )
        with np.load(a.root / "qualification-reference.npz") as z:
            reference_payload = dict(z)
        for i, origin in enumerate(config["origins"]):
            desc, raw = read_origin(Path(config["annual"]) / "test" / origin)
            result = rollout(
                model, data, desc, raw, origin_index=i, first_month_only=True
            )
            np.testing.assert_array_equal(
                result["surface"][:6], reference_payload["surface"][i]
            )
            np.testing.assert_array_equal(result["ohc"][0], reference_payload["ohc"][i])
        atomic_json(
            dict(
                producer=config["producer"],
                audit_sha256=digest(a.root / "AUDITED.json"),
                first_month_readout_exact=True,
            ),
            a.root / "QUALIFIED.json",
        )
        return
    qualified = json.loads((a.root / "QUALIFIED.json").read_text())
    if qualified["producer"] != config["producer"] or qualified[
        "audit_sha256"
    ] != digest(a.root / "AUDITED.json"):
        raise ValueError("Qualification contract differs")
    if a.index is None:
        raise ValueError("Model index required")
    row = config["models"][a.index]
    if digest(row["checkpoint"]) != proof["contracts"][row["checkpoint"]]:
        raise ValueError("Checkpoint changed")
    output = a.root / row["name"]
    output.mkdir(exist_ok=True)
    import wandb

    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="monthly-annual-" + row["name"],
        dir=str(output),
        config={
            **row,
            "producer": config["producer"],
            "protocol": "continuous-365-day-monthly-diffusion-v1",
        },
        mode="online",
    )
    if not run.url:
        raise ValueError("Online evaluation tracking unavailable")
    (output / "wandb-url.txt").write_text(run.url + "\n")
    reuse = config.get("rollout_reuse", {})
    model = None
    predictions = []
    refs = []
    pred_ohc = []
    ref_ohc = []
    climates = []
    climate_ohc = []
    for i, origin in enumerate(config["origins"]):
        desc, raw = read_origin(Path(config["annual"]) / "test" / origin)
        relative = row["name"] + "/" + origin + ".npz"
        reuse_sha = reuse.get("files", {}).get(relative)
        if reuse_sha:
            source = Path(reuse["root"]) / relative
            if digest(source) != reuse_sha:
                raise ValueError("Reused rollout changed after audit")
            with np.load(source) as saved:
                result = {
                    key: saved[key]
                    for key in ("surface", "ohc", "months", "latent_rms")
                }
                saved_reference = saved["reference_surface"]
                saved_ohc = saved["reference_ohc"]
            if (
                result["surface"].shape != (73, 2, 180, 360)
                or result["ohc"].shape != (12, 2, 180, 360)
                or result["latent_rms"].shape != (74,)
                or not np.isfinite(result["surface"]).all()
                or not np.isfinite(result["latent_rms"]).all()
            ):
                raise ValueError("Reused annual payload incomplete or nonfinite")
        else:
            if model is None:
                model = load_model(row)
            result = rollout(model, data, desc, raw, origin_index=i)
        reference = np.concatenate([raw["surface"][19:], raw["velocity"][19:]], 1)
        targets = []
        clim_ohc = []
        for record in desc["monthly_interiors"]:
            with np.load(
                Path(config["annual"]) / "test" / origin / record["file"]
            ) as z:
                targets.append(z["ohc"])
            sample = data.load(data.root / "test" / (record["file"][:7] + ".npz"))
            clim_ohc.append(data.climatology_fields(sample)["ohc"][0])
        if reuse_sha:
            np.testing.assert_array_equal(saved_reference, reference)
            np.testing.assert_array_equal(saved_ohc, np.stack(targets))
            np.testing.assert_array_equal(
                result["months"],
                [record["file"][:7] for record in desc["monthly_interiors"]],
            )
            shutil.copyfile(source, output / (origin + ".npz"))
            if digest(output / (origin + ".npz")) != reuse_sha:
                raise ValueError("Reused copy differs")
        else:
            np.savez_compressed(
                output / (origin + ".npz"),
                **result,
                reference_surface=reference,
                reference_ohc=np.stack(targets),
            )
        predictions.append(result["surface"])
        refs.append(reference)
        pred_ohc.append(result["ohc"])
        ref_ohc.append(np.stack(targets))
        climate_ohc.append(np.stack(clim_ohc))
        months = pd.DatetimeIndex(raw["midpoint"][19:]).month.to_numpy() - 1
        climates.append(data.stats["surface_climatology"][months])
        print(json.dumps(dict(origin=origin, complete=True)), flush=True)
    lead = lead_metrics(np.stack(predictions), np.stack(refs), data)
    baseline = lead_metrics(np.stack(climates), np.stack(refs), data)
    support = fixed_ohc_support(data.grid)
    ohc = monthly_ohc_metrics(
        np.stack(pred_ohc),
        np.stack(ref_ohc),
        np.stack(climate_ohc),
        data.ohc_area.cpu().numpy(),
        support,
    )
    for row_lead, control in zip(lead, baseline, strict=True):
        row_lead["surface_blend"] = float(
            np.mean(
                [
                    row_lead[k] / control[k]
                    for k in ["sst_rmse", "velocity_rmse", "eke_rmse"]
                ]
            )
        )
        run.log(row_lead, step=row_lead["day"])
    atomic_json(
        dict(
            leads=lead,
            climatology_leads=baseline,
            monthly_ohc=ohc,
            fixed_ohc_support_cells=support.sum((1, 2)).tolist(),
            reused_rollouts={
                key: value
                for key, value in reuse.get("files", {}).items()
                if key.startswith(row["name"] + "/")
            },
            eke_definition="Across the three origins at each fixed lead, matching short-horizon aggregation axis",
            checkpoint_sha256=proof["contracts"][row["checkpoint"]],
            producer=config["producer"],
        ),
        output / "metrics.json",
    )
    outputs = {f.name: digest(f) for f in output.glob("*.npz")}
    outputs["metrics.json"] = digest(output / "metrics.json")
    atomic_json(
        dict(
            outputs=outputs,
            origins=config["origins"],
            model=row,
            producer=config["producer"],
        ),
        output / "COMPLETE.json",
    )
    run.finish()


if __name__ == "__main__":
    main()
