#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-cooldown A-replay forecasts and source-matched OM4 initializations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_annual import require_endpoint_step, rollout
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_annual import (
    annual_indices,
    model_annual_payload,
    model_batch,
)
from samudra.experiments.monthly_model_data import ModelMonths
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_annual import read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def saved_model(row, device="cuda"):
    saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    require_endpoint_step(saved, row)
    if row["name"] == "A-replay-pre-cooldown":
        assert saved["counts"] == {"om4": 1000, "obs": 1000, "cm4": 0}
        assert saved["protocol"]["mode"] == "replay"
        assert saved["protocol"]["seed"] == 1729
        assert saved["protocol"]["main_end"] == saved["step"] == 2000
    model = MonthlyLatentOcean().to(device).eval()
    model.load_state_dict(saved["ema"], strict=True)
    return model


def write_npz(path, values):
    np.savez_compressed(path, **values)
    with np.load(path) as saved:
        for key, value in values.items():
            np.testing.assert_array_equal(saved[key], value)


def prepare(config, root):
    data = Samples(
        config["samples"], "cpu", surface_fill="zero", global_observations=True
    )
    om4 = Path(config["om4"])
    reader = ModelMonths(om4 / "OM4.zarr", "om4", data.grid, om4)
    contracts = {
        str(path): digest(path)
        for path in [
            Path(config["samples"]) / "grid.npz",
            Path(config["samples"]) / "statistics.npz",
            Path(config["samples"]) / "SHA256SUMS",
            om4 / "OM4.zarr/.zmetadata",
            om4 / "OM4_means.zarr/.zmetadata",
            om4 / "OM4_stds.zarr/.zmetadata",
        ]
    }
    for row in [config["model"], *config["comparison_models"]]:
        contracts[row["checkpoint"]] = digest(row["checkpoint"])
    # Check the fixed pre-cooldown checkpoint, not its later TRAIN_COMPLETE step.
    model = saved_model(config["model"], "cpu")
    del model
    prepared = root / "prepared"
    prepared.mkdir()
    for origin in config["om4_origins"]:
        payload = model_annual_payload(reader, data, origin)
        path = prepared / (origin + ".npz")
        write_npz(path, payload)
        contracts[str(path)] = digest(path)
        print(
            json.dumps(
                dict(
                    prepared_om4=origin,
                    first=str(payload["start"][0]),
                    last=str(payload["start"][-1]),
                )
            ),
            flush=True,
        )
    exclusions = {}
    for origin in config.get("excluded_om4_origins", []):
        try:
            annual_indices(reader.bounds, origin)
        except ValueError as error:
            exclusions[origin] = str(error)
        else:
            raise ValueError(
                "Expected source gap is no longer present; review the origin cohort"
            )
    for origin in config["obs_origins"]:
        path = Path(config["annual"]) / "test" / origin
        read_origin(path)  # Verifies every record hash and exact timestamps.
        contracts[str(path / "COMPLETE.json")] = digest(path / "COMPLETE.json")
    atomic_json(
        dict(
            config_sha256=digest(root / "paths.json"),
            contracts=contracts,
            excluded_om4_origins=exclusions,
            own_model_forcing=True,
            initialized_from_surface_history_only=True,
            checkpoint_policy="fixed main endpoint EMA before cooldown reset",
        ),
        root / "AUDITED.json",
    )


def source_case(config, root, data, source, origin):
    if source == "obs":
        description, raw = read_origin(Path(config["annual"]) / "test" / origin)
        return description, raw, None, None
    with np.load(root / "prepared" / (origin + ".npz")) as saved:
        raw = dict(saved)
    description = dict(
        origin=origin,
        source="om4",
        calendar="julian",
        monthly_interiors=[dict(file=str(m)) for m in raw["months"]],
    )
    return description, raw, model_batch(data, raw), raw


@torch.no_grad()
def initial_readout(model, data, batch, origin_index):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = model.initialize(
            batch["fields"][:, :19],
            batch["available"][:, :19],
            batch["forcing"][:, :19],
            batch["contexts"][:, 18],
        )
        draws = model.decoder.sample(
            state.expand(8, -1, -1, -1, -1),
            data.mask.bool(),
            torch.Generator(device=data.device).manual_seed(905001 + origin_index),
            steps=32,
            position_days=-2.5,
        )
    slots = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
    result = {}
    for prefix, value in [("", draws.mean(0, keepdim=True)), ("member_", draws[:1])]:
        physical = data.physical(value[:, None])[0, 0].cpu().numpy()
        result[prefix + "initial_interior"] = physical[slots]
        result[prefix + "initial_surface"] = physical[[38, 76]]
        result[prefix + "initial_ohc"] = data.ohc(value)[0]
    return result


def run_case(model, data, config, root, source, origin, index, *, first=False):
    desc, raw, batch, prepared = source_case(config, root, data, source, origin)
    result = rollout(
        model,
        data,
        desc,
        raw,
        origin_index=index,
        first_month_only=first,
        export_profiles=True,
        export_member=0,
        model_batch=batch,
    )
    if source == "obs":
        result["reference_surface"] = np.concatenate(
            [raw["surface"][19:], raw["velocity"][19:]], 1
        )
        reference_temperature, reference_ohc = [], []
        for month in result["months"]:
            sample = data.load(data.root / "test" / (str(month) + ".npz"))
            reference_temperature.append(
                sample["raw"]["interior"].reshape(2, 14, *model.shape)[0]
            )
            reference_ohc.append(sample["raw"]["ohc"])
        result["reference_temperature"] = np.stack(reference_temperature)
        result["reference_ohc"] = np.stack(reference_ohc)
    else:
        assert prepared is not None
        for key in (
            "reference_surface",
            "reference_temperature",
            "reference_ohc",
            "reference_initial",
        ):
            result[key] = prepared[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["prepare", "qualify", "forecast", "initializations"],
        required=True,
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    config = json.loads((args.root / "paths.json").read_text())
    assert config["producer"] == os.environ["SAMUDRA_CODE_COMMIT"]
    torch.set_num_threads(8)
    if args.stage == "prepare":
        prepare(config, args.root)
        return
    audit = json.loads((args.root / "AUDITED.json").read_text())
    assert audit["config_sha256"] == digest(args.root / "paths.json")
    for path, sha in audit["contracts"].items():
        assert digest(path) == sha, path
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
    if args.stage == "qualify":
        model = saved_model(config["model"])
        reference_path = args.root / "qualification-monthly.npz"
        evaluate(
            model,
            data,
            [data.root / "test" / (config["obs_origins"][0][:7] + ".npz")],
            prediction_path=reference_path,
        )
        for source, origins in [
            ("obs", config["obs_origins"]),
            ("om4", config["om4_origins"]),
        ]:
            origin = origins[0]
            result = run_case(
                model, data, config, args.root, source, origin, 0, first=True
            )
            desc, raw, batch, _ = source_case(config, args.root, data, source, origin)
            control = rollout(
                model,
                data,
                desc,
                raw,
                origin_index=0,
                first_month_only=True,
                model_batch=batch,
            )
            for key in control:
                np.testing.assert_array_equal(control[key], result[key])
            if source == "obs":
                with np.load(reference_path) as reference:
                    np.testing.assert_array_equal(
                        result["surface"][:6], reference["surface"][0]
                    )
                    np.testing.assert_array_equal(result["ohc"][0], reference["ohc"][0])
            else:
                initial = initial_readout(model, data, batch, 0)
                for key, value in initial.items():
                    np.testing.assert_array_equal(result[key], value)
        atomic_json(
            dict(
                producer=config["producer"],
                audit_sha256=digest(args.root / "AUDITED.json"),
                paired_means_exact=True,
                observation_monthly_overlap_exact=True,
                om4_initialization_matches_annual=True,
                surface_only_initialization=True,
                retained_member=0,
            ),
            args.root / "QUALIFIED.json",
        )
        return
    qualified = json.loads((args.root / "QUALIFIED.json").read_text())
    assert qualified["audit_sha256"] == digest(args.root / "AUDITED.json")
    assert qualified["producer"] == config["producer"]
    import wandb

    output = args.root / (
        f"case-{args.index}" if args.stage == "forecast" else "om4-initializations"
    )
    output.mkdir()
    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="pre-cooldown-viewer-" + output.name,
        dir=str(output),
        config=dict(producer=config["producer"], inference_only=True, stage=args.stage),
        mode="online",
    )
    assert run.url
    files = {}
    if args.stage == "forecast":
        case = config["cases"][args.index]
        model = saved_model(config["model"])
        result = run_case(
            model,
            data,
            config,
            args.root,
            case["source"],
            case["origin"],
            case["source_index"],
        )
        path = output / "forecast.npz"
        write_npz(path, result)
        files[path.name] = digest(path)
    else:
        for row in config["comparison_models"]:
            model = saved_model(row)
            for i, origin in enumerate(config["om4_origins"]):
                _, _, batch, prepared = source_case(
                    config, args.root, data, "om4", origin
                )
                result = initial_readout(model, data, batch, i)
                result["reference_initial"] = prepared["reference_initial"]
                path = output / (row["name"] + "-" + origin + ".npz")
                write_npz(path, result)
                files[path.name] = digest(path)
            del model
            torch.cuda.empty_cache()
            print(json.dumps(dict(initializations_complete=row["name"])), flush=True)
    atomic_json(
        dict(
            producer=config["producer"],
            files=files,
            stage=args.stage,
            index=args.index,
            wandb_url=run.url,
            checkpoint_sha256=audit["contracts"][config["model"]["checkpoint"]],
        ),
        output / "COMPLETE.json",
    )
    run.finish()


if __name__ == "__main__":
    main()
