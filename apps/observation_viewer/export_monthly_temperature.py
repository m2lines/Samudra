#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Replay frozen annual forecasts offline and retain calendar-month temperatures.

Use the historical Samudra source and a training Python environment on PYTHONPATH.
The Panel service never imports that runtime or runs inference. Inputs may be a
verified local mirror of the original scratch tree. Checkpoints are unchanged;
GPU/runtime changes can alter predictions, so export coherent OHC alongside T
and record differences from the original report rather than claiming identity.
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from export_interior import sha


def export(contract_path, inputs_root, base, output):
    contract = json.loads(contract_path.read_text())
    meta = json.loads((base / "catalog.json").read_text())
    for name, digest in contract["module_sha256"].items():
        module = importlib.import_module(name)
        if sha(module.__file__) != digest:
            raise ValueError(f"Historical source mismatch: {name}")
    annual = importlib.import_module("samudra.experiments.observation_annual")
    checkpoint = importlib.import_module("samudra.experiments.observation_checkpoint")
    training = importlib.import_module("samudra.experiments.observation_training")
    models = importlib.import_module("samudra.experiments.observation_model")
    torch = importlib.import_module("torch")
    torch.set_num_threads(4)
    output.mkdir()  # Never replace an earlier export.
    receipt = dict(
        producer=contract["producer"],
        script_sha256=sha(__file__),
        contract_sha256=sha(contract_path),
        base_catalog_sha256=sha(base / "catalog.json"),
        runtime=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            autocast="bfloat16",
        ),
        files={},
        cases={},
        sources={},
    )

    def source(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
        receipt["sources"][str(path.relative_to(inputs_root))] = digest

    def write(name, **arrays):
        path = output / name
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as stored:
            for key, values in arrays.items():
                np.testing.assert_array_equal(stored[key], values)
        receipt["files"][name] = dict(sha256=sha(path), bytes=path.stat().st_size)

    for path, digest in contract["input_sha256"].items():
        source(inputs_root / path, digest)
    for model_key, spec in contract["models"].items():
        run = inputs_root / spec["run"]
        lineage = meta["models"][model_key]["lineage"]
        manifest = json.loads((run / "manifest.json").read_text())
        source(run / "manifest.json", lineage["training_manifest_sha256"])
        source(run / lineage["checkpoint"], lineage["checkpoint_sha256"])
        state, actual = checkpoint.load_fixed_checkpoint(
            run,
            lineage["checkpoint"],
            "test",
            lineage["task_counts"]["om4"],
            lineage["task_counts"]["observation"],
        )
        if actual != lineage:
            raise ValueError(f"Checkpoint lineage mismatch: {model_key}")
        options = manifest["arguments"]
        data = training.Samples(
            inputs_root / contract["observation_data"],
            "cuda",
            options["surface_fill"],
            global_observations=True,
        )
        if manifest["normalization_mode"] == "observation-only":
            data.use_observation_normalization()
        for axis in ["lat", "lon"]:
            np.testing.assert_array_equal(data.grid[axis], meta[axis])
        indices = [data.grid["names"].tolist().index(f"thetao_{i}") for i in range(14)]
        mask = data.grid["mask"][indices].astype(bool)
        np.testing.assert_array_equal(mask, np.load(base / meta["interior_mask"])[:, 0])
        heat_mask = np.load(base / meta["heat_mask"])
        model = (
            models.ObservationTransfer.from_arguments(
                data.grid["names"].tolist(), options
            )
            .cuda()
            .eval()
        )
        model.load_state_dict(state, strict=True)
        del state
        for origin in meta["origins"]:
            print(f"Replaying {model_key} / {origin}", flush=True)
            directory = inputs_root / contract["annual_data"] / "test" / origin
            description, raw = annual.read_origin(directory)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, initial = model.forecast(*annual.inputs(data, raw))
            if prediction.shape[1] != 73 or not torch.isfinite(prediction).all():
                raise ValueError("Incomplete or nonfinite annual forecast")
            temperatures, references, heats = [], [], []
            months: list[str] = []
            for record in description["monthly_interiors"]:
                path = directory / record["file"]
                if path.parent != directory:
                    raise ValueError("Invalid monthly target path")
                source(path, record["sha256"])
                month = path.name[:7]
                weights = data.tensor(annual.month_weights(raw["start"][19:], month))
                monthly = (prediction * weights[None, :, None, None, None]).sum(1)
                temperature = (
                    data.physical(monthly[:, None])[0, 0, indices].cpu().numpy()
                )
                temperatures.append(
                    np.where(mask, temperature, np.nan).astype(np.float32)
                )
                heats.append(data.ohc(monthly)[0] / 1e9)
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["depth"], meta["depths"])
                    if str(saved["time"]) != month:
                        raise ValueError("Monthly target timestamp mismatch")
                    references.append(
                        np.where(mask, saved["values"][0], np.nan).astype(np.float32)
                    )
                    old_reference = np.load(
                        base / meta["records"][origin]["heat_reference"]
                    )[len(months)]
                    np.testing.assert_array_equal(
                        np.where(
                            heat_mask, saved["ohc"].astype(np.float64) / 1e9, np.nan
                        ),
                        old_reference,
                    )
                months.append(month)
            if months != meta["records"][origin]["heat_months"]:
                raise ValueError("Calendar months differ from the report")
            temperature, reference, heat = map(
                np.array, [temperatures, references, heats]
            )
            if not np.isfinite(temperature[:, mask]).all():
                raise ValueError("Nonfinite wet temperature")
            old = meta["records"][origin]["models"][model_key]
            difference = (heat - np.load(base / old["heat"]))[:, heat_mask]
            surface = data.physical(prediction)[0, :, [38, 76]].cpu().numpy()
            surface_difference = surface - np.load(base / old["surface"])
            case = dict(
                lineage=lineage,
                heat_difference_from_report_gj_m2=dict(
                    rms=float(np.sqrt(np.mean(difference**2))),
                    max_abs=float(np.max(np.abs(difference))),
                ),
                surface_difference_from_report={
                    variable: dict(
                        rms=float(np.sqrt(np.nanmean(surface_difference[:, i] ** 2))),
                        max_abs=float(np.nanmax(np.abs(surface_difference[:, i]))),
                    )
                    for i, variable in enumerate(["sst_celsius", "ssh_meters"])
                },
            )
            filename = f"{model_key}-{origin}.npz"
            write(
                filename,
                temperature=temperature,
                reference=reference,
                heat=heat,
                months=np.array(months),
            )
            receipt["cases"][filename] = case
            print(json.dumps(case), flush=True)
            del prediction, initial
        del model, data
        torch.cuda.empty_cache()
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "contract.json").write_bytes(contract_path.read_bytes())
    (output / "export_monthly_temperature.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["contract", "inputs-root", "base", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.contract, args.inputs_root, args.base, args.output)
