#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export frozen annual profiles, requiring exact replay of published readouts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_annual import require_endpoint_step, rollout
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.observation_annual import read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def load_model(row):
    saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    require_endpoint_step(saved, row)
    if row["adapter"]:
        raise ValueError("This viewer cohort has no model-forcing adapter")
    model = MonthlyLatentOcean().cuda().eval()
    model.load_state_dict(saved["ema"], strict=True)
    return model


def verify_sources(config):
    contracts = {}
    for row in config["models"]:
        source = Path(row["annual_root"])
        old_config = json.loads((source / "paths.json").read_text())
        audit = json.loads((source / "AUDITED.json").read_text())
        qualified = json.loads((source / "QUALIFIED.json").read_text())
        assert audit["config_sha256"] == digest(source / "paths.json")
        assert qualified["audit_sha256"] == digest(source / "AUDITED.json")
        assert qualified["first_month_readout_exact"] is True
        old = next(m for m in old_config["models"] if m["name"] == row["annual_name"])
        assert old["checkpoint"] == row["checkpoint"]
        assert old_config["origins"] == config["origins"]
        assert old_config["samples"] == config["samples"]
        assert old_config["annual"] == config["annual"]
        for path, sha in audit["contracts"].items():
            if path == row["checkpoint"] or path.endswith(
                ("grid.npz", "statistics.npz", "SHA256SUMS", "COMPLETE.json")
            ):
                assert digest(path) == sha, path
                contracts[path] = sha
        directory = source / row["annual_name"]
        complete = json.loads((directory / "COMPLETE.json").read_text())
        assert complete["origins"] == config["origins"]
        for name, sha in complete["outputs"].items():
            path = directory / name
            assert digest(path) == sha, str(path)
            contracts[str(path)] = sha
        contracts[str(directory / "COMPLETE.json")] = digest(
            directory / "COMPLETE.json"
        )
    return contracts


def check_overlap(result, path, first=False):
    with np.load(path) as saved:
        for key in ("surface", "ohc", "months", "latent_rms"):
            value, expected = result[key], saved[key]
            if first and key in ("surface", "ohc", "months"):
                expected = expected[: len(value)]
            np.testing.assert_array_equal(
                value, expected, err_msg=f"Published annual {key} changed"
            )


def export_case(model, data, config, row, i, *, first=False):
    origin = config["origins"][i]
    desc, raw = read_origin(Path(config["annual"]) / "test" / origin)
    result = rollout(
        model,
        data,
        desc,
        raw,
        origin_index=i,
        first_month_only=first,
        export_profiles=True,
    )
    path = Path(row["annual_root"]) / row["annual_name"] / (origin + ".npz")
    check_overlap(result, path, first)
    reference_temperature = []
    for month in result["months"]:
        sample = data.load(data.root / "test" / (str(month) + ".npz"))
        reference_temperature.append(
            sample["raw"]["interior"].reshape(2, 14, *model.shape)[0]
        )
    with np.load(path) as saved:
        result.update(
            reference_surface=saved["reference_surface"],
            reference_ohc=saved["reference_ohc"],
        )
    result.update(
        reference_temperature=np.stack(reference_temperature),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["audit", "qualify", "export"], required=True
    )
    parser.add_argument("--index", type=int, default=0)
    a = parser.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    assert os.environ["SAMUDRA_CODE_COMMIT"] == config["producer"]
    if a.stage == "audit":
        contracts = verify_sources(config)
        atomic_json(
            dict(config_sha256=digest(a.root / "paths.json"), contracts=contracts),
            a.root / "AUDITED.json",
        )
        return
    proof = json.loads((a.root / "AUDITED.json").read_text())
    assert proof["config_sha256"] == digest(a.root / "paths.json")
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
    row = config["models"][a.index]
    assert digest(row["checkpoint"]) == proof["contracts"][row["checkpoint"]]
    model = load_model(row)
    if a.stage == "qualify":
        export_case(model, data, config, row, 0, first=True)
        atomic_json(
            dict(
                producer=config["producer"],
                audit_sha256=digest(a.root / "AUDITED.json"),
                first_month_readouts_exact=True,
                independent_initial_noise=True,
            ),
            a.root / "QUALIFIED.json",
        )
        return
    qualified = json.loads((a.root / "QUALIFIED.json").read_text())
    assert qualified["producer"] == config["producer"]
    assert qualified["audit_sha256"] == digest(a.root / "AUDITED.json")
    assert qualified["first_month_readouts_exact"]
    output = a.root / row["name"]
    output.mkdir()
    import wandb

    run = wandb.init(
        entity="ocean_emulators",
        project="observational-transfer",
        name="monthly-viewer-" + row["name"],
        dir=str(output),
        config={**row, "producer": config["producer"], "inference_only": True},
        mode="online",
    )
    assert run.url
    files = {}
    for i, origin in enumerate(config["origins"]):
        result = export_case(model, data, config, row, i)
        filename = origin + ".npz"
        np.savez_compressed(output / filename, **result)
        with np.load(output / filename) as saved:
            for key, value in result.items():
                np.testing.assert_array_equal(saved[key], value)
        files[filename] = digest(output / filename)
        run.log(dict(origin_index=i, exact_annual_overlap=True))
        print(
            json.dumps(
                dict(model=row["name"], origin=origin, exact_annual_overlap=True)
            ),
            flush=True,
        )
    atomic_json(
        dict(
            producer=config["producer"],
            model=row,
            checkpoint_sha256=proof["contracts"][row["checkpoint"]],
            origins=config["origins"],
            files=files,
            exact_annual_overlap=True,
            wandb_url=run.url,
        ),
        output / "COMPLETE.json",
    )
    run.finish()


if __name__ == "__main__":
    main()
