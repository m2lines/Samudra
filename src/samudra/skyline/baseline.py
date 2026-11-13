# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Cache first-step temperature outputs from the frozen quarter-degree model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from samudra.config import SamudraConfig
from samudra.utils.ctx import BatchGrid
from samudra.utils.device import set_device

from .data import OceanSource


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    baseline_config = yaml.safe_load(args.baseline_config.read_text())
    assert baseline_config["data"]["input_steps"] == 2
    assert baseline_config["data"]["output_steps"] == 1
    assert (
        baseline_config["data"]["sources"][0]["prognostic_vars_key"]
        == "thermo_dynamic_all"
    )
    assert baseline_config["data"]["sources"][0]["boundary_vars_key"] == "tau_hfds"
    assert not baseline_config["model"]["pred_residuals"]
    device = torch.device("cuda")
    set_device(device)
    torch.set_num_threads(config["torch_threads"])
    source = OceanSource(
        config["source"], config["readers"], config.get("reference_root")
    )
    try:
        model = SamudraConfig(**baseline_config["model"]).build(
            prog_channels=154,
            boundary_channels=6,
            out_channels=77,
            input_steps=2,
            grid_sizes=[(source.height, source.width)],
        )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(
            {k.removeprefix("module."): v for k, v in state["model"].items()},
            strict=True,
        )
        checkpoint_epoch = state.get("epoch")
        del state
        model = model.to(device).eval()
        coordinates = (torch.from_numpy(source.lat), torch.from_numpy(source.lon))
        context = BatchGrid(
            torch.from_numpy(source.channel_wet[:77]).to(device),
            coordinates,
            coordinates,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_state_key": "model",
            "weight_selection": "matches published Samudra evaluator; no EMA substitution",
            "baseline_config": baseline_config,
            "source": source.receipt(),
            "torch": torch.__version__,
            "mode": "one prediction from true two-state history",
        }
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing baseline cache has different provenance")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if "dates" in config:
            from .maps import evaluation_dates

            origins = np.array([r["origin"] for r in evaluation_dates(source, config)])
        else:
            origins = source.origins["validation"]
        if "dates" not in config and config["validation_dates"]:
            origins = origins[
                np.linspace(
                    0,
                    len(origins) - 1,
                    min(config["validation_dates"], len(origins)),
                    dtype=int,
                )
            ]
        for origin in origins:
            path = args.output / f"temperature-{origin:04d}.npy"
            if path.exists():
                continue
            begin = time.monotonic()
            frame = source.read(int(origin))
            # Preserve history-major ordering separately for prognostic and forcing.
            prog = torch.from_numpy(
                frame["input"][:, :77].reshape(1, 154, source.height, source.width)
            ).to(device)
            forcing = torch.from_numpy(
                frame["input"][:, 77:].reshape(1, 6, source.height, source.width)
            ).to(device)
            prediction = (
                model.forward_once(prog, forcing, context)[0, 38:57]
                .float()
                .cpu()
                .numpy()
            )
            if not np.isfinite(prediction[source.wet]).all():
                raise FloatingPointError("Nonfinite wet baseline prediction")
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.save(f, prediction)
            temporary.replace(path)
            print(
                json.dumps(
                    {
                        "baseline_origin": int(origin),
                        "seconds": time.monotonic() - begin,
                    }
                ),
                flush=True,
            )
        (args.output / "COMPLETE.json").write_text(
            json.dumps({"origins": origins.tolist()}) + "\n"
        )
    finally:
        source.close()


if __name__ == "__main__":
    main()
