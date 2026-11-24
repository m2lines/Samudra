# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Write resolved experiment configs into a remote campaign directory."""

import argparse
import copy
import json
from pathlib import Path

import yaml
from train_score import ScoreTrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--auxiliary-calibration", type=Path)
    parser.add_argument("--structured-scales", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(Path(__file__).with_name("common.yaml").read_text())
    for mode in ("qualification", "train"):
        for arm in ("B", "C", "D"):
            raw = copy.deepcopy(config)
            raw["model"]["arm"] = arm
            raw["experiment"]["name"] = f"{mode}-{arm}"
            raw["experiment"]["base_output_dir"] = str(args.root / "outputs")
            raw["experiment"]["data_root"] = (
                "/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg"  # pragma: allowlist secret - dataset path
            )
            raw["preemptible"] = mode == "train"
            if mode == "qualification":
                raw["experiment"]["wandb"]["mode"] = "disabled"
            elif arm == "D":
                if args.auxiliary_calibration is None:
                    # A full D run requires the qualification's frozen weights.
                    continue
                raw["scales_path"] = str(
                    args.structured_scales
                    or args.root / "outputs/qualification-D/structured-scales.json"
                )
                calibration = json.loads(args.auxiliary_calibration.read_text())
                raw["model"]["spatial_weight"] = calibration["spatial_weight"]
                raw["model"]["difference_weight"] = calibration["difference_weight"]
            cfg = ScoreTrainConfig(**raw)
            config_dir = args.config_dir or args.root / "configs"
            output = config_dir / f"{mode}-{arm}.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            cfg.save_yaml(output)


if __name__ == "__main__":
    main()
