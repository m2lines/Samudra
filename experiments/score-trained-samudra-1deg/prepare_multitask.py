# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the two approved task mixtures from the first wave's common recipe."""

import argparse
import copy
from pathlib import Path

import yaml
from train_multitask import MultitaskTrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("data_root")
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args()
    common = yaml.safe_load(Path(__file__).with_name("common.yaml").read_text())
    for mode in ("forecast", "multitask"):
        for phase in ("qualification", "train"):
            raw = copy.deepcopy(common)
            raw["model"].update(arm="C", mode=mode)
            raw["rollout_validation"]["days"] = [360]
            raw["experiment"]["name"] = f"{phase}-{mode}"
            raw["experiment"]["base_output_dir"] = str(args.root / "outputs")
            raw["experiment"]["data_root"] = args.data_root
            raw["experiment"]["wandb"]["group"] = args.root.name
            raw["experiment"]["wandb"]["tags"] = ["C", "structured-multitask", mode]
            raw["preemptible"] = phase == "train"
            if phase == "qualification":
                raw["experiment"]["wandb"]["mode"] = "disabled"
            cfg = MultitaskTrainConfig(**raw)
            path = (args.config_dir or args.root / "configs") / f"{phase}-{mode}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            cfg.save_yaml(path)


if __name__ == "__main__":
    main()
