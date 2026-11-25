# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Write the training config and fork arguments for one replay window and arm.

Output: ``<run-root>/<window>/<arm>/train.yaml`` and ``arm.json``. Training then
writes into ``<run-root>/<window>/<arm>/`` (``base_output_dir`` is the window).
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from samudra.observations.config import ObservationTrainConfig

HERE = Path(__file__).resolve().parent
RECIPE_KEYS = {"gradient_clip", "adam_betas", "learning_rate", "loss_guard", "probe"}


def build(window: str, arm: str, run_root: Path, spec: dict) -> tuple[dict, dict]:
    win, recipe = spec["windows"][window], spec["arms"][arm]
    unknown = set(recipe) - RECIPE_KEYS - {"reseed"}
    if unknown:
        raise ValueError(f"Unknown arm settings: {sorted(unknown)}")
    cfg = yaml.safe_load((HERE / win["base"]).read_text())
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["base_output_dir"] = str(run_root / window)
    cfg["experiment"]["name"] = arm
    cfg["experiment"]["wandb"].update(
        group=f"spike-replay-{window}",
        tags=["llc", "six-hour", "spike-replay", window, arm],
    )
    cfg["stop_after"] = win["stop_after"]
    cfg["milestone_updates"] = []
    cfg.update({k: v for k, v in recipe.items() if k in RECIPE_KEYS})
    ObservationTrainConfig.model_validate(cfg)  # Fail before any allocation.
    jesse = Path(spec["jesse"])
    fork = dict(
        parent=str(jesse / win["parent"]),
        original_metrics=str(jesse / win["original_metrics"]),
        reseed=recipe.get("reseed"),
        window=window,
        arm=arm,
    )
    return cfg, fork


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window")
    parser.add_argument("arm")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load((HERE / "arms.yaml").read_text())
    cfg, fork = build(args.window, args.arm, args.run_root, spec)
    target = args.run_root / args.window / args.arm
    target.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("train.yaml", yaml.safe_dump(cfg, sort_keys=False)),
        ("arm.json", json.dumps(fork, indent=2) + "\n"),
    ):
        path = target / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"{path} exists with different content")
        path.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
