# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fail if anything besides the declared loss and output metadata differs."""

import json
from pathlib import Path

from samudra.config import EvalConfig, TrainConfig

root = Path(__file__).resolve().parents[2]
a = TrainConfig.from_yaml(
    root / "experiments/current-stochastic-depth-1deg/control-train.yaml"
).model_dump(mode="json")
b = TrainConfig.from_yaml(root / "experiments/current-band-loss/train.yaml").model_dump(
    mode="json"
)


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(
            [
                differences(a.get(k), b.get(k), prefix + "." + k if prefix else k)
                for k in sorted(a.keys() | b.keys())
            ],
            [],
        )
    return [] if a == b else [dict(path=prefix, baseline=a, band=b)]


changes = differences(a, b)
expected = {
    "loss.band_weight",
    "loss.band_scales_file",
    "loss.limit",
    "loss.metric",
    "loss.type",
    "experiment.name",
    "post_train_eval.eval_config_path",
}
assert {x["path"] for x in changes} == expected, changes
c = TrainConfig.from_yaml(
    root / "experiments/current-band-loss/mse-train.yaml"
).model_dump(mode="json")
assert c["loss"] == "mse"
assert {x["path"] for x in differences(c, b)} == {"loss", "experiment.name"}
assert b["model"]["unet"]["stochastic_depth_rate"] == 0
assert b["experiment"]["rand_seed"] == c["experiment"]["rand_seed"] == 15
assert not b["delayed_loss_estimate"]
x = EvalConfig.from_yaml(
    root / "experiments/current-stochastic-depth-1deg/control-eval.yaml"
).model_dump(mode="json")
y = EvalConfig.from_yaml(root / "experiments/current-band-loss/eval.yaml").model_dump(
    mode="json"
)
assert {v["path"] for v in differences(x, y)} == {"experiment.name"}
print(
    json.dumps(
        dict(
            status="matched",
            baseline_source="a20c444849",
            changes=changes,
            effective_batch=4 * 4 * 2,
            seed=15,
            stochastic_depth=0,
            epochs=70,
        ),
        indent=2,
    )
)
