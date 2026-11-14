"""Create paired full visualization configs for both selection rules."""

from pathlib import Path

import yaml

root = Path("/orcd/pool/008/jrusak/om4-hfds-ablation-20261021")
preset = yaml.safe_load(
    (root / "code/src/samudra/configs/samudra_om4/viz.yaml")
    .read_text()
    .replace("data: !include ../data/om4.yaml", "data: null")
    .replace("observations: !include ../data/obs.yaml", "observations: null")
)
train = yaml.safe_load((root / "configs/without.yaml").read_text())
preset.update(
    data=train["data"],
    data_root=train["experiment"]["data_root"],
    base_output_dir=str(root / "viz"),
    observations=yaml.safe_load(
        (root / "code/src/samudra/configs/data/obs.yaml").read_text()
    ),
    not_steps=["movies"],
)
for label in ["selected", "final-ema"]:
    preset["name"] = label
    preset["runs"] = [
        dict(
            name=arm,
            location=str(root / "evaluations" / f"{arm}-{label}" / "predictions.zarr"),
        )
        for arm in ["with", "without"]
    ]
    (root / "configs" / f"viz-{label}.yaml").write_text(yaml.safe_dump(preset))
