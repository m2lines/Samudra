"""Prepare a frozen, matched Engaging experiment from the checked-out source."""

import copy
import json
from pathlib import Path

import yaml

root = Path("/orcd/pool/008/jrusak/om4-hfds-ablation-20261021")
code = root / "code"
base = yaml.safe_load(
    (code / "src/samudra/configs/samudra_om4/train.yaml")
    .read_text()
    .replace("data: !include ../data/om4.yaml", "data: null")
    .replace("model: !include model.yaml", "model: null")
)
base["data"] = yaml.safe_load((code / "src/samudra/configs/data/om4.yaml").read_text())
base["model"] = yaml.safe_load(
    (code / "src/samudra/configs/samudra_om4/model.yaml").read_text()
)
base.update(
    preemptible=True,
    save_freq=70,
    gradient_accumulation_steps=8,
    backend="cuda",
    post_train_eval=None,
)
base["experiment"].update(
    base_output_dir=str(root / "outputs"),
    data_root="/orcd/data/abodner/002/jrusak/om4_onedeg_v3",
)
base["experiment"]["wandb"] = dict(
    mode="online",
    entity="ocean_emulators",
    project="default",
    group="hfds-anomaly-1deg-20261021",
)
base["data"]["loading"] = dict(
    type="cpu",
    num_workers=0,
    persistent_workers=False,
    pin_mem=True,
    concurrent_compute=True,
)
base["data"].update(input_steps=2, output_steps=2)
src = base["data"]["sources"][0]
for k, store in [
    ("data_means_location", "OM4_means.zarr"),
    ("data_stds_location", "OM4_stds.zarr"),
]:
    src[k] = str(root.parent / "om4-current-structure-20261020/normalization" / store)
for arm, forcing in [
    ("with", "tau_hfds_hfds_anom"),
    ("without", "tau_hfds"),
    ("without-detach", "tau_hfds"),
]:
    cfg = copy.deepcopy(base)
    cfg["experiment"]["name"] = "hfds-" + arm
    if arm == "without-detach":
        cfg["model"]["gradient_detach_interval"] = 1
    cfg["data"]["sources"][0]["boundary_vars_key"] = forcing
    (root / "configs" / f"{arm}.yaml").write_text(yaml.safe_dump(cfg))
    for epoch in [1, 2]:
        pilot = copy.deepcopy(cfg)
        pilot.update(
            debug=True,
            epochs=epoch,
            gradient_accumulation_steps=1,
            rollout_validation=None,
            validation_score="one_step_rmse",
        )
        pilot["experiment"]["name"] = "pilot-v3-" + arm
        pilot["experiment"]["wandb"]["mode"] = "disabled"
        (root / "configs" / f"pilot-{arm}-{epoch}.yaml").write_text(
            yaml.safe_dump(pilot)
        )
# Atomic checkpoint payload includes all RNGs; native restore handles optimizer,
# EMA, scheduler, DynamicLoss, epoch, sampler epoch, and batch count.
p = code / "src/samudra/train.py"
s = p.read_text()
s = s.replace("import datetime", "import datetime")
s = "import random\nimport numpy as np\n" + s
s = s.replace(
    '"model": self.model.state_dict(),',
    '"rng_torch": torch.get_rng_state(),\n                    "rng_cuda": torch.cuda.get_rng_state_all(),\n                    "rng_python": random.getstate(),\n                    "rng_numpy": (lambda s: (s[0], s[1].tolist(), s[2], s[3], s[4]))(np.random.get_state()),\n                    "model": self.model.state_dict(),',
)
p.write_text(s)
(root / "protocol.json").write_text(
    json.dumps(
        dict(
            source="f1e713220",
            arms=["with", "without", "without-detach"],
            epochs=70,
            seed=15,
            effective_batch=32,
            selection="best_validation_ckpt.pt by 360-day rollout RMSE; final EMA sensitivity",
            anomaly="standard full-source day-of-year climatology",
            data="om4_onedeg_v3",
            evaluation="continuous 2014-2022; quarterly 2015-2021 to 360 days; full observation metrics and spectra",
        ),
        indent=2,
    )
)
