# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate the matched configurations and record data/runtime provenance."""

import hashlib
import json
import sys
from pathlib import Path

import torch
import xarray as xr
import yaml

from samudra.config import TrainConfig

root = Path(sys.argv[1])
configs = {}
reference = yaml.safe_load((root / "configs" / "oct24-control-config.yaml").read_text())


def scientific_config(d):
    d = json.loads(json.dumps(d))
    for key in ("loss", "preemptible"):
        d.pop(key)
    for key in ("name", "base_output_dir", "data_root", "wandb"):
        d["experiment"].pop(key)
    d["post_train_eval"].pop("eval_config_path")
    return d


for arm in ("legacy_mse", "wet_mse", "dynamic_legacy"):
    raw = json.loads((root / "configs" / f"{arm}.json").read_text())
    assert scientific_config(raw) == scientific_config(reference), arm
    cfg = TrainConfig.from_yaml(root / "configs" / f"{arm}.json")
    d = cfg.model_dump(mode="json")
    (root / "provenance" / f"resolved-{arm}.json").write_text(json.dumps(d, indent=2))
    del d["loss"]
    del d["experiment"]["name"]
    del d["post_train_eval"]["eval_config_path"]
    configs[arm] = d
assert configs["legacy_mse"] == configs["wet_mse"] == configs["dynamic_legacy"]
data_root = Path(str(cfg.experiment.data_root))
metadata = {}
for name in ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr"):
    path = data_root / name
    ds = xr.open_zarr(path, consolidated=True)
    metadata[name] = {
        "sizes": dict(ds.sizes),
        "attrs": ds.attrs,
        "metadata_sha256": hashlib.sha256(
            (path / ".zmetadata").read_bytes()
        ).hexdigest(),
    }
    if name != "OM4.zarr":
        metadata[name]["values"] = {v: ds[v].values.tolist() for v in ds.data_vars}
    else:
        metadata[name]["time_bounds"] = [
            str(ds.time.values[0]),
            str(ds.time.values[-1]),
        ]
    ds.close()
(root / "provenance" / "qualification.json").write_text(
    json.dumps(
        {
            "config_diff_only": [
                "loss",
                "experiment.name",
                "post_train_eval.eval_config_path",
            ],
            "reference_run": "6j1z2mwc",
            "torch": torch.__version__,
            "python": sys.version,
            "data": metadata,
        },
        indent=2,
        default=str,
    )
)
print("Matched configuration equality and data inspection passed", flush=True)
