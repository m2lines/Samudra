# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys
from pathlib import Path

import dask
import pandas as pd
import xarray as xr

from samudra.config import CpuDataLoadingConfig, EvalConfig, ObsMetricsConfig
from samudra.metrics.run import score_rollouts
from samudra.utils.data import stack_levels
from samudra.utils.location import LocalLocation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
dask.config.set(scheduler="threads", num_workers=8)
root = Path("/scratch/jr7309/runs/current-sd-seeds16-17")
variant, seed = sys.argv[1:]
assert variant in {"control", "sd01"} and seed in {"16", "17"}
run_dir = root / "output" / f"current-1deg-{variant}-seed{seed}-torch-native1x1"
out = root / "observations" / f"{variant}-seed{seed}"
out.mkdir(parents=True, exist_ok=True)
assert not (out / "COMPLETE.json").exists(), "Already scored"
cfg = EvalConfig.from_yaml(root / f"configs/{variant}-eval.yaml")
cfg.data.loading = CpuDataLoadingConfig()
cfg.experiment.data_root = LocalLocation(
    path=Path("/scratch/jr7309/data/om4-v2026-09/om4_onedeg")
)
bundle = cfg.data.build(cfg.experiment.resolved_data_root)
assert bundle.inference_source is not None
obs = ObsMetricsConfig.model_validate(
    EvalConfig._load_yaml(Path("/opt/samudra-code/src/samudra/configs/data/obs.yaml"))
)
rollouts = {}
paths = {}
for checkpoint in ["epoch_0070", "ema_latest"]:
    label = f"{variant}_seed{seed}_{checkpoint}"
    path = run_dir / "evals" / checkpoint
    paths[label] = path
    prediction = xr.open_zarr(path / "predictions.zarr", chunks={})
    wet = stack_levels(
        xr.Dataset(
            {
                name: (
                    ("y", "x"),
                    bundle.inference_source.masks.prognostic[i].numpy().astype(bool),
                )
                for i, name in enumerate(bundle.data_layout.prognostic_var_names)
            },
            coords={"y": prediction.y, "x": prediction.x},
        ),
        bundle.data_layout,
    )
    for field in wet.data_vars:
        prediction[field] = prediction[field].where(wet[field])
    rollouts[label] = prediction
# Reuse the already-computed OM4 baseline; only model masking changed.
baseline = pd.read_csv(root / "observations/seed15_observation_metrics.csv")
baseline = baseline[baseline.model == "om4"]
result = score_rollouts(
    obs,
    rollouts=rollouts,
    data_layout=bundle.data_layout,
    data_root=cfg.experiment.resolved_data_root,
    primary_label=f"{variant}_seed{seed}_epoch_0070",
    output_dir=out,
)
result.frame = pd.concat([result.frame, baseline], ignore_index=True)
result.frame.to_csv(out / "observation_metrics.csv", index=False)
for label, path in paths.items():
    result.frame[result.frame.model.isin([label, "om4"])].to_csv(
        path / "observation_metrics.csv", index=False
    )
(out / "COMPLETE.json").write_text(
    json.dumps(
        {
            "rows": len(result.frame),
            "models": [*rollouts, "om4"],
            "masking": "Original per-channel training wet masks applied lazily; predictions stores unchanged",
            "baseline_job": 125006,
            "outputs": {k: str(v) for k, v in paths.items()},
        },
        indent=2,
    )
    + "\n"
)
print(
    result.frame[result.frame.period_kind != "annual"].to_string(index=False),
    flush=True,
)
