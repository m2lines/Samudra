# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

import dask
import xarray as xr

from samudra.config import CpuDataLoadingConfig, EvalConfig, ObsMetricsConfig
from samudra.metrics.run import score_rollouts
from samudra.utils.location import LocalLocation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
dask.config.set(scheduler="threads", num_workers=8)
root = Path(
    "/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006"  # pragma: allowlist secret
)
out = root / "observations"
out.mkdir(exist_ok=True)
cfg = EvalConfig.from_yaml(root / "control/eval.yaml")
cfg.data.loading = CpuDataLoadingConfig()
cfg.experiment.data_root = LocalLocation(
    path=Path("/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg")
)
bundle = cfg.data.build(cfg.experiment.resolved_data_root)
assert bundle.inference_source is not None
obs = ObsMetricsConfig.model_validate(
    EvalConfig._load_yaml(root / "source/src/samudra/configs/data/obs.yaml")
)
rollouts = {}
paths = {}
for variant, job in [("control", 210664), ("sd01", 210665)]:
    for checkpoint in ["epoch_0070", "ema_latest"]:
        label = f"{variant}_{checkpoint}"
        path = (
            root
            / variant
            / "output"
            / f"current-1deg-{variant}-ga2-{job}"
            / "evals"
            / checkpoint
        )
        paths[label] = path
        rollouts[label] = xr.open_zarr(path / "predictions.zarr", chunks={})
rollouts["om4"] = bundle.inference_source.to_xarray_dataset()
result = score_rollouts(
    obs,
    rollouts=rollouts,
    data_layout=bundle.data_layout,
    data_root=cfg.experiment.resolved_data_root,
    primary_label="control_epoch_0070",
    output_dir=out,
)
for label, path in paths.items():
    result.frame[result.frame.model.isin([label, "om4"])].to_csv(
        path / "observation_metrics.csv", index=False
    )
(out / "COMPLETE.json").write_text(
    json.dumps(
        {
            "rows": len(result.frame),
            "models": list(rollouts),
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
