# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched observation scores of saved continuous rollouts, with original masks."""

import argparse
import json
import logging
from pathlib import Path

import dask
import xarray as xr

from samudra.config import CpuDataLoadingConfig, EvalConfig, ObsMetricsConfig
from samudra.metrics.run import score_rollouts
from samudra.utils.data import stack_levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / "observations"
    out.mkdir(exist_ok=False)
    dask.config.set(scheduler="threads", num_workers=8)
    cfg = EvalConfig.from_yaml(
        root.parent / "eval-source/experiments/current-ke-aux/aligned-eval.yaml"
    )
    cfg.data.loading = CpuDataLoadingConfig()
    bundle = cfg.data.build(cfg.experiment.resolved_data_root)
    assert bundle.inference_source is not None
    # Use the published five-day products, matching the original control scores.
    # Full-range prepared local stores are daily and deliberately fail exact-time checks.
    obs = ObsMetricsConfig.model_validate(
        EvalConfig._load_yaml(
            root.parent / "eval-source/src/samudra/configs/data/obs.yaml"
        )
    )
    rollouts = {}
    for arm in ("control", "aligned", "seasonal"):
        for kind in ("ema", "raw"):
            label = f"{arm}-{kind}"
            assert (root / label / "COMPLETE.json").exists()
            prediction = xr.open_zarr(
                root / label / "continuous/predictions.zarr", chunks={}
            )
            wet = stack_levels(
                xr.Dataset(
                    {
                        name: (
                            ("y", "x"),
                            bundle.inference_source.masks.prognostic[i]
                            .numpy()
                            .astype(bool),
                        )
                        for i, name in enumerate(
                            bundle.data_layout.prognostic_var_names
                        )
                    },
                    coords={"y": prediction.y, "x": prediction.x},
                ),
                bundle.data_layout,
            )
            for field in wet.data_vars:
                prediction[field] = prediction[field].where(wet[field])
            rollouts[label] = prediction
    result = score_rollouts(
        obs,
        rollouts=rollouts,
        data_layout=bundle.data_layout,
        data_root=cfg.experiment.resolved_data_root,
        primary_label="control-ema",
        output_dir=out,
    )
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                rows=len(result.frame),
                models=list(rollouts),
                masking="Original per-channel training wet masks applied lazily; stores unchanged",
                observations=obs.model_dump(mode="json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        result.frame[result.frame.period_kind != "annual"].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
