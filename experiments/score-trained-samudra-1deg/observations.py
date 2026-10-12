# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Score all continuous mean/member trajectories on published five-day products."""

import argparse
import json
import logging
import os
from pathlib import Path

import dask
import numpy as np
import wandb
import xarray as xr
from train_score import ScoreTrainConfig

from samudra.config import CpuDataLoadingConfig, EvalConfig, ObsMetricsConfig
from samudra.metrics import comparisons, kernels, observations, report
from samudra.metrics.run import score_rollouts
from samudra.utils.data import stack_levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dask.config.set(scheduler="threads", num_workers=8)
    cfg = ScoreTrainConfig.from_yaml(args.campaign / "configs/train-B.yaml")
    raw = cfg.model_dump(mode="json")
    raw["data"]["sources"][0]["inference_time"] = {
        "start": "2014-10-10",
        "end": "2022-12-24",
    }
    cfg = ScoreTrainConfig(**raw)
    cfg.data.loading = CpuDataLoadingConfig()
    bundle = cfg.data.build(cfg.experiment.resolved_data_root, include_inference=True)
    source = bundle.inference_source
    assert source is not None
    obs = ObsMetricsConfig.model_validate(
        EvalConfig._load_yaml(
            Path(__file__).resolve().parents[2] / "src/samudra/configs/data/obs.yaml"
        )
    )
    rollouts, manifests = {}, {}
    times = None
    for arm in "ABCD":
        root = args.export / arm
        assert json.loads((root / "COMPLETE.json").read_text()) == {
            "frames": 598,
            "score_rows": 46046,
        }
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["replay_verified_against"]
        assert not manifest["partial"] and manifest["checkpoint_epoch"] == 70
        expected_slots = (
            ["mean"] if arm in "AB" else ["mean"] + [f"member{i}" for i in range(4)]
        )
        assert manifest["rollout_slots"] == expected_slots
        manifests[arm] = manifest
        for slot in expected_slots:
            prediction = xr.open_zarr(root / slot / "predictions.zarr", chunks={})
            assert prediction.sizes["time"] == 598
            if times is None:
                times = prediction.time.values
            else:
                np.testing.assert_array_equal(prediction.time.values, times)
            wet = stack_levels(
                xr.Dataset(
                    {
                        name: (
                            ("y", "x"),
                            source.masks.prognostic[i].numpy().astype(bool),
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
            rollouts[f"{arm}-{slot}"] = prediction
    assert len(rollouts) == 12 and times is not None
    run = wandb.init(
        entity=cfg.experiment.wandb.entity,
        project=cfg.experiment.wandb.project,
        name="score-continuous-observations",
        mode="online",
        dir=str(args.output),
        config={
            "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
            "rollouts": list(rollouts),
            "observations": obs.model_dump(mode="json"),
        },
    )
    print(f"Online observation scoring: {run.url}", flush=True)
    # Metadata-only cadence checks before spatial interpolation or expensive scoring.
    products = observations.open_products(obs, cfg.experiment.resolved_data_root)
    model_time = kernels.coerce_datetime_values(times)
    clock = xr.DataArray(
        np.zeros(len(model_time)), dims="time", coords={"time": model_time}
    )
    product_records = {}
    for name, product in products.items():
        full, _ = comparisons.align(
            clock, product.time, None, f"{name} full-span cadence preflight"
        )
        scored, _ = comparisons.align(
            clock, product.time, obs.window, f"{name} scoring-window cadence preflight"
        )
        product_records[name] = {
            "full_common_samples": full.sizes["time"],
            "scored_samples": scored.sizes["time"],
            "scored_first": str(scored.time.values[0]),
            "scored_last": str(scored.time.values[-1]),
            "attributes": dict(product.attrs),
        }
        product.close()
    provenance = {
        "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
        "container": os.environ["SAMUDRA_CONTAINER_IMAGE_REF"],
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "wandb_url": run.url,
        "rollout_manifests": manifests,
        "observations": obs.model_dump(mode="json"),
        "products": product_records,
        "masking": "Original per-channel training wet masks; stores unchanged",
        "ensemble_semantics": "Each member and ensemble mean scored separately; nonlinear metrics are not interchangeable",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    print(
        "All product and trajectory time axes passed exact-match preflight.", flush=True
    )
    result = score_rollouts(
        obs,
        rollouts=rollouts,
        data_layout=bundle.data_layout,
        data_root=cfg.experiment.resolved_data_root,
        primary_label="A-mean",
        output_dir=args.output,
    )
    for label in rollouts:
        run.log(
            {
                f"{label}/{key}": value
                for key, value in report.to_wandb(result.frame, label).items()
            }
        )
    run.finish()
    (args.output / "COMPLETE.json").write_text(
        json.dumps({"rows": len(result.frame), "models": list(rollouts)}) + "\n"
    )
    print(
        result.frame[result.frame.period_kind != "annual"].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
