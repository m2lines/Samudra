# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged rollout, audit its saved fields, and add a valid channel mean."""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from samudra.config import EvalConfig
from samudra.eval import Eval
from samudra.utils.location import LocalLocation
from samudra.utils.logging import handle_logging, handle_warnings
from samudra.utils.wandb import MetricsDict


def audit_predictions(output: Path, data_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    path = output / "predictions.zarr"
    source = xr.open_zarr(data_root / "CM4.zarr", consolidated=True)
    held_times = source.time.sel(time=slice("0251-01-06", "0351-01-01")).values
    assert len(held_times) == 7300
    expected_times = held_times[2:]
    prediction = xr.open_zarr(path, consolidated=True)
    np.testing.assert_array_equal(prediction.time.values, expected_times)
    assert prediction.sizes["lev"] == 19
    assert prediction.sizes["y"] == 180 and prediction.sizes["x"] == 360
    np.testing.assert_array_equal(prediction.lev.values, source.lev.values)
    wet = source.wetmask.transpose("lev", "y", "x").values.astype(bool)
    report: dict[str, Any] = {
        "initial_condition_times": [str(value) for value in held_times[:2]],
        "forecast_count": len(expected_times),
        "forecast_first": str(expected_times[0]),
        "forecast_last": str(expected_times[-1]),
        "variables": {},
    }
    for name in ("thetao", "so", "uo", "vo", "zos"):
        array = prediction[name]
        assert array.dtype == np.dtype("float32"), (name, array.dtype)
        mask = wet[0] if name == "zos" else wet
        expected_shape = (7298, 180, 360) if name == "zos" else (7298, 19, 180, 360)
        assert array.shape == expected_shape, (name, array.shape)
        minimum, maximum = math.inf, -math.inf
        chunk_shape = array.encoding["chunks"]
        for start in range(0, len(expected_times), chunk_shape[0]):
            block = array.isel(time=slice(start, start + chunk_shape[0])).values
            ocean = block[:, mask]
            assert np.isfinite(ocean).all(), (name, start, "non-finite ocean forecast")
            assert np.isnan(block[:, ~mask]).all(), (name, start, "non-NaN dry cell")
            minimum = min(minimum, float(ocean.min()))
            maximum = max(maximum, float(ocean.max()))
        report["variables"][name] = {
            "shape": array.shape,
            "chunks": chunk_shape,
            "minimum": minimum,
            "maximum": maximum,
            "all_wet_finite": True,
            "all_dry_nan": True,
        }
        print("READBACK_PASSED", name, report["variables"][name], flush=True)
    report["audit_seconds"] = time.perf_counter() - started
    report["passed"] = True
    return report


class AuditedEval(Eval):
    def standalone_inference(self) -> MetricsDict:
        metrics = super().standalone_inference()
        # The existing all-field mean includes derived heat content. Preserve
        # historical logs, but explicitly label a mean over the 77 prognostics.
        values: list[float] = []
        for name in self.prognostic_var_names:
            value = metrics[f"inference/time_mean_norm/rmse/{name}"]
            assert isinstance(value, (int, float)) and math.isfinite(value), name
            values.append(float(value))
        score = sum(values) / len(values)
        metrics["inference/time_mean_norm/prognostic_channel_mean"] = score
        assert isinstance(self.resolved_data_root, LocalLocation)
        report = audit_predictions(self.output_dir, self.resolved_data_root.path)
        report.update(
            checkpoint=self.model_path,
            wandb_id=self.wandb_id,
            time_mean_normalized_rmse_prognostic_channel_mean=score,
            metric_definition="RMSE of time-mean states; not mean per-timestep RMSE",
        )
        (self.output_dir / "COMPLETE.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return metrics


if __name__ == "__main__":
    config = EvalConfig.from_yaml_and_cli()
    config.prepare_output_dirs()
    config.save_yaml(config.experiment.output_dir / "config.yaml")
    handle_logging(config.debug, config.experiment.output_dir)
    handle_warnings()
    evaluator = AuditedEval(config)
    evaluator.run()
