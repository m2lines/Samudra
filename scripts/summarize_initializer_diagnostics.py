#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit initialization error decompositions against the primary lead-zero metric."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes


def frame(path):
    return pd.read_csv(io.BytesIO(read_bytes(path)))


def variable(channel):
    return "sst" if channel == "thetao_0" else channel.split("_")[0]


def summarize(raw, analysis):
    analysis = Path(analysis)
    primary = frame(analysis / "depth-summary.csv")
    temporal_rows, spatial_rows = [], []
    audits = {}
    components = ["mean_bias_mse", "temporal_amplitude_mse", "temporal_pattern_mse"]
    for run in sorted(primary["run"].unique()):
        root = Path(raw) / run
        temporal = frame(root / "temporal-decomposition.csv")
        spatial = frame(root / "spatial-scales.csv")
        expected = primary[
            (primary["run"] == run)
            & (primary["mode"] == "inferred")
            & (primary["lead_days"] == 0)
        ]
        keys = ["region", "channel"]
        if (
            temporal.duplicated(keys).any()
            or spatial.duplicated(keys + ["box_width_cells"]).any()
        ):
            raise ValueError("Duplicate diagnostic rows")
        paired = temporal.merge(expected, on=keys, validate="one_to_one")
        if len(paired) != len(temporal) or len(paired) != len(expected):
            raise ValueError(
                "Diagnostic channels or regions differ from primary metrics"
            )
        np.testing.assert_allclose(
            paired["mse"], paired["normalized_mse"], rtol=2e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            temporal[components].sum(1), temporal["mse"], rtol=2e-5, atol=1e-8
        )
        moments = [
            "predicted_anomaly_second_moment",
            "target_anomaly_second_moment",
            "anomaly_cross_moment",
            "anomaly_mse",
        ]
        if (
            not np.isfinite(temporal[["mse", *components]]).all().all()
            or not np.isfinite(spatial[moments]).all().all()
        ):
            raise ValueError("Nonfinite diagnostic moments")
        np.testing.assert_allclose(
            spatial[moments[0]] + spatial[moments[1]] - 2 * spatial[moments[2]],
            spatial["anomaly_mse"],
            rtol=2e-5,
            atol=1e-8,
        )
        if set(spatial["box_width_cells"]) != {0, 3, 9}:
            raise ValueError("Spatial scales differ")
        expected_keys = set(temporal[keys].itertuples(index=False, name=None))
        for _, scale in spatial.groupby("box_width_cells"):
            if set(scale[keys].itertuples(index=False, name=None)) != expected_keys:
                raise ValueError("Incomplete spatial scale coverage")
        origins = json.loads(read_bytes(root / "EVAL_COMPLETE.json"))["origins"]
        if (
            not (temporal["origins"] == origins).all()
            or not (spatial["origins"] == origins).all()
        ):
            raise ValueError("Diagnostic origin counts differ")
        zero_scale = spatial[spatial["box_width_cells"] == 0].merge(
            temporal, on=keys, validate="one_to_one"
        )
        if len(zero_scale) != len(temporal):
            raise ValueError(
                "Spatial and temporal diagnostics do not cover the same channels"
            )
        np.testing.assert_allclose(
            zero_scale["anomaly_mse"], zero_scale["mse"], rtol=2e-5, atol=1e-8
        )
        temporal["variable"] = temporal["channel"].map(variable)
        spatial["variable"] = spatial["channel"].map(variable)
        ts = (
            temporal.groupby(["region", "variable"])[["mse", *components]]
            .mean()
            .reset_index()
        )
        for component in components:
            ts[component + "_fraction"] = ts[component] / ts["mse"].replace(0, np.nan)
        ss = (
            spatial.groupby(["region", "variable", "box_width_cells"])[moments]
            .mean()
            .reset_index()
        )
        ss["anomaly_rms_ratio"] = np.sqrt(
            ss[moments[0]] / ss[moments[1]].replace(0, np.nan)
        )
        ss["anomaly_correlation"] = ss[moments[2]] / np.sqrt(
            ss[moments[0]] * ss[moments[1]]
        ).replace(0, np.nan)
        ts["run"] = ss["run"] = run
        temporal_rows.append(ts)
        spatial_rows.append(ss)
        audits[run] = {
            "temporal_rows": len(temporal),
            "spatial_rows": len(spatial),
            "max_primary_mse_difference": float(
                (paired["mse"] - paired["normalized_mse"]).abs().max()
            ),
        }
    pd.concat(temporal_rows, ignore_index=True).to_csv(
        analysis / "temporal-summary.csv", index=False
    )
    pd.concat(spatial_rows, ignore_index=True).to_csv(
        analysis / "spatial-summary.csv", index=False
    )
    (analysis / "diagnostic-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    summarize(args.raw, args.analysis)


if __name__ == "__main__":
    main()
