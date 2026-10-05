#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render matched-budget global-domain results from verified metrics and arrays."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_observation_missingness import (  # type: ignore[import-not-found]
    FIELDS,
    panel_plot,
)

from samudra.experiments.observation_metrics import PROTOCOL, selection_score

NAMES = {
    "restricted": "Restricted-loss control",
    "global": "Global physical-only",
    "latent10": "Global + 10 memory channels",
}
ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(gzip.decompress(args.bundle.read_bytes()))
    assert all(bundle["status"][arm] == "verified complete" for arm in NAMES)
    files = bundle["files"]
    reference = files["global/run/reference"]
    assert reference == files["latent10/run/reference"]
    keys = reference["spectral_keys"]
    control = files["restricted/endpoint/monthly/seasonal-climatology"]
    result: dict[str, Any] = dict(
        bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        models={},
        figures={},
        allocated_gpu_hours=bundle["allocated_gpu_hours"],
        score_reference="Same held-out global seasonal-climatology errors; validation-selected spectral keys",
        spectral_keys=keys,
    )
    grids, manifests = {}, {}
    for arm, name in NAMES.items():
        manifests[arm] = json.loads((args.arrays / arm / "manifest.json").read_text())
        for filename, receipt in manifests[arm]["files"].items():
            assert (
                hashlib.sha256((args.arrays / arm / filename).read_bytes()).hexdigest()
                == receipt["sha256"]
            )
        grids[arm] = dict(np.load(args.arrays / arm / "grid.npz"))
        for label in ("lat", "lon", "mask", "names", "channels"):
            np.testing.assert_array_equal(grids[arm][label], grids["restricted"][label])
        record: dict[str, Any] = dict(
            name=name,
            selected=files[arm + "/run/best"],
            held_out={},
            annual={},
            regional=manifests[arm]["regional"],
            latent=manifests[arm]["latent"],
        )
        for choice in ("endpoint", "selected"):
            root = f"{arm}/{choice}/monthly/"
            stem = "fixed-budget" if choice == "endpoint" else "selected"
            other_control = files[root + "seasonal-climatology"]
            for k in PROTOCOL["integrated"]:
                np.testing.assert_allclose(
                    control["metrics"][k], other_control["metrics"][k], rtol=1e-6
                )
            record["held_out"][choice] = {}
            for method, path in [
                ("forecast", stem),
                ("persistence", stem + "-inferred-persistence"),
                ("anomaly_persistence", stem + "-inferred-anomaly-persistence"),
                ("climatology", "seasonal-climatology"),
            ]:
                values = files[root + path]
                values = values["metrics"] if method == "forecast" else values
                record["held_out"][choice][method] = dict(
                    composite=selection_score(values, control, keys),
                    metrics=values["metrics"],
                    spectral_dex=float(
                        np.mean([values["spectra"][k]["error_dex"] for k in keys])
                    ),
                    integrated_normalized={
                        k: values["metrics"][k] / control["metrics"][k]
                        for k in PROTOCOL["integrated"]
                    },
                )
            record["annual"][choice] = {
                origin: files[f"{arm}/{choice}/annual/{origin}"]["leads"]
                for origin in ORIGINS
            }
            record["annual"][choice + "_day365_mean"] = {
                k: float(
                    np.mean([record["annual"][choice][o]["365"][k] for o in ORIGINS])
                )
                for k in record["annual"][choice][ORIGINS[0]]["365"]
            }
        for choice in ("endpoint", "selected"):
            for origin in ORIGINS:
                for day in (5, 30, 365):
                    for variable, metric in (("sst", "sst_rmse"), ("adt", "adt_rmse")):
                        np.testing.assert_allclose(
                            record["regional"][choice + "/" + origin][str(day)][
                                "global"
                            ][variable]["rmse"],
                            record["annual"][choice][origin][str(day)][metric],
                            rtol=1e-7,
                        )
        validations = sorted(
            [v for k, v in files.items() if k.startswith(arm + "/validation/")],
            key=lambda v: v["global_step"],
        )
        record["validation_curve"] = [
            {k: v[k] for k in ("global_step", "task_counts", "score", "om4_retention")}
            for v in validations
        ]
        result["models"][arm] = record
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for arm in ("global", "latent10"):
        values = result["models"][arm]["validation_curve"]
        x = [v["global_step"] for v in values]
        axes[0].plot(x, [v["score"] for v in values], label=NAMES[arm])
        axes[1].plot(
            x, [v["om4_retention"]["ts_mse"] for v in values], label=NAMES[arm]
        )
    for ax in axes:
        ax.set_xlabel("Total updates (OM4 + observations)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Global integrated + spectral validation score")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("OM4 validation T/S normalized MSE")
    axes[1].set_yscale("log")
    fig.savefig(args.output / "global-training-curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for arm in NAMES:
        for ax, metric, units in zip(
            axes,
            ("sst_rmse", "adt_rmse", "velocity_rmse"),
            ("SST RMSE (°C)", "ADT RMSE (m)", "Geostrophic velocity RMSE (m/s)"),
            strict=True,
        ):
            leads = sorted(
                map(int, result["models"][arm]["annual"]["endpoint"][ORIGINS[0]])
            )
            y = [
                np.mean(
                    [
                        result["models"][arm]["annual"]["endpoint"][o][str(day)][metric]
                        for o in ORIGINS
                    ]
                )
                for day in leads
            ]
            ax.plot(leads, y, "o-", label=NAMES[arm], markersize=3)
            ax.set_xlabel("Autoregressive lead (days)")
            ax.set_ylabel(units)
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(args.output / "endpoint-annual-errors.png", dpi=150)
    plt.close(fig)
    for choice, origins in [("endpoint", ORIGINS), ("selected", ORIGINS[:1])]:
        for origin in origins:
            arrays = {
                arm: np.load(args.arrays / arm / f"{choice}-{origin}.npz")
                for arm in NAMES
            }
            for phase, key, index in [
                ("initial", "initial", None),
                ("day30", "state", 0),
                ("day365", "state", 1),
            ]:
                for field in FIELDS:
                    channel, title, *_ = field
                    panels = [
                        (
                            name,
                            arrays[arm][key]
                            if index is None
                            else arrays[arm][key][index],
                            grids[arm],
                        )
                        for arm, name in NAMES.items()
                    ]
                    if index is not None and channel in (0, 6):
                        truth = np.full_like(arrays["restricted"]["initial"], np.nan)
                        truth[channel] = arrays["restricted"]["reference"][
                            index, 0 if channel == 0 else 1
                        ]
                        panels.append(
                            ("Observation product", truth, grids["restricted"])
                        )
                    filename = f"{choice}-{origin}-{phase}-channel{channel}.png"
                    result["figures"][filename] = panel_plot(
                        panels,
                        f"{title}: {origin}, {phase}, {choice}",
                        field,
                        args.output / filename,
                    )
    # Absolute regional spectra: all models and the same observed reference.
    for choice in ("endpoint", "selected"):
        values = {
            arm: files[
                f"{arm}/{choice}/monthly/"
                + ("fixed-budget" if choice == "endpoint" else "selected")
            ]["metrics"]["spectra"]
            for arm in NAMES
        }
        regions = sorted({k.split("/")[1] for k in keys})
        fig, axes = plt.subplots(
            3,
            len(regions),
            figsize=(4 * len(regions), 9),
            squeeze=False,
            layout="constrained",
        )
        for i, variable in enumerate(("sst", "adt", "eke")):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                key = f"{variable}/{region}/day30"
                for arm in NAMES:
                    v = values[arm][key]
                    ax.loglog(v["k_rad_km"], v["prediction_power"], label=NAMES[arm])
                v = values["restricted"][key]
                ax.loglog(v["k_rad_km"], v["reference_power"], "k--", label="Observed")
                ax.set_title(f"{variable.upper()}: {region}")
                ax.set_xlabel("Wavenumber (rad/km)")
                ax.set_ylabel("Spatial power")
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(args.output / (choice + "-regional-spectra-day30.png"), dpi=130)
        plt.close(fig)
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                arm: record["held_out"]["endpoint"]["forecast"]["composite"]
                for arm, record in result["models"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
