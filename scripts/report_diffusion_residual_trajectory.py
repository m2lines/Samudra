#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify frozen residual milestones and compare their annual error trajectories."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_diffusion_interventions import (  # type: ignore[import-not-found]
    COMPONENTS,
    ORIGINS,
    endpoint,
    manifests,
    pool,
    read,
    sha,
    table,
)

from samudra.experiments.diffusion_scratch_evaluate import verify_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.presentation, "rt") as f:
        source = json.load(f)
    climate = source["metrics"]["Training seasonal climatology"]
    expected = manifests(source["models"]["U-global"]["inputs"])
    rows, native, validation = [], [], []
    for count in (2000, 4000, 6000):
        root = a.root / f"residual-obs{count}"
        sig = read(root / "input.json")
        done = read(root / "COMPLETE.json")
        assert done["inputs"] == sig
        assert sig["counts"]["observation"] == count
        for section in ("validation", "native", "annual"):
            assert verify_output(root, section, sig)
            assert sha(root / section / "COMPLETE.json") == done["receipts"][section]
        assert manifests(read(root / "annual/input.json")) == expected
        for day in (30, 365):
            norm = pool([endpoint(climate[o], o, day) for o in ORIGINS])
            values = pool(
                [endpoint(read(root / "annual" / f"{o}.json"), o, day) for o in ORIGINS]
            )
            rows.append(
                dict(
                    observation=count,
                    om4=sig["counts"]["om4"],
                    day=day,
                    score=sum(values[k] / norm[k] for k in COMPONENTS) / 4,
                    **values,
                )
            )
        val = read(root / "validation/COMPLETE.json")
        validation.append(dict(observation=count, score=val["score"]))
        for origin in ORIGINS:
            metrics = read(root / "native" / f"{origin[:4]}.json")["metrics"]
            for day, values in metrics.items():
                for channel, variable in ((47, "thetao_9"), (66, "so_9")):
                    for key in (
                        "normalized_mse",
                        "member_normalized_mse",
                        "normalized_spread_variance",
                    ):
                        native.append(
                            dict(
                                observation=count,
                                origin=origin,
                                day=int(day),
                                variable=variable,
                                metric=key,
                                value=float(np.asarray(values[key])[0, channel]),
                            )
                        )
    table(a.output / "milestone-components.csv", rows)
    table(a.output / "native-550m-metrics.csv", native)
    table(a.output / "validation-milestones.csv", validation)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for day, marker in ((30, "o"), (365, "s")):
        r = [v for v in rows if v["day"] == day]
        ax.plot(
            [v["observation"] for v in r],
            [v["score"] for v in r],
            marker=marker,
            label=f"Day {day}",
        )
    ax.set(
        xlabel="Observation updates (main mixed run)",
        ylabel="Blended normalized RMSE",
        title="Residual + replay + bound: frozen milestones",
    )
    ax.set_xticks([2000, 4000, 6000])
    ax.grid(alpha=0.25)
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(a.output / f"milestone-trajectory.{ext}", dpi=140)
    plt.close(fig)
    for path in a.output.iterdir():
        if path.suffix in (".csv", ".png", ".pdf"):
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    print(rows)


if __name__ == "__main__":
    main()
