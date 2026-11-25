# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Step 0: what is the spurious daily cycle in #1037's 30-day forecasts?

Reads the case receipts written by ``llc-observation-six-hour/export_viewer.py``
(``update-*/<case>/COMPLETE.json`` plus the six-hourly ``prediction-surface``
arrays, 120 steps over 30 days) and measures, per checkpoint, case and surface
variable:

- ``day_range[d]``: wet-area mean of the within-day range (max minus min of the
  four six-hour states) on forecast day d, in physical units; and the same for
  matched LLC truth where all four states of a day are exact LLC snapshots.
- The dominant sub-daily frequency by latitude band. A forced or aliased daily
  cycle sits at 1 cycle/day at every latitude. An inertial instability follows
  the local inertial frequency, 2|sin(lat)| cycles/day (inertial periods are
  above 24 h equatorward of 30 degrees, so only poleward bands can separate the
  two within the 2 cycles/day six-hour Nyquist limit).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from samudra.observations.snapshot import LLC_STATE_NAMES

VARIABLES = {"sst": "thetao_0", "ssh": "zos"}
STEPS_PER_DAY = 4


def day_ranges(series: np.ndarray, wet: np.ndarray, area: np.ndarray) -> list[float]:
    """series: [time, y, x] six-hourly; returns one wet-area mean per whole day."""
    days = series[: len(series) // STEPS_PER_DAY * STEPS_PER_DAY]
    days = days.reshape(-1, STEPS_PER_DAY, *series.shape[1:])
    ranges = days.max(1) - days.min(1)
    w = area * wet
    return [float(np.nansum(r * w) / w.sum()) for r in ranges]


def peak_frequencies(series: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Per-cell frequency (cycles/day) of the largest spectral peak in 0.6-2 cpd."""
    values = series[:, wet].astype(np.float64)
    t = np.arange(len(values), dtype=np.float64)
    design = np.stack([np.ones_like(t), t], 1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    anomaly = (values - design @ coefficients) * np.hanning(len(values))[:, None]
    power = np.abs(np.fft.rfft(anomaly, axis=0)) ** 2
    frequency = np.fft.rfftfreq(len(values), d=1 / STEPS_PER_DAY)
    band = (frequency >= 0.6) & (frequency <= 2.0)
    return frequency[band][power[band].argmax(0)]


def analyse(receipt: Path) -> list[dict]:
    payload = json.loads(receipt.read_text())
    case = payload["case"]
    base = receipt.parent.parent
    grid = np.load(receipt.parent / "grid.npz")
    lat = np.broadcast_to(grid["lat"][:, None], grid["wet_area"].shape[1:])
    prediction = np.load(base / case["prediction"]["surface"])  # [time, 2, y, x]
    truth = np.load(base / case["llc_surface"])
    exact = np.asarray(case["llc_surface_exact"], dtype=bool)
    rows = []
    for j, (short, name) in enumerate(VARIABLES.items()):
        area = grid["wet_area"][LLC_STATE_NAMES.index(name)]
        wet = area > 0
        row = dict(
            checkpoint=payload["identity"]["checkpoint"],
            case=case["id"],
            source=case["source"],
            variable=short,
            day_range=day_ranges(prediction[:, j], wet, area),
        )
        whole_days = exact[: len(exact) // 4 * 4].reshape(-1, 4).all(1)
        truth_ranges = day_ranges(truth[:, j], wet, area)
        row["llc_day_range"] = [
            value if ok else None for value, ok in zip(truth_ranges, whole_days)
        ]
        peaks = peak_frequencies(prediction[:, j], wet)
        bands = {}
        for lo in range(-80, 80, 10):
            cells = (lat[wet] >= lo) & (lat[wet] < lo + 10)
            if cells.sum() < 20:
                continue
            centre = np.deg2rad(lo + 5)
            bands[f"{lo}..{lo + 10}"] = dict(
                median_peak_cpd=float(np.median(peaks[cells])),
                share_at_daily=float(np.mean(np.abs(peaks[cells] - 1) < 0.05)),
                inertial_cpd=float(2 * abs(np.sin(centre))),
            )
        row["peak_by_latitude"] = bands
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.exports:
        for receipt in sorted(root.glob("update-*/*/COMPLETE.json")):
            rows.extend(analyse(receipt))
    if not rows:
        raise ValueError("No export case receipts found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(
            r["checkpoint"].split("/")[-1],
            r["case"],
            r["variable"],
            "range d1/d15/d30:",
            [round(r["day_range"][d], 3) for d in (0, 14, 29)],
        )


if __name__ == "__main__":
    main()
