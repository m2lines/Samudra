# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Join reproduced 1850 m failures with exact accepted training exposures."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
out = Path(__file__).parent
cases = json.loads((args.root / "infer/validation-cases.json").read_text())
replay = json.loads((args.root / "replay/replay.json").read_text())
inference = json.loads((args.root / "infer/inference.json").read_text())
points = np.load(args.root / "replay/training-draws.npz")
metadata = points["metadata"]
deep = metadata[:, 2] == 13
deep_meta = metadata[deep]
deep_targets = points["targets"][deep, 2]
scale, mean = replay["source"]["std"][51], replay["source"]["mean"][51]
physical = deep_targets * scale + mean
all_t_mass = points["weights"][deep, 2].astype(np.float64) * points["importance"][deep]
rows, nearby = [], []
for case in cases:
    bins = 2048 if "flat2048" in case["arm"] else 8192
    counts, mass = points[f"count_{bins}"], points[f"mass_{bins}"]
    supervised = (
        slice(38, 57) if case["arm"].startswith("temperature-") else slice(None)
    )
    row = dict(case)
    for kind, key, value_key in (
        ("true", "true_bin", "target_c"),
        ("predicted", "predicted_bin", "prediction_c"),
        ("greedy", "greedy_bin", "greedy_c"),
    ):
        idx = case[key]
        for width in (0, 1, 4):
            indices = slice(max(0, idx - width), min(bins, idx + width + 1))
            tag = f"{kind}_neighbor{width}"
            row[f"{tag}_depth13_count"] = int(counts[51, indices].sum())
            row[f"{tag}_depth13_mass"] = float(mass[51, indices].sum())
            row[f"{tag}_depth13_mass_share"] = float(
                mass[51, indices].sum() / mass[51].sum()
            )
            row[f"{tag}_all_temperature_count"] = int(counts[38:57, indices].sum())
            row[f"{tag}_shared_supervised_count"] = int(
                counts[supervised, indices].sum()
            )
            row[f"{tag}_shared_supervised_mass"] = float(
                mass[supervised, indices].sum()
            )
        physical_near = np.abs(physical - case[value_key]) <= 0.1
        row[f"{kind}_physical_0p1c_count"] = int(physical_near.sum())
        row[f"{kind}_physical_0p1c_mass"] = float(all_t_mass[physical_near].sum())
        if bins == 8192:
            coarse, fine = idx // 64, idx % 64
            coarse_slice = slice(coarse * 64, (coarse + 1) * 64)
            row[f"{kind}_coarse_depth13_count"] = int(counts[51, coarse_slice].sum())
            row[f"{kind}_coarse_depth13_mass"] = float(mass[51, coarse_slice].sum())
            row[f"{kind}_coarse_shared_mass"] = float(
                mass[supervised, coarse_slice].sum()
            )
            row[f"{kind}_coarse_shared_count"] = int(
                counts[supervised, coarse_slice].sum()
            )
            row[f"{kind}_fine_token_depth13_count"] = int(
                counts[51].reshape(128, 64)[:, fine].sum()
            )
            row[f"{kind}_fine_token_shared_count"] = int(
                counts[supervised].reshape(-1, 128, 64)[:, :, fine].sum()
            )
    # Repeated accepted occurrences are the exposure count; unique geography is separate.
    row["same_cell_training_count"] = int(
        ((deep_meta[:, 3] == case["row"]) & (deep_meta[:, 4] == case["col"])).sum()
    )
    near_cell = (np.abs(deep_meta[:, 3] - case["row"]) <= 4) & (
        np.minimum(
            np.abs(deep_meta[:, 4] - case["col"]),
            1440 - np.abs(deep_meta[:, 4] - case["col"]),
        )
        <= 4
    )
    row["within_4_grid_cells_training_count"] = int(near_cell.sum())
    # Nearest labels are descriptive examples, not input-space nearest neighbors.
    closest = np.argsort(np.abs(physical - case["target_c"]), kind="stable")[:5]
    for index in closest:
        nearby.append(
            dict(
                arm=case["arm"],
                validation_origin=case["origin"],
                validation_row=case["row"],
                validation_col=case["col"],
                training_update=int(deep_meta[index, 0]),
                training_origin=int(deep_meta[index, 1]),
                training_row=int(deep_meta[index, 3]),
                training_col=int(deep_meta[index, 4]),
                training_target_c=float(physical[index]),
                target_difference_c=float(physical[index] - case["target_c"]),
            )
        )
    rows.append(row)

summaries = []
for arm in sorted({r["arm"] for r in rows}):
    arm_rows = sorted(
        (r for r in rows if r["arm"] == arm),
        key=lambda r: r["weighted_squared_error"],
        reverse=True,
    )
    total = sum(r["weighted_squared_error"] for r in arm_rows)
    for rank, row in enumerate(arm_rows, 1):
        row["error_rank"] = rank
        row["error_share"] = row["weighted_squared_error"] / total
    summaries.append(
        dict(
            arm=arm,
            top1_share=arm_rows[0]["error_share"],
            top5_share=sum(r["error_share"] for r in arm_rows[:5]),
            count=len(arm_rows),
            full_grid_nmse=total / len(arm_rows),
            true_bin_zero_count_cases=sum(
                r["true_neighbor0_depth13_count"] == 0 for r in arm_rows
            ),
            predicted_bin_zero_count_cases=sum(
                r["predicted_neighbor0_depth13_count"] == 0 for r in arm_rows
            ),
            true_bin_zero_error_share=sum(
                r["error_share"]
                for r in arm_rows
                if r["true_neighbor0_depth13_count"] == 0
            ),
            true_physical_0p1c_zero_cases=sum(
                r["true_physical_0p1c_count"] == 0 for r in arm_rows
            ),
            true_physical_0p1c_zero_error_share=sum(
                r["error_share"] for r in arm_rows if r["true_physical_0p1c_count"] == 0
            ),
        )
    )
    # Frequency strata are descriptive associations on these same held-out cases.
    for lo, hi in ((0, 0), (1, 4), (5, 19), (20, 99), (100, 10**9)):
        group = [r for r in arm_rows if lo <= r["true_neighbor0_depth13_count"] <= hi]
        if group:
            summaries[-1][f"frequency_{lo}_{hi}"] = dict(
                cases=len(group),
                error_share=sum(r["error_share"] for r in group),
                mean_squared_normalized_error=float(
                    np.mean([r["squared_normalized_error"] for r in group])
                ),
            )
rows.sort(key=lambda r: (r["arm"], r["error_rank"]))


def write_csv(name, values):
    keys = list(dict.fromkeys(k for r in values for k in r))
    opener = (
        gzip.open(out / name, "wt", newline="")
        if name.endswith(".gz")
        else (out / name).open("w", newline="")
    )
    with opener as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)


write_csv("validation-cases.csv.gz", rows)
write_csv("top-cases.csv", [r for r in rows if r["error_rank"] <= 5])
write_csv("nearest-training-labels.csv.gz", nearby)
with gzip.open(out / "receipts.json.gz", "wt") as f:
    json.dump(
        dict(
            replay=replay,
            inference=inference,
            summaries=summaries,
            accounting=(args.root / "accounting.psv").read_text(),
        ),
        f,
    )
(out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
