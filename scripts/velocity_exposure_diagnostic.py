# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare validation learning curves under common DUACS-example caps."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def history(path):
    config = json.loads((path / "config.json").read_text())
    payload = (path / "history.jsonl").read_bytes()
    rows, previous = [], None
    frequency, world = config["validate_every"], config["world_size"]
    for line in payload.splitlines():
        record = json.loads(line)
        if "train/loss" in record:
            previous = record
        if "validation/vector_rmse_10d" not in record:
            continue
        if previous is None:
            raise ValueError(f"Validation without preceding training record: {path}")
        # Validation lacks an update field. Training logs every ten updates, so
        # the next multiple of validate_every identifies its update uniquely.
        update = ((previous["update"] + frequency - 1) // frequency) * frequency
        gap = update - previous["update"]
        if not 0 <= gap < 10:
            raise ValueError(f"Cannot locate validation update: {path}, {previous}")
        lower = previous["exposure/duacs"]
        upper = lower + gap * world
        if config["variant"] == "D0":
            lower = upper  # All intervening examples are DUACS for D0.
        rows.append(
            {
                "variant": config["variant"],
                "seed": config["seed"],
                "update": update,
                "duacs_examples_lower": lower,
                "duacs_examples_upper": upper,
                "preceding_gpu_hours": previous["gpu_hours"],
                **{
                    f"rmse_{d}d": record[f"validation/vector_rmse_{d}d"]
                    for d in (5, 10, 20, 30)
                },
            }
        )
    if not rows:
        raise ValueError(f"No validation history: {path}")
    return rows, hashlib.sha256(payload).hexdigest()


def write(root, output):
    if output.exists():
        raise FileExistsError(output)
    rows, hashes = [], {}
    for variant in ("D0", "D4"):
        for seed in (15, 16, 17):
            name = f"confirm-{variant}-s{seed}"
            records, digest = history(root / name)
            rows.extend(records)
            hashes[name] = digest
    frame = pd.DataFrame(rows)
    groups = list(frame.groupby(["variant", "seed"]))
    common_limit = min(int(group.duacs_examples_lower.max()) for _, group in groups)
    caps = [cap for cap in (4096, 16384, 65536, 262144, 524288) if cap <= common_limit]
    caps.append(common_limit)
    selected = []
    for cap in sorted(set(caps)):
        for (variant, seed), group in groups:
            eligible = group[group.duacs_examples_upper <= cap]
            if eligible.empty:
                raise ValueError(f"No checkpoint below cap {cap}: {variant}/{seed}")
            best = eligible.loc[eligible.rmse_10d.idxmin()].to_dict()
            selected.append(
                {"duacs_example_cap": cap, "validation_checks": len(eligible), **best}
            )
    table = pd.DataFrame(selected)
    text = [
        "# DUACS exposure diagnostic",
        "",
        "These are best-so-far scores on the 12 checkpoint-selection validation dates, "
        "not the full held-out evaluation. RMSE is in m/s. Each pair is restricted to the same maximum "
        "number of DUACS window draws. D4 also sees OM4 examples. The limits do not "
        "match optimizer steps, total compute, learning-rate phase, geometry channels, "
        "or the number of validation opportunities, so this is a learning-curve "
        "diagnostic rather than a causal transfer estimate.",
        "",
        "| DUACS draw cap | Seed | D0 10-day RMSE | D4 10-day RMSE | D4 improvement |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for (cap, seed), group in table.groupby(["duacs_example_cap", "seed"]):
        pair = group.set_index("variant").rmse_10d.to_dict()
        d0, d4 = float(pair["D0"]), float(pair["D4"])
        text.append(
            f"| {cap} | {seed} | {d0:.6f} | {d4:.6f} | {100 * (1 - d4 / d0):+.2f}% |"
        )
    text += [
        "",
        "Training exposure is logged every ten updates; validation every 256. The "
        "CSV brackets D4 exposure between the preceding count and that count plus "
        "all intervening examples (at most 36 draws). A checkpoint is admitted only "
        "when its upper bound is within the cap. D0 exposure is exact because every "
        "update uses DUACS. The common final cap is the minimum, across all six runs, "
        "of the largest logged validation exposure lower bound. Later training "
        "without validation cannot extend these curves. Repeated log records after "
        "recovery are retained as observed validation opportunities.",
        "",
        "The two longer-lead columns in matched-exposure.csv refer to the same "
        "10-day-selected checkpoint, rather than independent lead-specific minima. "
        "Input history checksums are recorded in provenance.json so this snapshot "
        "can be distinguished from the final completed histories.",
        "",
    ]
    output.mkdir(parents=True)
    for (variant, seed), group in frame.groupby(["variant", "seed"]):
        group.to_csv(
            output / f"validation-exposure-{variant}-s{seed}.csv.xz", index=False
        )
    table.to_csv(output / "matched-exposure.csv", index=False)
    (output / "report.md").write_text("\n".join(text))
    (output / "provenance.json").write_text(
        json.dumps({"history_sha256": hashes}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.campaign_root, args.output)
