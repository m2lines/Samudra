# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint sensitivity audit; never reselect the primary final-EMA results."""

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_score import LICENSE, bootstrap_pairs, plt, save_csv, save_figure

ROLES = ("final-ema", "final-raw", "validation-selected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("primary", type=Path, help="Parent of eval-full-A-212403 etc.")
    args = parser.parse_args()
    primary = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted((args.results / "annual-raw").glob("*.csv.gz"))
        ],
        ignore_index=True,
    )
    primary["role"] = "final-ema"
    primary["epoch"] = 70
    contract = json.loads((args.results / "annual-contract.json").read_text())
    date_keys = ["origin", "lead_days", "variable"]
    expected = set(
        itertools.product(
            contract["origin_times"], contract["leads_days"], contract["variables"]
        )
    )
    target_dates = primary[primary.arm == "A"].set_index(date_keys).target.sort_index()
    raw_dir = args.results / "sensitivity-raw"
    raw_dir.mkdir(exist_ok=True)
    frames, manifests, intervals = [primary], {}, []
    for role in ROLES[1:]:
        role_frames = []
        for arm in "ABCD":
            root = args.sensitivity / f"{arm}-{role}"
            manifest = json.loads((root / "manifest.json").read_text())
            complete = json.loads((root / "COMPLETE.json").read_text())
            members = 1 if arm in "AB" else 16
            assert complete == {"rows": 16170, "origins": 30, "members": members}
            assert manifest["arm"] == arm and manifest["checkpoint_role"] == role
            assert manifest["members"] == members
            assert 1 <= manifest["checkpoint_epoch"] <= 70
            if role == "final-raw":
                assert manifest["checkpoint_epoch"] == 70
            else:
                assert manifest["validation_selection_identity"]
            for key in ("variables", "normalization_std", "origin_times", "leads_days"):
                assert manifest[key] == contract[key]
            frame = pd.concat(
                [pd.read_csv(p) for p in sorted(root.glob("metrics-rank*.csv"))],
                ignore_index=True,
            )
            keys = set(frame[date_keys].itertuples(index=False, name=None))
            assert len(frame) == len(keys) == 16170 and keys == expected
            assert frame.arm.eq(arm).all() and frame.members.eq(members).all()
            assert np.isfinite(frame.select_dtypes("number")).all().all()
            pd.testing.assert_series_equal(
                frame.set_index(date_keys).target.sort_index(), target_dates
            )
            scale = frame.variable.map(
                dict(
                    zip(
                        manifest["variables"],
                        manifest["normalization_std"],
                        strict=True,
                    )
                )
            )
            for column in map(str, frame.columns):
                if column.endswith("_normalized"):
                    np.testing.assert_allclose(
                        frame[column.replace("_normalized", "_physical")],
                        frame[column] * scale,
                        rtol=2e-6,
                        atol=1e-8,
                    )
            for column in ("coverage_member_range", "coverage_q05_q95"):
                assert frame[column].between(0, 1).all()
            assert (
                frame.crps_fair_normalized <= frame.crps_empirical_normalized + 1e-6
            ).all()
            if members == 1:
                assert frame.spread_normalized.eq(0).all()
                np.testing.assert_array_equal(
                    frame.crps_empirical_normalized, frame.mae_normalized
                )
            else:
                assert frame.spread_normalized.gt(0).all()
            with np.load(root / "rank-histograms.npz") as ranks:
                assert ranks["counts"].shape == (7, 77, members + 1)
                np.testing.assert_allclose(ranks["counts"].sum(-1), 30, rtol=1e-6)
            for lead in (5, 40, 360):
                with (
                    np.load(root / f"fields-{lead:03d}d.npz") as fields,
                    np.load(
                        args.primary
                        / "score-eval-full-A-212403"
                        / f"fields-{lead:03d}d.npz"
                    ) as reference,
                ):
                    for key in ("variables", "target", "mask"):
                        np.testing.assert_array_equal(fields[key], reference[key])
                    for key in ("mean", "member0", "member1", "spread"):
                        assert np.isfinite(fields[key]).all()
                        assert not fields[key][~fields["mask"].astype(bool)].any()
                    if members > 1:
                        assert not np.array_equal(fields["member0"], fields["member1"])
            frame["role"], frame["epoch"] = role, manifest["checkpoint_epoch"]
            for lead_value, group in frame.groupby("lead_days"):
                ordered = group.sort_values(["origin", "variable"])
                for part, ids in enumerate(np.array_split(np.arange(len(ordered)), 3)):
                    path = (
                        raw_dir
                        / f"{arm}-{role}-{int(str(lead_value)):03d}d-{part}.csv.gz"
                    )
                    path.write_bytes(
                        gzip.compress(
                            ordered.iloc[ids].to_csv(index=False).encode(), mtime=0
                        )
                    )
                    assert path.stat().st_size < 250 * 1024
                    path.with_suffix(path.suffix + ".license").write_text(LICENSE)
            frames.append(frame)
            role_frames.append(frame)
            manifests[f"{arm}-{role}"] = manifest
        paired = bootstrap_pairs(pd.concat(role_frames, ignore_index=True))
        paired["role"] = role
        intervals.append(paired)
    data = pd.concat(frames, ignore_index=True)
    metrics = [
        k
        for k in map(str, data.columns)
        if k.endswith("_normalized") or k.startswith("coverage")
    ]
    summary = (
        data.groupby(["arm", "role", "epoch", "lead_days"])[metrics]
        .mean()
        .reset_index()
    )
    save_csv(summary, args.results / "sensitivity-summary.csv")
    data["family"] = data.variable.str.split("_").str[0]
    save_csv(
        data.groupby(["arm", "role", "epoch", "lead_days", "family"])[
            ["rmse_physical", "crps_empirical_physical", "rmse_normalized"]
        ]
        .mean()
        .reset_index(),
        args.results / "sensitivity-family.csv",
    )
    paired = pd.concat(intervals, ignore_index=True)
    save_csv(paired, args.results / "sensitivity-paired-year-intervals.csv")
    manifest_path = args.results / "sensitivity-manifests.json.gz"
    manifest_path.write_bytes(
        gzip.compress(json.dumps(manifests, indent=2).encode(), mtime=0)
    )
    manifest_path.with_suffix(manifest_path.suffix + ".license").write_text(LICENSE)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey="row")
    styles = [("#666666", "o"), ("#b44e2f", "s"), ("#009e73", "^")]
    for col, arm in enumerate("ABCD"):
        for row, metric in enumerate(("rmse_normalized", "crps_empirical_normalized")):
            ax = axes[row, col]
            for role, (color, marker) in zip(ROLES, styles, strict=True):
                subset = summary[(summary.arm == arm) & (summary.role == role)]
                ax.plot(
                    subset.lead_days,
                    subset[metric],
                    color=color,
                    marker=marker,
                    label=role,
                )
            ax.set_title(f"{arm}: {'RMSE' if row == 0 else 'empirical CRPS'}")
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("Nominal lead (days)")
            if col == 0:
                ax.set_ylabel("Mean normalized channel score")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, args.results / "sensitivity-scores.jpg")
    print(
        "Sensitivity audit passed: eight complete evaluations, common targets and units."
    )
    print(summary[summary.lead_days == 360].to_string(index=False))
    print(
        paired[(paired.lead_days == 360) & (paired.reference == "A")].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
