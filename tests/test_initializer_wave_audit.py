# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import itertools
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

AUDIT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/audit_initializer_wave.py")
)["summarize"]


def fixture(tmp_path):
    run = tmp_path / "raw" / "A"
    run.mkdir(parents=True)
    names = ["thetao_0", "thetao_1", "so_0", "uo_0", "vo_0", "zos"]
    origins = ["2015-01-01 12:00:00", "2015-02-01 12:00:00"]
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(origins))
    (run / "EVAL_COMPLETE.json").write_text(json.dumps({"ranks": 1, "origins": 2}))
    (run / "manifest.json").write_text(
        json.dumps({"channels": names, "arguments": {"phase": "reconstruction"}})
    )
    rows = []
    for mode, region, origin, lead, channel in itertools.product(
        ["inferred", "true", "inferred_persistence", "true_persistence"],
        ["global", "tropics", "extratropics"],
        origins,
        range(0, 31, 5),
        names,
    ):
        error = 0 if mode.startswith("true") and lead == 0 else 0.04
        rows.append(
            dict(
                mode=mode,
                region=region,
                origin_time=origin,
                lead_days=lead,
                channel=channel,
                normalized_mse=error,
                physical_mse=error * 4,
                prediction_second_moment=9.0,
                target_second_moment=4.0,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "heldout-rank0.csv", index=False)
    return run, expected, frame


def test_full_paired_product_and_rmse_summary(tmp_path):
    run, expected, _ = fixture(tmp_path)
    AUDIT(run.parent, expected, tmp_path / "summary")
    summary = pd.read_csv(tmp_path / "summary" / "summary.csv")
    selected = summary[
        (summary["mode"] == "inferred") & (summary["variable"] == "thetao")
    ]
    assert (selected["normalized_rmse"] == 0.2).all()
    assert (selected["physical_rmse"] == 0.4).all()


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "nonfinite", "wrong_origin"]
)
def test_corrupt_evaluation_rejected(tmp_path, damage):
    run, expected, frame = fixture(tmp_path)
    if damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "duplicate":
        frame.iloc[-1] = frame.iloc[0]
    elif damage == "nonfinite":
        frame.loc[0, "normalized_mse"] = float("nan")
    else:
        frame["origin_time"] = frame["origin_time"].replace(
            "2015-02-01 12:00:00", "2016-02-01 12:00:00"
        )
    frame.to_csv(run / "heldout-rank0.csv", index=False)
    with pytest.raises(AssertionError):
        AUDIT(run.parent, expected, tmp_path / "summary")


def test_packed_rank_parts_preserve_values(tmp_path):
    run, expected, _ = fixture(tmp_path)
    path = run / "heldout-rank0.csv"
    compressed = gzip.compress(path.read_bytes())
    middle = len(compressed) // 2
    Path(str(path) + ".gz.part000").write_bytes(compressed[:middle])
    Path(str(path) + ".gz.part001").write_bytes(compressed[middle:])
    Path(str(path) + ".gz.part000.license").write_text("license")
    path.unlink()
    AUDIT(run.parent, expected, tmp_path / "summary")
    summary = pd.read_csv(tmp_path / "summary" / "summary.csv")
    velocity = summary[summary["variable"] == "uo"]
    assert (velocity["rms_ratio"] == 1.5).all()
    assert (velocity["prediction_rms"] == 3).all()
