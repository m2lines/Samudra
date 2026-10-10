# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

root = Path("/scratch/jr7309/runs/current-sd-seeds16-17")
records = []
for seed in (16, 17):
    for variant in ("control", "sd01"):
        p = root / "output" / f"current-1deg-{variant}-seed{seed}-torch-native1x1"
        summaries = json.loads((p / "evals/summary.json").read_text())
        assert {s["label"] for s in summaries} == {"epoch_0070", "ema_latest"}
        for s in summaries:
            z = p / "evals" / s["label"] / "predictions.zarr"
            meta = json.loads((z / ".zmetadata").read_text())["metadata"]
            expected = actual = 0
            fields = {}
            for var in ("thetao", "so", "uo", "vo", "zos"):
                m = meta[f"{var}/.zarray"]
                dims = meta[f"{var}/.zattrs"]["_ARRAY_DIMENSIONS"]
                assert m["shape"][dims.index("time")] == 598
                n = math.prod(math.ceil(a / b) for a, b in zip(m["shape"], m["chunks"]))
                files = [
                    f
                    for f in (z / var).rglob("*")
                    if f.is_file() and not f.name.startswith(".")
                ]
                assert len(files) == n, (var, len(files), n)
                expected += n
                actual += len(files)
                fields[var] = m["shape"]
            assert all(math.isfinite(v) for v in s["metrics"].values())
            records.append(
                dict(
                    seed=seed,
                    variant=variant,
                    checkpoint=s["label"],
                    n_times=598,
                    fields=fields,
                    expected_chunks=expected,
                    actual_chunks=actual,
                    n_finite_metrics=len(s["metrics"]),
                )
            )
(root / "output-verification.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records))
