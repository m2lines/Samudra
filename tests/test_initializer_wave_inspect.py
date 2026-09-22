# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

INSPECT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/inspect_initializer_wave.py")
)["inspect"]


def test_sparse_validation_survives_long_training_tail(tmp_path, monkeypatch):
    run = tmp_path / "A"
    run.mkdir()
    (tmp_path / "A-submission.json").write_text(
        json.dumps(
            {
                "id": "123",
                "arguments": {
                    "root": str(tmp_path),
                    "name": "A",
                    "arm": "A",
                    "stage": "production",
                    "phase": "reconstruction",
                    "gpu": "rtx6000",
                },
            }
        )
    )
    events = [{"event": "validation", "step": 100, "ts_mse": 0.004}]
    events += [{"step": i, "padding": "x" * 2000} for i in range(101, 201)]
    (run / "progress.jsonl").write_text(
        "\n".join(map(json.dumps, events)) + '\n{"partial":'
    )

    def command(args, **kwargs):
        return SimpleNamespace(
            stdout=(
                "123|RUNNING|120|gres/gpu=2|node|0:0|2026-09-22T00:00:00|Unknown|\n"
                if args[0] == "sacct"
                else "123|RUNNING|node\n"
            )
        )

    monkeypatch.setattr("subprocess.run", command)
    result = INSPECT([tmp_path])
    job = result["jobs"][0]
    assert job["latest"]["step"] == 200
    assert job["validation"]["ts_mse"] == 0.004
    assert job["malformed_progress_records"] == 1
    assert result["accounting_rows"][0]["job_id"] == "123"
