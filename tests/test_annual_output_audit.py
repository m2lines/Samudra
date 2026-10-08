# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import runpy
from pathlib import Path
from typing import Any

import pytest


def test_annual_audit_hashes_all_declared_payloads_and_rejects_missing(tmp_path):
    runner = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/run_early_fine_annual.py")
    )
    complete = {
        "origins": {
            "2015-01-01": {
                "metrics": "2015-01-01.json",
                "arrays": "2015-01-01.npz",
                "persistence": "2015-01-01-persistence.json",
            }
        }
    }
    expected = {}
    for name in [
        "COMPLETE.json",
        "input.json",
        *complete["origins"]["2015-01-01"].values(),
    ]:
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    assert runner["annual_output_hashes"](tmp_path, complete) == expected
    (tmp_path / "2015-01-01.npz").unlink()
    with pytest.raises(ValueError, match="Missing or invalid annual output"):
        runner["annual_output_hashes"](tmp_path, complete)


@pytest.mark.parametrize("supplemental", [False, True])
def test_collector_checks_nonempty_output_audits(tmp_path, supplemental):
    import json

    import numpy as np

    from samudra.experiments.observation_pilot import digest

    collector = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/collect_early_fine_annual.py")
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez(
        samples / "grid.npz",
        lat=np.arange(2),
        lon=np.arange(3),
        mask=np.ones((1, 2, 3), bool),
    )
    np.savez(samples / "statistics.npz", surface_climatology=np.zeros((12, 2, 2, 3)))
    config: dict[str, Any] = dict(
        observations=str(samples),
        origins=["2015-01-01"],
        models=[
            dict(name=n, checkpoint_sha256="selected")
            for n in ["parent", "parent-cooldown"]
        ],
    )
    (tmp_path / "paths.json").write_text(json.dumps(config))
    for row in config["models"]:
        run = tmp_path / row["name"]
        run.mkdir()
        (run / "score.json").write_text("{}")
        np.savez(
            run / "arrays.npz",
            surface=np.zeros((73, 2, 2, 3)),
            persistence_surface=np.zeros((73, 2, 2, 3)),
            reference=np.zeros((73, 4, 2, 3)),
        )
        complete = dict(
            inputs=dict(checkpoint_sha256="selected"),
            origins={
                "2015-01-01": dict(
                    metrics="score.json", persistence="score.json", arrays="arrays.npz"
                )
            },
        )
        (run / "COMPLETE.json").write_text(json.dumps(complete))
        hashes = {
            n: digest(run / n) for n in ["score.json", "arrays.npz", "COMPLETE.json"]
        }
        verified = dict(
            config_sha256=digest(tmp_path / "paths.json"),
            model=row,
            outputs={} if supplemental else hashes,
        )
        (run / "VERIFIED.json").write_text(json.dumps(verified))
        if supplemental:
            (run / "VERIFIED-OUTPUT-HASHES.json").write_text(
                json.dumps(
                    dict(
                        original_verified_sha256=digest(run / "VERIFIED.json"),
                        outputs=hashes,
                    )
                )
            )
    collector["collect"](tmp_path, tmp_path / "output")
    assert (tmp_path / "output/parent-maps.npz").is_file()
    (tmp_path / "parent/score.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Annual output changed"):
        collector["collect"](tmp_path, tmp_path / "output")
