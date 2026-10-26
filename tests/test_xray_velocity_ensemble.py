# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""The ensemble exporter rejects altered saved-conditioning provenance."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ensemble_export",
    Path(__file__).parents[1] / "scripts/export_xray_velocity_ensemble.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trace_provenance(tmp_path):
    names = ["grid.npz", "obs/initial-full/final.npz", "obs/initial-full/trace.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved trace")
    manifest = {
        "cases": {"obs": {"step": 3000}},
        "files": {name: module.digest(tmp_path / name) for name in names},
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(manifest))
    assert module.verify_trace(tmp_path, "obs", "initial-full")["step"] == 3000
    (tmp_path / names[1]).write_bytes(b"modified latent")
    with pytest.raises(ValueError, match="changed"):
        module.verify_trace(tmp_path, "obs", "initial-full")
