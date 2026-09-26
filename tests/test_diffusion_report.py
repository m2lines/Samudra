# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from samudra.experiments.diffusion_report import verified_annual_origins
from samudra.experiments.observation_pilot import digest


def test_annual_readiness_uses_frozen_campaign_audit_and_resolved_store(tmp_path):
    store = tmp_path / "bundles/annual"
    for origin in ("2015-01-01", "2018-01-01", "2021-01-01"):
        folder = store / "test" / origin
        folder.mkdir(parents=True)
        (folder / "COMPLETE.json").write_text("{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/annual_observations").symlink_to(store, target_is_directory=True)
    audit = tmp_path / "DATA_READY.json"
    audit.write_text(
        json.dumps(
            dict(annual=dict(path=str(store), files_verified=416, payload_bytes=100))
        )
    )
    signature = dict(staged_data_manifest_sha256=digest(audit))
    assert len(verified_annual_origins(tmp_path, signature)) == 3
    audit.write_text(audit.read_text() + "\n")
    with pytest.raises(ValueError, match="audit differs"):
        verified_annual_origins(tmp_path, signature)
    signature["staged_data_manifest_sha256"] = digest(audit)
    (tmp_path / "data/annual_observations").unlink()
    (tmp_path / "data/annual_observations").mkdir()
    with pytest.raises(ValueError, match="store differs"):
        verified_annual_origins(tmp_path, signature)
