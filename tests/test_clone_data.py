# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from scripts.clone_data import DEFAULT_DATA_ROOT, source_store


def test_default_clone_source_uses_canonical_september_release():
    assert DEFAULT_DATA_ROOT == (
        "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/"
    )
    assert source_store(DEFAULT_DATA_ROOT, "OM4") == f"{DEFAULT_DATA_ROOT}OM4.zarr"


def test_clone_source_accepts_root_without_trailing_slash():
    assert source_store("https://example.test/data", "OM4_means") == (
        "https://example.test/data/OM4_means.zarr"
    )
