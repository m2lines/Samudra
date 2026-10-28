# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from physical_xray import PhysicalXray
from prepare_physical_xray import digest, prepare


def test_verified_exports_and_unavailable_interior(tmp_path):
    source = tmp_path / "source"
    annual = source / "annual"
    annual.mkdir(parents=True)
    shape = (77, 4, 8)
    np.savez(
        annual / "full-state-2018-01-01.npz",
        channel_names=[f"field{i}" for i in range(77)],
        mask=np.ones(shape, bool),
        lat=np.linspace(-80, 80, 4),
        lon=np.arange(8),
        normalization_mean=np.ones(77),
        normalization_std=np.full(77, 2),
        initial_mean=np.full((2, *shape), 5.0),
        initial_draw1=np.full((2, *shape), 7.0),
        initial_spread=np.ones((2, *shape)),
        mean=np.full((2, *shape), 5.0),
        draw1=np.full((2, *shape), 7.0),
        spread=np.ones((2, *shape)),
        leads_days=[30, 365],
    )
    np.savez(annual / "2018-01-01.npz", reference=np.full((73, 4, 4, 8), 3.0))
    np.savez(
        annual / "maps-2018-01-01.npz",
        initial_surface_reference=np.full((2, 4, 8), 3.0),
    )
    (source / "COMPLETE.json").write_text(
        json.dumps(
            {
                "inputs": {"step": 1200},
                "files": {
                    str(p.relative_to(source)): digest(p) for p in annual.iterdir()
                },
            }
        )
    )
    output = tmp_path / "bundle"
    prepare([source], output)
    page = PhysicalXray(output)
    page.field.value = 0
    assert "No aligned reference" in page.metrics.object
    page.field.value = 38
    assert "**2**" in page.metrics.object
    page.readout.value = "Member 1"
    assert "**4**" in page.metrics.object
    page.units.value = "Normalized"
    assert "**2**" in page.metrics.object
    page.case.value = 2
    assert "day 365" in page.caption.object
    (output / "case-000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        PhysicalXray(output)


def test_om4_interior_reference_and_preserved_base(tmp_path):
    source = tmp_path / "source"
    om4 = source / "om4"
    om4.mkdir(parents=True)
    shape = (77, 4, 8)
    np.savez(
        om4 / "full-state-2015-01-01.npz",
        channel_names=[f"field{i}" for i in range(77)],
        mask=np.ones(shape, bool),
        lat=np.linspace(-80, 80, 4),
        lon=np.arange(8),
        normalization_mean=np.ones(77),
        normalization_std=np.full(77, 2),
        initial_mean=np.full((2, *shape), 5.0),
        initial_draw1=np.full((2, *shape), 7.0),
        initial_spread=np.ones((2, *shape)),
        mean=np.full((1, *shape), 5.0),
        draw1=np.full((1, *shape), 7.0),
        spread=np.ones((1, *shape)),
        leads_days=[30],
        gold_initial=np.full((2, *shape), 3.0),
        gold=np.full((1, *shape), 3.0),
    )
    (source / "COMPLETE.json").write_text(
        json.dumps(
            {
                "inputs": {"step": 1200},
                "files": {str(p.relative_to(source)): digest(p) for p in om4.iterdir()},
            }
        )
    )
    base = tmp_path / "base"
    prepare([source], base)
    output = tmp_path / "extended"
    prepare([source], output, base)
    page = PhysicalXray(output)
    page.field.value = 60
    assert "**2**" in page.metrics.object
    assert "OM4" in page.caption.object
    assert "not ocean observations" in page.caption.object
    assert len(page.cases) == 4
    assert digest(output / "case-000.npz") == digest(base / "case-000.npz")
