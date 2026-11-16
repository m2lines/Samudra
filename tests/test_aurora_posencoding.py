# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility with Aurora 1.8.0 without installing its model package."""

import json
from pathlib import Path

import pytest
import torch

from samudra.models.modules.fourier import (
    FourierExpansion,
    pos_expansion,
    scale_expansion,
)
from samudra.models.modules.posencoding import (
    lat_lon_meshgrid,
    pos_scale_enc,
    pos_scale_enc_grid,
)

# Recorded by calling aurora.model.posencoding.pos_scale_enc with the release's
# pos_expansion and scale_expansion. Provenance is embedded in the fixture.
_REFERENCE = json.loads(
    (Path(__file__).with_name("aurora_posencoding.json")).read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"])
def test_upstream_encoding_parity(case):
    # Float64 avoids platform-dependent float32 pooling/area rounding being
    # amplified by Aurora's shortest scale wavelength (about 0.0001 km).
    lat = torch.tensor(case["lat"], dtype=torch.float64)
    lon = torch.tensor(case["lon"], dtype=torch.float64)
    actual = pos_scale_enc(
        case["encode_dim"], lat, lon, case["patch"], pos_expansion, scale_expansion
    )
    expected = (torch.tensor(case["position"]), torch.tensor(case["scale"]))
    for encoding, reference in zip(actual, expected):
        assert encoding.dtype == torch.float32
        torch.testing.assert_close(encoding, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("case", _REFERENCE["cases"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coordinate_forms_agree(case, dtype):
    lat = torch.tensor(case["lat"], dtype=dtype)
    lon = torch.tensor(case["lon"], dtype=dtype)
    kwargs = dict(pos_expansion=pos_expansion, scale_expansion=scale_expansion)
    dim, patch = case["encode_dim"], tuple(case["patch"])
    expected = pos_scale_enc(dim, lat, lon, patch, **kwargs)
    grids = torch.meshgrid(lat, lon, indexing="ij")
    encodings = [pos_scale_enc(dim, grids[0], grids[1], list(patch), **kwargs)]
    batched = pos_scale_enc_grid(
        dim, torch.stack(grids).unsqueeze(0).expand(2, -1, -1, -1), patch, **kwargs
    )
    encodings.extend(zip(*batched))
    for encoding in encodings:
        for actual, reference in zip(encoding, expected):
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)


def test_scalar_patch_size():
    lat = torch.linspace(-75, 75, 6)
    lon = torch.linspace(15, 345, 6)
    scalar = pos_scale_enc(8, lat, lon, 2, pos_expansion, scale_expansion)
    pair = pos_scale_enc(8, lat, lon, (2, 2), pos_expansion, scale_expansion)
    for actual, expected in zip(scalar, pair):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_meshgrid_ordering():
    lat, lon = torch.tensor([-45.0, 45.0]), torch.tensor([0.0, 90.0, 180.0])
    grid = lat_lon_meshgrid(lat, lon)
    torch.testing.assert_close(grid[0], lat[:, None].expand(2, 3))
    torch.testing.assert_close(grid[1], lon[None, :].expand(2, 3))


def test_encoding_rejects_mixed_coordinate_dimensions():
    with pytest.raises(ValueError, match="both be vectors or both be matrices"):
        pos_scale_enc(
            8, torch.zeros(4), torch.zeros(4, 4), 2, pos_expansion, scale_expansion
        )


def test_fourier_wavelength_bounds():
    assert pos_expansion.lower == 0.01
    assert pos_expansion.upper == 720
    assert scale_expansion.lower == 0.00010814085263058073
    expansion = FourierExpansion(1, 4)
    # Zero and signed values are allowed, but out-of-range magnitudes are not.
    output = expansion(torch.tensor([0.0, -1.0, 4.0]), 4)
    torch.testing.assert_close(output[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(AssertionError, match="not within the configured range"):
        expansion(torch.tensor([0.5]), 4)
    with pytest.raises(ValueError, match="multiple of two"):
        expansion(torch.tensor([1.0]), 3)
