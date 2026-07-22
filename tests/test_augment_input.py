# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.models.modules.augment_input import (
    ProcessorGeometryConditioner,
    make_position_scale_grid,
)


def test_position_scale_grid_contains_finite_geometry():
    lat = torch.tensor([-67.5, -22.5, 22.5, 67.5])
    lon = torch.arange(8) * 45.0 + 22.5

    geometry = make_position_scale_grid(lat, lon)

    assert geometry.shape == (4, 4, 8)
    assert torch.isfinite(geometry).all()
    torch.testing.assert_close(geometry[:3].square().sum(dim=0), torch.ones(4, 8))
    torch.testing.assert_close(geometry[3].square().mean(), torch.tensor(1.0))


def test_processor_geometry_sidecar_is_exact_noop_at_initialization():
    conditioner = ProcessorGeometryConditioner(channels=6)
    features = torch.randn(2, 6, 4, 8)
    resolution = (
        torch.tensor([-67.5, -22.5, 22.5, 67.5]),
        torch.arange(8) * 45.0 + 22.5,
    )

    output = conditioner(features, resolution)

    torch.testing.assert_close(output, features)
    output.sum().backward()
    assert conditioner.projection.weight.grad is not None
    assert torch.count_nonzero(conditioner.projection.weight.grad) > 0


def test_processor_geometry_sidecar_rejects_grid_mismatch():
    conditioner = ProcessorGeometryConditioner(channels=6)

    with pytest.raises(ValueError, match="disagree"):
        conditioner(
            torch.randn(2, 6, 3, 8),
            (
                torch.tensor([-67.5, -22.5, 22.5, 67.5]),
                torch.arange(8) * 45.0 + 22.5,
            ),
        )
