# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.experiments.diffusion_geometry import RegisteredGridMap


def test_nonuniform_latitudes_interpolate_by_coordinates_and_preserve_source():
    latitude = torch.tensor([-80.0, -20.0, 10.0, 70.0])
    longitude = torch.tensor([0.0, 90.0, 180.0, 270.0])
    target = torch.tensor([-60.0, -5.0, 40.0])
    source = latitude[None, None, :, None].expand(1, 1, 4, 4).clone().requires_grad_()
    before = source.detach().clone()
    mapper = RegisteredGridMap(latitude, longitude, target, longitude)
    result = mapper(source)
    torch.testing.assert_close(
        result, target[None, None, :, None].expand_as(result), atol=1e-5, rtol=1e-6
    )
    result.sum().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    torch.testing.assert_close(source.detach(), before, rtol=0, atol=0)
    assert list(mapper.parameters()) == []


def test_longitude_wrap_and_polar_extension_use_registered_centers():
    source = torch.tensor([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]])
    mapper = RegisteredGridMap(
        [-60.0, 60.0], [0.0, 90.0, 180.0, 270.0], [-80.0, 80.0], [-45.0, 315.0, 675.0]
    )
    result = mapper(source)
    torch.testing.assert_close(
        result, torch.tensor([[[[1.5, 1.5, 1.5], [5.5, 5.5, 5.5]]]])
    )
    with pytest.raises(ValueError, match="Latent grid"):
        mapper(source[..., :2])
    with pytest.raises(ValueError, match="strictly increase"):
        RegisteredGridMap([60.0, -60.0], [0.0, 180.0], [0.0], [0.0])
