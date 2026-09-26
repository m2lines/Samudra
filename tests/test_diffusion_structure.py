# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.experiments.diffusion_structure import field_structure


def test_native_increments_wrap_longitude_but_exclude_coasts_and_latitude_wrap():
    field = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]])
    wet = torch.ones_like(field, dtype=torch.bool)
    result = field_structure(field, wet, torch.ones(2, 1))
    torch.testing.assert_close(result["zonal_increment_mse"], torch.tensor([3.0]))
    torch.testing.assert_close(
        result["meridional_increment_mse"], torch.tensor([100.0])
    )
    wet[..., 0] = False
    field[..., 0] = torch.nan
    result = field_structure(field, wet, torch.ones(2, 1))
    torch.testing.assert_close(result["zonal_increment_mse"], torch.tensor([1.0]))
    torch.testing.assert_close(result["zonal_pair_area"], torch.tensor([4.0]))
    field[..., 1] = torch.nan
    with pytest.raises(FloatingPointError):
        field_structure(field, wet, torch.ones(2, 1))


def test_member_variability_is_not_variability_of_ensemble_mean():
    members = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]], [[[-1.0, 1.0, -1.0, 1.0]]]])
    mask = torch.ones(1, 1, 4)
    each = field_structure(members, mask, torch.ones(1, 1))
    mean = field_structure(members.mean(0), mask, torch.ones(1, 1))
    torch.testing.assert_close(each["variance"], torch.ones(2, 1))
    torch.testing.assert_close(mean["variance"], torch.zeros(1))
    assert torch.isnan(each["meridional_increment_mse"]).all()


def test_area_weighting_and_empty_channels():
    values = torch.tensor([[[0.0, 2.0], [4.0, 6.0]], [[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor(
        [[[True, True], [True, True]], [[False, False], [False, False]]]
    )
    result = field_structure(values, mask, torch.tensor([[1.0], [3.0]]))
    assert result["mean"][0] == 4
    assert result["variance"][0] == 4
    assert torch.isnan(result["mean"][1])
    assert result["area"][1] == 0


def test_unsupported_structure_exports_as_json_null():
    import json

    from samudra.experiments.diffusion_report import json_statistics

    records = json_statistics(
        {
            "field": {
                "mean": torch.tensor([1.0, torch.nan]),
                "area": torch.tensor([2.0, 0.0]),
            }
        }
    )
    assert json.loads(json.dumps(records, allow_nan=False))["field"]["mean"] == [
        1.0,
        None,
    ]
