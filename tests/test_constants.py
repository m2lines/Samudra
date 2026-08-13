import math

import pytest
import torch

from ocean_emulators.constants import DEPTH_LEVELS, TensorMap
from ocean_emulators.utils.multiton import MultitonScope


def test_channel_depth_centers_are_nan_only_for_2d_channels():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        centers = tensor_map.channel_depth_centers

        assert centers.shape == (len(tensor_map.prognostic_var_names),)
        for name, center in zip(tensor_map.prognostic_var_names, centers.tolist()):
            if "_" in name:
                assert center == pytest.approx(
                    DEPTH_LEVELS[int(name.rsplit("_", 1)[-1])]
                )
            else:
                assert math.isnan(center)


def test_vertical_spacing_matches_the_depth_level_table():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        expected = torch.tensor([n - p for p, n in zip(DEPTH_LEVELS, DEPTH_LEVELS[1:])])

        for variable in tensor_map.VAR_SET_3D:
            torch.testing.assert_close(tensor_map.vertical_spacing(variable), expected)


def test_vertical_spacing_is_strongly_non_uniform():
    """The reason the vertical gradient loss cannot treat a level index as a unit
    of depth: LLC4320 levels are ~42x further apart at the floor than at the
    surface."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        spacing = tensor_map.vertical_spacing("Theta")

        assert spacing.min() == pytest.approx(1.07, abs=1e-3)
        assert spacing.max() == pytest.approx(45.46, abs=1e-3)
        assert spacing.max() / spacing.min() > 40.0


def test_vertical_spacing_rejects_a_2d_variable():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        with pytest.raises(ValueError, match="at least two depth levels"):
            tensor_map.vertical_spacing("Eta")


def test_vertical_spacing_rejects_a_single_level_variable():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("single_1", "all")
        with pytest.raises(ValueError, match="at least two depth levels"):
            tensor_map.vertical_spacing("Theta")


def test_vertical_spacing_rejects_an_unknown_variable():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        with pytest.raises(ValueError, match="not a prognostic variable"):
            tensor_map.vertical_spacing("W")


def test_var_set_3d_has_one_entry_per_variable():
    """It is a set of variable names, not one name per channel: the vertical loss
    iterates it to reach each whole depth column exactly once."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")

        assert tensor_map.VAR_SET_3D == ["U", "V", "Theta", "Salt"]
        assert tensor_map.VAR_SET_2D == ["Eta"]
