import math

import pytest
import torch

from ocean_emulators.constants import (
    DEPTH_I_LEVELS,
    DEPTH_INTERFACES,
    DEPTH_LEVELS,
    PROGNOSTIC_VARS,
    TensorMap,
    depth_of_channel,
)
from ocean_emulators.utils.multiton import MultitonScope


def test_vertical_velocity_is_off_by_default():
    """Turning W on is a deliberate edit to PROGNOSTIC_VARS, not something that
    should arrive by accident: it changes the channel count, and needs W in both
    the means/stds and the patch cache to train at all."""
    assert [name for name in PROGNOSTIC_VARS["all"] if name.startswith("W")] == []
    assert len(PROGNOSTIC_VARS["all"]) == 4 * len(DEPTH_I_LEVELS) + 1


def test_depth_interfaces_reproduce_the_llc_layer_thicknesses():
    """The faces are derived from the centres rather than hardcoded, so this
    pins the derivation to LLC4320's actual grid: drF starts 1.00, 1.14, 1.30,
    1.49 m and the surface face is exactly 0."""
    assert len(DEPTH_INTERFACES) == len(DEPTH_LEVELS) + 1
    assert DEPTH_INTERFACES[0] == 0.0

    thickness = [b - a for a, b in zip(DEPTH_INTERFACES, DEPTH_INTERFACES[1:])]
    assert thickness[:4] == pytest.approx([1.00, 1.14, 1.30, 1.49], abs=1e-9)
    assert all(t > 0 for t in thickness)
    # Each centre must sit halfway between its own two faces.
    for center, top, bottom in zip(
        DEPTH_LEVELS, DEPTH_INTERFACES, DEPTH_INTERFACES[1:]
    ):
        assert (top + bottom) / 2 == pytest.approx(center)


def test_depth_of_channel_puts_face_variables_on_interfaces():
    assert depth_of_channel("Theta_0") == pytest.approx(DEPTH_LEVELS[0])
    assert depth_of_channel("Salt_50") == pytest.approx(DEPTH_LEVELS[50])
    # W_i is the top face of cell i, so W_0 is the surface itself.
    assert depth_of_channel("W_0") == 0.0
    assert depth_of_channel("W_50") == pytest.approx(DEPTH_INTERFACES[50])
    assert depth_of_channel("W_7") != depth_of_channel("Theta_7")


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


def test_vertical_spacing_for_a_face_variable_is_the_layer_thickness(monkeypatch):
    """The load-bearing consequence of putting W on interfaces. Two W levels are
    one cell thick apart, whereas two Theta levels are one centre-to-centre
    spacing apart -- different numbers, and at depth different by metres. Getting
    this wrong would feed the vertical gradient loss the wrong dz for W."""
    monkeypatch.setitem(
        PROGNOSTIC_VARS,
        "all",
        PROGNOSTIC_VARS["all"] + [f"W_{level}" for level in DEPTH_I_LEVELS],
    )
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")

        w_spacing = tensor_map.vertical_spacing("W")
        theta_spacing = tensor_map.vertical_spacing("Theta")
        expected = torch.tensor(
            [b - a for a, b in zip(DEPTH_INTERFACES, DEPTH_INTERFACES[1:])][
                : len(w_spacing)
            ]
        )

        torch.testing.assert_close(w_spacing, expected)
        assert not torch.allclose(w_spacing, theta_spacing)
        # W is picked up as a 3D variable with no other change.
        assert tensor_map.VAR_SET_3D == ["U", "V", "Theta", "Salt", "W"]
        # And it landed after Eta, so nothing already indexed has moved.
        assert tensor_map.prognostic_var_names.index("Eta") == 204
        assert tensor_map.prognostic_var_names.index("W_0") == 205


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
