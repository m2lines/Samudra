# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical invariants of the experimental decoder edits and diagnostics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "onedeg_weight_edit", Path(__file__).parents[1] / "scripts/onedeg_weight_edit.py"
)
assert _spec is not None and _spec.loader is not None
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


@pytest.mark.parametrize(
    "family",
    [
        "spatial_contrast",
        "zonal_curvature",
        "meridional_curvature",
        "checkerboard",
        "velocity_direction",
    ],
)
def test_edit_preserves_constant_features_and_untouched_rows(family):
    generator = torch.Generator().manual_seed(912)
    weights = torch.randn(154, 280, 3, 3, generator=generator) * 0.02
    rows = [48, 67, 125, 144]
    delta = experiment.candidate_delta(weights, family, rows)
    untouched = torch.ones(154, dtype=torch.bool)
    untouched[rows] = False
    assert torch.count_nonzero(delta[untouched]) == 0
    features = torch.randn(1, 280, 1, 1, generator=generator).expand(1, 280, 7, 9)
    before = torch.nn.functional.conv2d(features, weights)
    after = torch.nn.functional.conv2d(features, weights + 0.2 * delta)
    torch.testing.assert_close(before, after, atol=2e-5, rtol=1e-5)


def test_land_values_never_enter_gradient_metric():
    mask = np.ones((77, 5, 6), dtype=bool)
    mask[:, 2, 2:4] = False
    static = {
        "mask": mask,
        "std": np.ones(77),
        "mean": np.zeros(77),
        "area": np.ones((5, 6)),
        "lev": np.arange(19),
        "lat": np.linspace(-40, 40, 5),
        "lon": np.arange(6) * 60,
    }
    diag = experiment.Diagnostics(static)
    constant_ocean = np.ones((2, 77, 5, 6))
    constant_ocean[:, ~mask] = 10000
    np.testing.assert_array_equal(diag.gradients(constant_ocean), 0)
    np.testing.assert_array_equal(diag.gradients(constant_ocean, coastal=True), 0)
    # The x seam is periodic, and a genuine wet-to-wet variation is detected.
    constant_ocean[..., 0] = 2
    assert (diag.gradients(constant_ocean) > 0).all()


def test_cached_decoder_effect_matches_direct_edit():
    generator = torch.Generator().manual_seed(3)
    weights = torch.randn(154, 280, 3, 3, generator=generator) * 0.02
    bias = torch.randn(154, generator=generator)
    features = torch.randn(1, 280, 8, 9, generator=generator)
    rows = [48, 67, 125, 144]
    delta = experiment.candidate_delta(weights, "zonal_curvature", rows)
    original = torch.nn.functional.conv2d(features, weights, bias)
    effect = torch.nn.functional.conv2d(features, delta)
    edited = torch.nn.functional.conv2d(features, weights + 0.1 * delta, bias)
    torch.testing.assert_close(original + 0.1 * effect, edited, atol=2e-5, rtol=1e-5)


def test_temporal_edit_preserves_pair_mean():
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(154, 280, 3, 3, generator=generator) * 0.02
    bias = torch.randn(154, generator=generator)
    delta, db = experiment.temporal_pair_delta(weight, bias, [48, 67, 125, 144])
    torch.testing.assert_close(delta[:77] + delta[77:], torch.zeros_like(delta[:77]))
    torch.testing.assert_close(db[:77] + db[77:], torch.zeros_like(db[:77]))
    updated = weight + delta
    torch.testing.assert_close(
        updated[[48, 67]], updated[[125, 144]], atol=1e-7, rtol=1e-5
    )


def test_carry_circuit_preserves_input_and_isolates_nonlinear_branches(tmp_path):
    from scripts.onedeg_weight_circuit import build_state

    generator = torch.Generator().manual_seed(71)
    state = {}
    shapes = {
        "unet.layers.0.skip_module": (280, 162, 1, 1),
        "unet.layers.0.convblock.6": (280, 4, 1, 1),
        "unet.layers.2.skip_module": (4, 280, 1, 1),
        "unet.layers.2.convblock.0": (4, 280, 3, 3),
        "unet.layers.14.skip_module": (280, 4, 1, 1),
        "unet.layers.14.convblock.6": (280, 4, 1, 1),
        "unet.layers.16.convblock.0": (4, 280, 3, 3),
        "unet.layers.16.convblock.6": (280, 4, 1, 1),
        "decoder": (154, 280, 3, 3),
    }
    for prefix, shape in shapes.items():
        state[f"module.{prefix}.weight"] = (
            torch.randn(shape, generator=generator) * 0.02
        )
        state[f"module.{prefix}.bias"] = (
            torch.randn(shape[0], generator=generator) * 0.02
        )
    checkpoint = tmp_path / "original.pt"
    torch.save({"model": state}, checkpoint)
    original_hash = experiment.sha256(checkpoint)
    basis, _ = torch.linalg.qr(torch.randn(280, 8, generator=generator))
    torch.save({"basis": basis}, tmp_path / "carry_basis.pt")
    edited, basis, channels = build_state(
        SimpleNamespace(
            checkpoint=checkpoint,
            output=tmp_path,
            min_level=15,
            basis="energy",
            alpha=0.5,
        )
    )
    x = torch.randn(162, 12, generator=generator)
    first = edited["unet.layers.0.skip_module.weight"][:, :, 0, 0] @ x
    first += edited["unet.layers.0.skip_module.bias"][:, None]
    # Arbitrary nonlinear residuals cannot overwrite the carried inputs.
    for prefix in (
        "unet.layers.0.convblock.6",
        "unet.layers.14.skip_module",
        "unet.layers.14.convblock.6",
        "unet.layers.16.convblock.6",
    ):
        first += edited[prefix + ".weight"][:, :, 0, 0] @ torch.randn(
            4, 12, generator=generator
        )
        first += edited[prefix + ".bias"][:, None]
    torch.testing.assert_close(
        basis.T @ first, x[[77 + c for c in channels]], atol=3e-6, rtol=1e-5
    )
    for prefix in (
        "unet.layers.2.skip_module",
        "unet.layers.2.convblock.0",
        "unet.layers.16.convblock.0",
    ):
        coupling = torch.einsum("oihw,ik->okhw", edited[prefix + ".weight"], basis)
        torch.testing.assert_close(
            coupling, torch.zeros_like(coupling), atol=1e-7, rtol=0
        )
    readout = torch.einsum("oihw,ik->okhw", edited["decoder.weight"], basis)
    expected = torch.zeros_like(readout)
    for t in range(2):
        expected[[77 * t + c for c in channels], :, 1, 1] = 0.5 * torch.eye(8)
    torch.testing.assert_close(readout, expected, atol=5e-7, rtol=1e-5)
    assert experiment.sha256(checkpoint) == original_hash
