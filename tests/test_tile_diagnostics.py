"""Test the seam-fidelity diagnostic geometry and reductions."""

import numpy as np
import pytest

from ocean_emulators.tile_diagnostics import (
    derivative_jump,
    high_k_power_ratio,
    profile_along_axis,
    response_by_distance,
    seam_anomaly,
    seam_windows,
    signed_offsets,
)

# The live canonical grid: 720 cells, seam at 360, 16-cell overlap.
EXTENT = 720
SEAM = 360


def make_windows(half_width: int = 64):
    return seam_windows(
        axis="i",
        seam_centre=SEAM,
        extent=EXTENT,
        half_width=half_width,
        orthogonal_seam_centre=SEAM,
    )


def test_pseudo_seam_windows_are_interior_and_disjoint_from_the_real_seam() -> None:
    """This is what makes the comparison matched rather than apples-to-oranges."""
    windows = make_windows(half_width=64)
    assert windows.real_centre == SEAM
    assert windows.pseudo_centres == (180, 540)

    real = set(range(SEAM - 64, SEAM + 65))
    for centre in windows.pseudo_centres:
        span = set(range(centre - 64, centre + 65))
        assert not span & real, "pseudo-seam window touches the real seam"
        assert min(span) > 0 and max(span) < EXTENT - 1, "window hits the padded edge"


def test_every_window_has_the_same_width() -> None:
    windows = make_windows()
    assert len(signed_offsets(windows.half_width)) == 2 * windows.half_width + 1


def test_half_width_that_would_reach_the_seam_or_the_edge_is_rejected() -> None:
    """Rather than silently sampling contaminated cells."""
    with pytest.raises(ValueError, match="overlap the real seam"):
        seam_windows(axis="i", seam_centre=SEAM, extent=EXTENT, half_width=120)
    with pytest.raises(ValueError, match="run off"):
        seam_windows(axis="i", seam_centre=SEAM, extent=EXTENT, half_width=200)


def test_profile_reduces_to_a_curve_over_signed_distance() -> None:
    field = np.zeros((EXTENT, EXTENT))
    field[:, SEAM] = 3.0  # a spike exactly on the seam
    windows = make_windows(half_width=8)
    curve = profile_along_axis(
        field, axis="i", centre=SEAM, half_width=8,
        orthogonal_exclusion=windows.orthogonal_exclusion,
    )
    offsets = signed_offsets(8)
    assert curve.shape == offsets.shape
    assert curve[offsets == 0] == pytest.approx(3.0)
    assert np.all(curve[offsets != 0] == 0.0)


def test_orthogonal_exclusion_removes_the_other_seam() -> None:
    """Without it, the horizontal seam leaks into the vertical seam's curve."""
    field = np.zeros((EXTENT, EXTENT))
    field[SEAM, :] = 100.0  # the *orthogonal* seam
    without = profile_along_axis(field, axis="i", centre=SEAM, half_width=8)
    with_exclusion = profile_along_axis(
        field, axis="i", centre=SEAM, half_width=8,
        orthogonal_exclusion=(SEAM - 8, SEAM + 9),
    )
    assert without.max() > 0
    assert with_exclusion.max() == 0.0


def test_seam_anomaly_is_zero_on_a_statistically_uniform_field() -> None:
    """A field with no seam signature must produce no anomaly beyond noise."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((EXTENT, EXTENT))
    result = seam_anomaly(field, make_windows(half_width=64))
    assert result["seam"].shape == result["pseudo"].shape
    assert np.abs(result["anomaly"]).max() < 0.15


def test_seam_anomaly_isolates_an_injected_seam_signature() -> None:
    rng = np.random.default_rng(1)
    field = rng.standard_normal((EXTENT, EXTENT))
    field[:, SEAM - 1 : SEAM + 2] += 5.0
    result = seam_anomaly(field, make_windows(half_width=64))
    offsets = result["offsets"]
    assert result["anomaly"][offsets == 0] > 3.0
    far = np.abs(offsets) > 16
    assert np.abs(result["anomaly"][far]).max() < 0.3


def test_seam_anomaly_survives_a_background_gradient() -> None:
    """The point of pseudo-seams: smooth spatial variation must cancel.

    A raw curve would rise across a gradient and look like a seam. The real and
    pseudo windows both sample the gradient, so the anomaly should not.
    """
    ramp = np.tile(np.linspace(0.0, 10.0, EXTENT), (EXTENT, 1))
    result = seam_anomaly(ramp, make_windows(half_width=64))
    assert result["seam"].max() - result["seam"].min() > 1.0  # raw curve does vary
    assert np.abs(result["anomaly"]).max() < 1e-9  # anomaly does not


def test_derivative_jump_spikes_at_a_hard_stitch() -> None:
    """A kink is a delta function in the derivative; that is the seam's fingerprint."""
    smooth = np.tile(np.linspace(0.0, 1.0, EXTENT), (EXTENT, 1))
    kinked = smooth.copy()
    kinked[:, SEAM:] += 0.5  # a hard offset, exactly what blending removes

    smooth_jump = derivative_jump(smooth, axis="i")
    kinked_jump = derivative_jump(kinked, axis="i")
    assert smooth_jump.shape == smooth.shape
    assert np.abs(kinked_jump[:, SEAM - 1]).mean() > 100 * np.abs(smooth_jump).mean()


def test_high_k_power_ratio_detects_a_smoothed_field() -> None:
    """Over-broad blending eats small scales; the ratio drops below one."""
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((64, 64))
    smoothed = truth.copy()
    for _ in range(6):  # repeated 3-point average kills high wavenumbers
        smoothed = 0.25 * np.roll(smoothed, 1, 0) + 0.5 * smoothed + 0.25 * np.roll(smoothed, -1, 0)

    ratio = high_k_power_ratio(smoothed, truth, axis="i")
    assert np.nanmedian(ratio) < 0.5
    same = high_k_power_ratio(truth, truth, axis="i")
    np.testing.assert_allclose(np.nanmedian(same), 1.0, rtol=1e-10)


def test_response_by_distance_recovers_a_decaying_signal() -> None:
    """The far-field readout must distinguish decay from a flat floor."""
    height = width = 64
    centre = (32, 32)
    j = np.arange(height)[:, None] - centre[0]
    i = np.arange(width)[None, :] - centre[1]
    distance = np.sqrt(j**2 + i**2)

    decaying = np.exp(-distance / 5.0)[None]
    flat = np.ones((1, height, width))

    bins, decay_curve = response_by_distance(decaying, centre=centre, num_bins=16)
    _, flat_curve = response_by_distance(flat, centre=centre, num_bins=16)

    assert bins.shape == (16,)
    assert decay_curve.shape == (1, 16)
    assert decay_curve[0, 0] > 10 * decay_curve[0, -1]        # decays
    np.testing.assert_allclose(flat_curve[0], 1.0, rtol=1e-10)  # floor stays flat


def test_response_by_distance_keeps_leading_dimensions() -> None:
    response = np.ones((3, 5, 16, 16))
    _, curve = response_by_distance(response, centre=(8, 8), num_bins=8)
    assert curve.shape == (3, 5, 8)


def test_max_reduction_ranks_a_hard_stitch_worse_than_a_blend() -> None:
    """The discriminating test for the whole suite: it must rank rung 2 below rung 3.

    A hard stitch puts all of its excess variation in one column; a blend spreads
    the same total across the overlap. Under a MEAN the spike gets divided by the
    window width and the hard stitch scores *better*, which is backwards. Under a
    MAX it ranks correctly. This pins the reduction the notebook uses.
    """
    rng = np.random.default_rng(3)
    truth = rng.standard_normal((EXTENT, EXTENT))

    hard = truth.copy()
    hard[:, SEAM:] += 0.5  # one discontinuous column

    # Same total step, spread smoothly over a 16-cell overlap.
    blended = truth.copy()
    ramp = np.linspace(0.0, 1.0, 16)
    blended[:, SEAM - 8 : SEAM + 8] += 0.5 * ramp
    blended[:, SEAM + 8 :] += 0.5

    windows = make_windows(half_width=64)
    band = np.abs(signed_offsets(64)) <= 12

    peaks, means = {}, {}
    for label, field in [("hard", hard), ("blend", blended)]:
        excess = np.abs(derivative_jump(field, axis="i")) - np.abs(
            derivative_jump(truth, axis="i")
        )
        anomaly = seam_anomaly(excess, windows, reduce="absmean")["anomaly"]
        peaks[label] = anomaly[band].max()
        means[label] = anomaly[band].mean()

    assert peaks["hard"] > 5 * peaks["blend"], "max must rank the hard stitch worse"
    # And the trap this guards against: the mean does not separate them usefully.
    assert means["hard"] < 2 * means["blend"]


def test_derivative_jump_signature_sits_just_below_the_discontinuity() -> None:
    """Forward differencing puts element k at the k/k+1 interface, so a knife-edge
    band centred on the seam can miss the spike. The notebook's band allows for it."""
    truth = np.zeros((32, EXTENT))
    field = truth.copy()
    field[:, SEAM:] += 1.0

    excess = np.abs(derivative_jump(field, axis="i"))
    peak_column = int(np.argmax(excess.mean(axis=0)))
    assert peak_column == SEAM - 1
