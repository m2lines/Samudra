# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the pinned values in `test_metrics_spectra.py`.

`samudra.metrics.spectra.isotropic_spectrum` is a port, and the test that pins
it holds fixed numbers produced by the implementation it was ported from. This
script is how those numbers were produced and how they are checked again.

The reference lives outside this repository, so the numbers cannot be
regenerated from a clean checkout alone:

    source : YuanYuan98/Ocean_Emulator, viz_jupyter/obs_evaluation.py
    commit : 33d4ae2e
    symbols: compute_isotropic_spectrum_torch, _detrend_linear_torch

Run it with a checkout of that file:

    python tests/reference/regenerate_spectral_golden.py path/to/obs_evaluation.py

It prints the sampled values and the largest relative disagreement with this
repository's implementation. The call parameters below are the reference's
*figure* call sites, which differ from its function defaults -- `n_factor=2`
and `cutoff_before_bins=False` -- and matching them is the point.
"""

import sys

import numpy as np
import torch

from samudra.metrics import spectra

# The field the pinned test builds. Closed-form rather than random so the
# numbers do not depend on a generator implementation.
HEIGHT, WIDTH = 120, 240
DY = 0.125 * spectra.METRES_PER_DEGREE  # a 1/8 degree box, the DUACS grid
DX = DY * np.cos(np.deg2rad(37.5))
SAMPLED = [0, 7, 15, 22, 28, 36]


def field() -> np.ndarray:
    row, col = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    return (
        np.sin(2 * np.pi * col / 40.0) * np.cos(2 * np.pi * row / 25.0)
        + 0.5 * np.sin(2 * np.pi * (col + 2 * row) / 13.0)
        + 0.25 * np.cos(2 * np.pi * col / 7.0)
        + 0.01 * col
        - 0.02 * row
        + 3.0
    )


def reference(path: str):
    """Load the two reference functions out of the source file."""
    source = open(path).read()
    start = source.index("def _detrend_linear_torch")
    end = source.index("def process_dataset")
    namespace: dict = {"torch": torch, "np": np}
    exec(compile(source[start:end], path, "exec"), namespace)
    return namespace["compute_isotropic_spectrum_torch"]


def main(path: str) -> int:
    data = field()
    wavenumber, power = reference(path)(
        torch.tensor(data, dtype=torch.float64),
        dx=DX,
        dy=DY,
        n_factor=2,
        detrend="linear",
        window="hann",
        cutoff_before_bins=False,
    )
    wavenumber = wavenumber.numpy() * spectra.RADIANS_PER_KM
    power = power.numpy()

    ours_k, ours_p = spectra.isotropic_spectrum(data, dx=DX, dy=DY)
    ours_k = ours_k * spectra.RADIANS_PER_KM

    print(f"bins: {power.size}")
    print(f"sampled indices: {SAMPLED}")
    print("wavenumber:", [float(f"{wavenumber[i]:.12e}") for i in SAMPLED])
    print("power     :", [float(f"{power[i]:.12e}") for i in SAMPLED])
    print()
    print(f"max relative wavenumber difference: {_worst(ours_k, wavenumber):.2e}")
    print(f"max relative power difference     : {_worst(ours_p, power):.2e}")
    print(
        "\nBins whose edge coincides with a mode wavenumber can disagree in the\n"
        "last bit and move a handful of modes, so a few tenths of a percent on\n"
        "a small number of bins is expected; anything larger is not."
    )
    return 0


def _worst(ours: np.ndarray, theirs: np.ndarray) -> float:
    if ours.shape != theirs.shape:
        return float("inf")
    return float((np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-300)).max())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
