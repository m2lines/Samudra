# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Coding primitives shared by every model: a tolerance quantization per
variable, interval probabilities for parametric and kernel predictive
distributions, and blosc-based compressibility diagnostics.

Every forecaster is scored as a compressor: the cost of the true next value is
-log2 P(true bin), where P comes from the model's predictive distribution over
the next value, integrated over that value's bin.

Bins (protocol v2). Following Keisler (2022, arXiv:2202.07575, sec. 3.3.3),
each *variable* is scaled by the spread of its time differences, pooled over all
levels, so one physical unit counts the same at every depth. The bin width is
w = eps * sigma_delta(variable): bins are unbounded integers floor(x / w), so
there is no range to fit and no end bins.

Floor. Each model's probability is mixed with weight FLOOR into a reference
distribution, Laplace(persistence, REF_MULT * sigma_delta), so every cost is
finite. The reference is the same for all models, so differences are exact.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numcodecs import Blosc  # type: ignore[import-untyped]
from scipy.special import ndtr

FLOOR = 1e-9
REF_MULT = 10.0  # reference Laplace scale, in units of sigma_delta


class ToleranceBins:
    """Unbounded uniform bins of a given width: bin i covers [i*w, (i+1)*w)."""

    def __init__(self, width: float, sigma_delta: float):
        if not (width > 0 and sigma_delta > 0):
            raise ValueError(
                f"need positive width and sigma, got {width}, {sigma_delta}"
            )
        self.width = float(width)
        self.sigma_delta = float(sigma_delta)
        self.ref_scale = REF_MULT * self.sigma_delta

    def index(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("cannot bin non-finite values")
        return np.floor(x / self.width).astype(np.int64)

    def edges(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = idx * self.width
        return a, a + self.width

    def tendency_interval(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """The bin of the next value x + y, in tendency coordinates (minus x)."""
        a, b = self.edges(self.index(np.asarray(x) + np.asarray(y)))
        return a - x, b - x

    def ref_prob(self, a, b) -> np.ndarray:
        """Reference (floor) probability of the tendency interval (a, b]."""
        return interval_prob("laplace", 0.0, self.ref_scale, a, b)

    def bits(self, p, a, b) -> np.ndarray:
        """Cost of the true bin given model probability p and its interval."""
        return bits(p, self.ref_prob(a, b))


def _laplace_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    neg = z < 0
    out[neg] = 0.5 * np.exp(z[neg])
    out[~neg] = 1.0 - 0.5 * np.exp(-z[~neg])
    return out


CDFS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gauss": ndtr,
    "laplace": _laplace_cdf,
}


def interval_prob(family: str, mu, scale, a, b) -> np.ndarray:
    """P(a < Y <= b) for Y ~ family(mu, scale). Arrays broadcast."""
    cdf = CDFS[family]
    scale = np.maximum(scale, 1e-30)
    za = np.asarray((a - mu) / scale, dtype=np.float64)
    zb = np.asarray((b - mu) / scale, dtype=np.float64)
    if family == "laplace":
        # In the upper tail use survival functions, so narrow far-tail bins keep
        # their (tiny) probability instead of cancelling to 0 in 1 - 1.
        upper = za > 0
        sa = 0.5 * np.exp(-np.where(upper, za, 0.0))
        sb = 0.5 * np.exp(-np.where(upper, zb, 0.0))
        p = np.where(upper, sa - sb, cdf(zb) - cdf(za))
    else:
        p = cdf(zb) - cdf(za)
    return np.clip(p, 0.0, 1.0)


def kernel_interval_prob(centers, weights, bandwidth, a, b) -> np.ndarray:
    """Gaussian-kernel mixture. centers/weights: (n, k); bandwidth, a, b: (n,)."""
    h = np.maximum(bandwidth, 1e-30)[:, None]
    p = ndtr((b[:, None] - centers) / h) - ndtr((a[:, None] - centers) / h)
    return np.clip((weights * p).sum(axis=1), 0.0, 1.0)


def bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """-log2 of the model probability p floored by the reference probability q."""
    return -np.log2((1.0 - FLOOR) * p + FLOOR * q)


# ---------------------------------------------------------------- blosc side quest
_BLOSC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)  # matches the OM4 stores


def blosc_bits_per_value(codes: np.ndarray, n_values: int) -> float:
    """Compressed size of an integer array, in bits per (wet) value."""
    buf = np.ascontiguousarray(codes)
    return 8.0 * len(_BLOSC.encode(buf)) / max(n_values, 1)


def quantized_field(x: np.ndarray, wet: np.ndarray, bins: ToleranceBins) -> np.ndarray:
    """Bin indices of a 2D field as int32 with land set to 0 (what a Zarr chunk sees)."""
    q = np.zeros(x.shape, dtype=np.int32)
    q[wet] = bins.index(x[wet]).astype(np.int32)
    return q
