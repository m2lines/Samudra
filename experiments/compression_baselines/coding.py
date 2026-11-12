# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Coding primitives shared by every model: one fixed quantization per channel,
interval probabilities for parametric and kernel predictive distributions, and
blosc-based compressibility diagnostics.

Every forecaster is scored as a compressor: the cost of the true next value is
-log2 P(true bin), where P comes from the model's predictive distribution over
the *next value*, integrated over that value's bin. A tiny uniform floor keeps
costs finite; it is identical for all models, so differences are exact.
"""

from __future__ import annotations

import numpy as np
from numcodecs import Blosc
from scipy.special import ndtr

FLOOR = 1e-9  # uniform mixing weight; bounds the worst-case cost at ~log2(K/FLOOR)


class Bins:
    """K uniform bins over [lo, hi]; the two end bins absorb the tails."""

    def __init__(self, lo: float, hi: float, k: int):
        self.lo, self.hi, self.k = float(lo), float(hi), int(k)
        self.width = (self.hi - self.lo) / self.k

    @classmethod
    def fit(
        cls, values: np.ndarray, k: int, pad: float = 0.05, tail: float = 1e-5
    ) -> Bins:
        """Range from robust quantiles so a few outliers cannot widen every bin;
        values outside fall in the end bins.
        """
        v = values[np.isfinite(values)]
        lo, hi = (float(q) for q in np.quantile(v, [tail, 1 - tail]))
        span = max(hi - lo, 1e-12)
        return cls(lo - pad * span, hi + pad * span, k)

    def index(self, x: np.ndarray) -> np.ndarray:
        i = np.floor((x - self.lo) / self.width)
        return np.clip(np.nan_to_num(i), 0, self.k - 1).astype(np.int64)

    def edges(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.lo + idx * self.width
        b = a + self.width
        a = np.where(idx == 0, -np.inf, a)
        b = np.where(idx == self.k - 1, np.inf, b)
        return a, b

    def center(self, idx: np.ndarray) -> np.ndarray:
        return self.lo + (idx + 0.5) * self.width


def _laplace_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    neg = z < 0
    out[neg] = 0.5 * np.exp(z[neg])
    out[~neg] = 1.0 - 0.5 * np.exp(-z[~neg])
    return out


CDFS = {"gauss": ndtr, "laplace": _laplace_cdf}


def interval_prob(family: str, mu, scale, a, b) -> np.ndarray:
    """P(a < Y <= b) for Y ~ family(mu, scale). Arrays broadcast."""
    cdf = CDFS[family]
    scale = np.maximum(scale, 1e-30)
    return np.clip(cdf((b - mu) / scale) - cdf((a - mu) / scale), 0.0, 1.0)


def kernel_interval_prob(centers, weights, bandwidth, a, b) -> np.ndarray:
    """Gaussian-kernel mixture. centers/weights: (n, k); bandwidth, a, b: (n,)."""
    h = np.maximum(bandwidth, 1e-30)[:, None]
    p = ndtr((b[:, None] - centers) / h) - ndtr((a[:, None] - centers) / h)
    return np.clip((weights * p).sum(axis=1), 0.0, 1.0)


def bits(p: np.ndarray, k: int) -> np.ndarray:
    return -np.log2((1.0 - FLOOR) * p + FLOOR / k)


# ---------------------------------------------------------------- blosc side quest
_BLOSC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)  # matches the OM4 stores


def blosc_bits_per_value(codes: np.ndarray, n_values: int) -> float:
    """Compressed size of an integer array, in bits per (wet) value."""
    buf = np.ascontiguousarray(codes)
    return 8.0 * len(_BLOSC.encode(buf)) / max(n_values, 1)


def quantized_field(x: np.ndarray, wet: np.ndarray, bins: Bins) -> np.ndarray:
    """Bin indices of a 2D field as uint16 with land set to 0 (what a Zarr chunk sees)."""
    q = np.zeros(x.shape, dtype=np.uint16)
    q[wet] = bins.index(x[wet]).astype(np.uint16)
    return q
