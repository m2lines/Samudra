# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Patch features and the low-parameter baseline ladder.

All models predict the five-day tendency y = x[t+1] - x[t] of one channel at
one wet cell from a 3x3 patch of the two-state history (t-1, t) plus the
co-located tendencies of the other variables and the levels above and below.
Each returns a predictive distribution, scored by the probability it gives to
the true next value's bin.

Rungs (parameters are per channel):
  persistence  y ~ Laplace(0, b)                                   1
  ar1          y ~ Laplace(a * tendency, b)                         2
  lpc          y ~ Laplace(w_band . f, b_band); ridge, 3 lat bands  ~63
  lpc_ctx      lpc mean, Laplace scale from a 6x6 context table     ~96
               (local gradient x last-tendency), as in JPEG-LS/CALIC
  analog       kernel density over the outcomes of k nearest training
               patches (kernel analog forecasting); library + 1 bandwidth
  lpc_analog   lpc mean + kernel density over neighbours' lpc residuals
  mix          convex mixture of all of the above, weights by EM on val  6
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from coding import FLOOR, ToleranceBins, interval_prob, kernel_interval_prob
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]

DIRS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]  # (dy north, dx east)
FEATURES = [f"d_{dy}_{dx}" for dy, dx in DIRS] + [
    "tend",
    "lap_tend",
    "other0",
    "other1",
    "other2",
    "other3",
    "up_tend",
    "down_tend",
    "anom",
    "u_dTdx",
    "v_dTdy",
    "bias",
]
# Atmospheric forcing, as Samudra receives it (boundary_vars_key tau_hfds): the
# value at the latest input time t and its change since t-1. Inserted before
# "bias" when enabled, so every index above stays the same and bias stays last.
FORCING_VARS = ("tauuo", "tauvo", "hfds")
FORCING_FEATURES = [f"{f}{suffix}" for f in FORCING_VARS for suffix in ("", "_tend")]


def shift(f: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Value at (y+dy, x+dx): periodic in longitude, edge-replicated in latitude."""
    p = np.pad(f, ((1, 1), (0, 0)), mode="edge")
    return np.roll(p, -dx, axis=1)[1 + dy : 1 + dy + f.shape[0]]


def _z(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def patch_features(
    cur, prev, others, up, down, u, v, mean_c, inv_cos, forcing=None
) -> np.ndarray:
    """Full-grid features (H, W, d) for one time step. others: list of (cur, prev)
    pairs, up/down/u/v may be None. forcing: None, or one (cur, prev) pair per
    FORCING_VARS entry, which adds FORCING_FEATURES before the bias.
    """
    tend = cur - prev
    feats = []
    for dy, dx in DIRS:
        feats.append(_z(shift(cur, dy, dx) - cur))
    feats.append(_z(tend))
    nb = np.stack([shift(cur, dy, dx) - shift(prev, dy, dx) for dy, dx in DIRS[:4]])
    nb_tend = np.where(np.isfinite(nb), nb, tend[None])
    feats.append(_z(nb_tend.mean(0) - tend))
    for oc, op in others[:4]:
        feats.append(_z(oc - op))
    for _ in range(4 - len(others)):
        feats.append(np.zeros_like(cur))
    for pair in (up, down):
        feats.append(_z(pair[0] - pair[1]) if pair is not None else np.zeros_like(cur))
    feats.append(_z(cur - mean_c))
    dTdx = 0.5 * _z(shift(cur, 0, 1) - shift(cur, 0, -1)) * inv_cos
    dTdy = 0.5 * _z(shift(cur, 1, 0) - shift(cur, -1, 0))
    feats.append(_z(u) * dTdx if u is not None else np.zeros_like(cur))
    feats.append(_z(v) * dTdy if v is not None else np.zeros_like(cur))
    if forcing is not None:
        if len(forcing) != len(FORCING_VARS):
            raise ValueError(f"expected {len(FORCING_VARS)} forcing pairs")
        for fc, fp in forcing:
            feats += [_z(fc), _z(fc - fp)]
    feats.append(np.ones_like(cur))
    return np.stack(feats, axis=-1)


@dataclass
class Samples:
    F: np.ndarray  # (n, d) features
    y: np.ndarray  # (n,) tendency
    x: np.ndarray  # (n,) current value
    t: np.ndarray  # (n,) time index of the current state
    cell: np.ndarray  # (n,) flat cell index
    band: np.ndarray  # (n,) latitude band 0 tropics, 1 mid, 2 high
    grad: np.ndarray  # (n,) mean |neighbour difference|
    abs_tend: np.ndarray  # (n,) |x[t] - x[t-1]|


def lat_band(lat: np.ndarray) -> np.ndarray:
    a = np.abs(lat)
    return np.where(a < 20, 0, np.where(a < 50, 1, 2))


# ----------------------------------------------------------------- the ladder
def fit_ridge(F, y, lam=1e-3):
    sd = F.std(0)
    sd[sd == 0] = 1.0
    sd[-1] = 1.0  # bias
    Z = F / sd
    A = Z.T @ Z
    A[np.diag_indices_from(A)] += lam * np.trace(A) / len(A)
    w = np.linalg.solve(A, Z.T @ y)
    return w / sd


class Ladder:
    """Fit on train Samples; produce (mean, P(true bin)) on any Samples."""

    def __init__(self, k_nn=32, library=20000, seed=0):
        self.k_nn, self.library, self.rng = k_nn, library, np.random.default_rng(seed)

    # -- fitting
    def fit(self, tr: Samples):
        y = tr.y
        self.d = tr.F.shape[1]
        self.b_persist = np.mean(np.abs(y)) + 1e-30
        t = tr.F[:, FEATURES.index("tend")]
        self.a_ar1 = float(t @ y / max(t @ t, 1e-30))
        self.b_ar1 = np.mean(np.abs(y - self.a_ar1 * t)) + 1e-30
        self.w = {}
        self.b_lpc = {}
        r = np.zeros_like(y)
        for band in range(3):
            m = tr.band == band
            if m.sum() < 50:
                m = np.ones_like(m)
            self.w[band] = fit_ridge(tr.F[m], y[m])
            rb = y[m] - tr.F[m] @ self.w[band]
            self.b_lpc[band] = np.mean(np.abs(rb)) + 1e-30
        r = y - self.lpc_mean(tr)
        # context table (JPEG-LS-style): Laplace scale per (gradient, last-tendency) bucket
        self.g_edges = np.quantile(tr.grad, np.linspace(0, 1, 7)[1:-1])
        self.t_edges = np.quantile(tr.abs_tend, np.linspace(0, 1, 7)[1:-1])
        ctx = self._ctx(tr)
        b_glob = np.mean(np.abs(r)) + 1e-30
        sums = np.bincount(ctx, weights=np.abs(r), minlength=36)
        cnts = np.bincount(ctx, minlength=36)
        self.b_ctx = (sums + 20 * b_glob) / (cnts + 20)
        # analog library
        n = len(y)
        idx = self.rng.choice(n, size=min(self.library, n), replace=False)
        self.fsd = tr.F[:, :-1].std(0) + 1e-12
        self.nn = NearestNeighbors(n_neighbors=min(self.k_nn, len(idx))).fit(
            tr.F[idx, :-1] / self.fsd
        )
        self.lib_y = y[idx]
        self.lib_r = r[idx]
        self.h_min = 0.05 * b_glob
        self.c_analog = self.c_lpc_analog = 0.5
        self.mix_w: dict[str, float] | None = None
        return self

    def _ctx(self, s: Samples):
        return np.searchsorted(self.g_edges, s.grad) * 6 + np.searchsorted(
            self.t_edges, s.abs_tend
        )

    def lpc_mean(self, s: Samples):
        out = np.empty(len(s.y))
        for band, w in self.w.items():
            m = s.band == band
            out[m] = s.F[m] @ w
        return out

    # -- predictive distributions
    def predict(self, s: Samples, bins: ToleranceBins, with_knn=True):
        """Returns {model: (mean_tendency, prob_of_true_bin)}."""
        a, b = bins.tendency_interval(s.x, s.y)
        out = {}
        out["persistence"] = (
            np.zeros_like(s.y),
            interval_prob("laplace", 0.0, self.b_persist, a, b),
        )
        m = self.a_ar1 * s.F[:, FEATURES.index("tend")]
        out["ar1"] = (m, interval_prob("laplace", m, self.b_ar1, a, b))
        m = self.lpc_mean(s)
        scale = np.choose(s.band, [self.b_lpc[i] for i in range(3)])
        out["lpc"] = (m, interval_prob("laplace", m, scale, a, b))
        out["lpc_ctx"] = (
            m,
            interval_prob("laplace", m, self.b_ctx[self._ctx(s)], a, b),
        )
        if with_knn:
            _, nb = self.nn.kneighbors(s.F[:, :-1] / self.fsd)
            w = np.full(nb.shape, 1.0 / nb.shape[1])
            cy = self.lib_y[nb]
            h = self.c_analog * cy.std(1) + self.h_min
            out["analog"] = (cy.mean(1), kernel_interval_prob(cy, w, h, a, b))
            cr = m[:, None] + self.lib_r[nb]
            h = self.c_lpc_analog * cr.std(1) + self.h_min
            out["lpc_analog"] = (cr.mean(1), kernel_interval_prob(cr, w, h, a, b))
            self._last_nb = (nb, cy, cr, a, b)
        if self.mix_w is not None and with_knn:
            names = list(self.mix_w)
            P = np.stack([out[n][1] for n in names])
            M = np.stack([out[n][0] for n in names])
            wv = np.array([self.mix_w[n] for n in names])[:, None]
            out["mix"] = ((wv * M).sum(0), (wv * P).sum(0))
        return out

    def tune_on_val(self, va: Samples, bins: ToleranceBins, grid=(0.25, 0.5, 1.0, 2.0)):
        """Pick kernel bandwidth multipliers, then mixture weights (EM), on val."""
        self.predict(va, bins)
        nb, cy, cr, a, b = self._last_nb
        w = np.full(nb.shape, 1.0 / nb.shape[1])
        best = {}
        for name, C in (("analog", cy), ("lpc_analog", cr)):
            costs = [
                bins.bits(
                    kernel_interval_prob(C, w, c * C.std(1) + self.h_min, a, b), a, b
                ).mean()
                for c in grid
            ]
            best[name] = grid[int(np.argmin(costs))]
        self.c_analog, self.c_lpc_analog = best["analog"], best["lpc_analog"]
        out = self.predict(va, bins)
        names = list(out)
        q = bins.ref_prob(a, b)
        P = (1 - FLOOR) * np.stack([out[n][1] for n in names]) + FLOOR * q
        wv = np.full(len(names), 1.0 / len(names))
        for _ in range(200):
            R = wv[:, None] * P
            R /= R.sum(0, keepdims=True)
            wv = R.mean(1)
        self.mix_w = dict(zip(names, wv))
        return self

    def n_params(self):
        d = self.d
        return {
            "persistence": 1,
            "ar1": 2,
            "lpc": 3 * d + 3,
            "lpc_ctx": 3 * d + 36 + 10,
            "analog": 1,
            "lpc_analog": 3 * d + 1,
            "mix": 6,
        }
