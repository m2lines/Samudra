# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Does a compressor make a better analog-distance than Euclidean distance?

Same patch features, same library, same kernel density over neighbour
outcomes; only the neighbour search changes. Features are standardized and
quantized to one byte each so every distance sees identical inputs. NCD is
the Jiang et al. (2023) normalized compression distance with gzip (zlib) and
with blosc-zstd (what our Zarr stores use).

    python ncd_ablation.py CACHE OUT --channel thetao_0
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
from coding import Bins, bits, kernel_interval_prob
from numcodecs import Blosc
from run import Cache, build_samples, subset

ZSTD = Blosc(cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE)
COMPRESSORS = {
    "gzip": lambda b: len(zlib.compress(b, 9)),
    "blosc_zstd": lambda b: len(ZSTD.encode(np.frombuffer(b, np.uint8))),
}


def to_bytes(F, mu, sd):
    q = np.clip(np.round((F - mu) / sd * 32 + 128), 0, 255).astype(np.uint8)
    return q, [r.tobytes() for r in q]


def ncd_neighbours(qb, lb, clen, k):
    cl = np.array([clen(b) for b in lb])
    out = np.empty((len(qb), k), dtype=np.int64)
    for i, x in enumerate(qb):
        cx = clen(x)
        cxy = np.array([clen(x + y) for y in lb])
        d = (cxy - np.minimum(cx, cl)) / np.maximum(cx, cl)
        out[i] = np.argsort(d, kind="stable")[:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--channel", default="thetao_0")
    ap.add_argument("--library", type=int, default=3000)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--bins", type=int, default=8192)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cache = Cache(a.cache)
    c = a.channel
    vals = np.asarray(cache.arr("train", c))[:, cache.masks[cache.meta["levels"][c]]]
    bins = Bins.fit(vals, a.bins)
    mean_c = float(np.nanmean(vals))
    tr = build_samples(cache, "train", c, mean_c)
    te = build_samples(cache, "test", c, mean_c)
    lib = subset(tr, rng.choice(len(tr.y), a.library, replace=False))
    qry = subset(te, rng.choice(len(te.y), a.queries, replace=False))
    mu, sd = lib.F[:, :-1].mean(0), lib.F[:, :-1].std(0) + 1e-12
    lq, lb = to_bytes(lib.F[:, :-1], mu, sd)
    qq, qb = to_bytes(qry.F[:, :-1], mu, sd)
    lo, hi = bins.edges(bins.index(qry.x + qry.y))
    lo, hi = lo - qry.x, hi - qry.x
    res = {
        "channel": c,
        "library": a.library,
        "queries": a.queries,
        "k": a.k,
        "bytes_per_patch": int(lq.shape[1]),
        "methods": {},
    }

    def score(name, nb, secs):
        C = lib.y[nb]
        w = np.full(nb.shape, 1.0 / nb.shape[1])
        h_min = 0.05 * np.mean(np.abs(lib.y))
        best = min(
            bits(
                kernel_interval_prob(C, w, m * C.std(1) + h_min, lo, hi), bins.k
            ).mean()
            for m in (0.25, 0.5, 1.0, 2.0)
        )
        res["methods"][name] = {
            "bits": float(best),
            "rmse": float(np.sqrt(np.mean((C.mean(1) - qry.y) ** 2))),
            "seconds": secs,
        }
        print(
            f"[ncd] {name:12s} {best:.3f} bits  rmse {res['methods'][name]['rmse']:.4g}  ({secs:.1f}s)",
            flush=True,
        )

    t0 = time.time()
    d = ((qq[:, None, :].astype(np.float32) - lq[None].astype(np.float32)) ** 2).sum(-1)
    score("euclidean", np.argsort(d, 1)[:, : a.k], time.time() - t0)
    score("random", rng.integers(0, a.library, (a.queries, a.k)), 0.0)
    for name, clen in COMPRESSORS.items():
        t0 = time.time()
        score(f"ncd_{name}", ncd_neighbours(qb, lb, clen, a.k), time.time() - t0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"ncd_{c}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
