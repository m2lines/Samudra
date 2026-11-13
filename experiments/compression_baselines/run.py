# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Fit the baseline ladder channel by channel and score it in bits per value.

    python run.py CACHE OUT --eps 0.0625 [--n-eval 8000] [--channels thetao_0,so_0]

Writes OUT/results.json (per-channel tables) and OUT/maps.npz (bit-gain maps).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from coding import ToleranceBins, blosc_bits_per_value, quantized_field
from models import FORCING_VARS, Ladder, Samples, lat_band, patch_features

MAP_CHANNELS = ("thetao_0", "thetao_7", "so_0", "uo_0", "zos")


class Cache:
    def __init__(self, root: Path):
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text())
        st = np.load(root / "static.npz")
        self.masks, self.lat, self.area = st["masks"], st["lat"], st["area"]
        self.dz = st["dz"]
        self.inv_cos = 1.0 / np.maximum(np.cos(np.deg2rad(self.lat)), 0.1)
        self.band = lat_band(self.lat)
        chans = self.meta["channels"]
        self.levels_of: dict[str, list[int]] = {}
        for c in chans:
            if c != "zos":
                v, k = c.rsplit("_", 1)
                self.levels_of.setdefault(v, []).append(int(k))

    def arr(self, split, c):
        return np.load(self.root / split / f"{c}.npy", mmap_mode="r")

    def neighbours(self, c):
        """Other variables at the same level, levels above/below, and velocities."""
        if c == "zos":
            k0 = self.levels_of["thetao"][0]
            return (
                [f"{v}_{k0}" for v in ("thetao", "so", "uo", "vo")],
                None,
                None,
                f"uo_{k0}",
                f"vo_{k0}",
            )
        v, k = c.rsplit("_", 1)
        ks = self.levels_of[v]
        i = ks.index(int(k))
        others = [f"{o}_{k}" for o in ("thetao", "so", "uo", "vo") if o != v] + (
            ["zos"] if i == 0 else []
        )
        up = f"{v}_{ks[i - 1]}" if i > 0 else None
        dn = f"{v}_{ks[i + 1]}" if i + 1 < len(ks) else None
        return others, up, dn, f"uo_{k}", f"vo_{k}"


def build_samples(
    cache: Cache, split: str, c: str, mean_c: float, forcing: bool = False
) -> Samples:
    A = cache.arr(split, c)
    if forcing:
        missing = [f for f in FORCING_VARS if f not in cache.meta.get("forcing", [])]
        if missing:
            raise ValueError(
                f"cache has no forcing {missing}; run data.py CACHE --add-forcing"
            )
    forc = [cache.arr(split, f) for f in FORCING_VARS] if forcing else []
    others, up, dn, un, vn = cache.neighbours(c)
    other_arrs = [cache.arr(split, o) for o in others]
    U = cache.arr(split, up) if up else None
    D = cache.arr(split, dn) if dn else None
    Uv, Vv = cache.arr(split, un), cache.arr(split, vn)
    wet = cache.masks[cache.meta["levels"][c]]
    flat_band = cache.band.ravel()
    parts = []
    for t in range(1, A.shape[0] - 1):
        cur, prev, nxt = np.asarray(A[t]), np.asarray(A[t - 1]), np.asarray(A[t + 1])
        ok = wet & np.isfinite(cur) & np.isfinite(prev) & np.isfinite(nxt)
        if not ok.any():
            continue
        F = patch_features(
            cur,
            prev,
            [(np.asarray(o[t]), np.asarray(o[t - 1])) for o in other_arrs],
            (np.asarray(U[t]), np.asarray(U[t - 1])) if U is not None else None,
            (np.asarray(D[t]), np.asarray(D[t - 1])) if D is not None else None,
            np.asarray(Uv[t]),
            np.asarray(Vv[t]),
            mean_c,
            cache.inv_cos,
            forcing=[(np.asarray(f[t]), np.asarray(f[t - 1])) for f in forc]
            if forcing
            else None,
        )
        cell = np.flatnonzero(ok)
        Fv = F.reshape(-1, F.shape[-1])[cell]
        parts.append(
            (
                Fv,
                (nxt - cur).ravel()[cell],
                cur.ravel()[cell],
                np.full(len(cell), t),
                cell,
                flat_band[cell],
            )
        )
    F, y, x, tt, cell, band = (np.concatenate(z) for z in zip(*parts))
    grad = np.abs(F[:, :8]).mean(1)
    return Samples(
        F.astype(np.float64),
        y.astype(np.float64),
        x.astype(np.float64),
        tt,
        cell,
        band,
        grad,
        np.abs(F[:, 8]),
    )


def subset(s: Samples, idx) -> Samples:
    return Samples(*(getattr(s, f)[idx] for f in Samples.__dataclass_fields__))


def area_sample(cache: Cache, s: Samples, n: int, rng) -> np.ndarray:
    w = cache.area.ravel()[s.cell].astype(np.float64)
    return rng.choice(len(w), size=min(n, len(w)), replace=True, p=w / w.sum())


def var_of(c: str) -> str:
    return "zos" if c == "zos" else c.rsplit("_", 1)[0]


def sigma_delta(cache: Cache, var: str) -> float:
    """Area-weighted std of the five-day change of one variable, pooled over all
    its levels and the training window (Keisler 2022, sec. 3.3.3).
    """
    sw = swy = swy2 = 0.0
    for c in cache.meta["channels"]:
        if var_of(c) != var:
            continue
        A = cache.arr("train", c)
        wet = cache.masks[cache.meta["levels"][c]]
        d = (np.asarray(A[1:]) - np.asarray(A[:-1]))[:, wet].astype(np.float64)
        w = np.broadcast_to(cache.area[wet], d.shape)
        ok = np.isfinite(d)
        sw += w[ok].sum()
        swy += (w[ok] * d[ok]).sum()
        swy2 += (w[ok] * d[ok] ** 2).sum()
    if sw == 0:
        raise ValueError(f"no finite training tendencies for {var}")
    mean = swy / sw
    return float(np.sqrt(swy2 / sw - mean**2))


def run_channel(
    cache: Cache,
    c: str,
    eps: float,
    sig: float,
    n_eval: int,
    n_train: int,
    rng,
    diagnostics: bool = True,
    forcing: bool = False,
) -> tuple[dict, dict]:
    t0 = time.time()
    A = cache.arr("train", c)
    vals = np.asarray(A)[:, cache.masks[cache.meta["levels"][c]]]
    mean_c = float(np.nanmean(vals))
    bins = ToleranceBins(eps * sig, sig)
    tr = build_samples(cache, "train", c, mean_c, forcing)
    if len(tr.y) > n_train:
        tr = subset(tr, rng.choice(len(tr.y), n_train, replace=False))
    lad = Ladder(seed=int(rng.integers(1 << 31))).fit(tr)
    va = build_samples(cache, "val", c, mean_c, forcing)
    lad.tune_on_val(subset(va, area_sample(cache, va, n_eval // 2, rng)), bins)
    te = build_samples(cache, "test", c, mean_c, forcing)
    ev = subset(te, area_sample(cache, te, n_eval, rng))
    out = lad.predict(ev, bins)
    a_ev, b_ev = bins.tendency_interval(ev.x, ev.y)
    res = {
        "channel": c,
        "depth": cache.meta["depth"][c],
        "eps": eps,
        "sigma_delta": sig,
        "bin_width": bins.width,
        "n_train_rows": int(len(tr.y)),
        "n_eval_rows": int(len(ev.y)),
        "bits": {
            m: float(bins.bits(p, a_ev, b_ev).mean()) for m, (_, p) in out.items()
        },
        "rmse": {
            m: float(np.sqrt(np.mean((mu - ev.y) ** 2))) for m, (mu, _) in out.items()
        },
        "mix_weights": {k: float(v) for k, v in lad.mix_w.items()},
        "bandwidth_mult": {"analog": lad.c_analog, "lpc_analog": lad.c_lpc_analog},
        "n_params": lad.n_params(),
    }
    res["seconds"] = time.time() - t0
    if not diagnostics:
        return res, {}
    # ---- side quest, on every wet test cell for the cheap (non-kNN) rungs
    full = lad.predict(te, bins, with_knn=False)
    a_te, b_te = bins.tendency_interval(te.x, te.y)
    aw = cache.area.ravel()[te.cell]
    res["bits_full_field"] = {
        m: float(np.average(bins.bits(p, a_te, b_te), weights=aw))
        for m, (_, p) in full.items()
    }
    wet = cache.masks[cache.meta["levels"][c]]
    H, W = wet.shape
    bl: dict[str, list[float]] = {
        "truth": [],
        "ar1_forecast": [],
        "lpc_forecast": [],
        "residual_vs_persistence": [],
    }
    gain_map = np.zeros(H * W)
    cnt_map = np.zeros(H * W)
    g = bins.bits(full["persistence"][1], a_te, b_te) - bins.bits(
        full["lpc_ctx"][1], a_te, b_te
    )
    np.add.at(gain_map, te.cell, g)
    np.add.at(cnt_map, te.cell, 1)
    for t in np.unique(te.t):
        m = te.t == t
        cells = te.cell[m]
        okmask = np.zeros(H * W, bool)
        okmask[cells] = True
        okmask = okmask.reshape(H, W)
        n = int(m.sum())

        def field(vals):
            f = np.full(H * W, np.nan)
            f[cells] = vals
            return f.reshape(H, W)

        truth = te.x[m] + te.y[m]
        bl["truth"].append(
            blosc_bits_per_value(quantized_field(field(truth), okmask, bins), n)
        )
        for name, model in (("ar1_forecast", "ar1"), ("lpc_forecast", "lpc")):
            fc = te.x[m] + full[model][0][m]
            bl[name].append(
                blosc_bits_per_value(quantized_field(field(fc), okmask, bins), n)
            )
        r = np.zeros(H * W, np.int32)
        r[cells] = bins.index(truth) - bins.index(te.x[m])
        bl["residual_vs_persistence"].append(blosc_bits_per_value(r, n))
    res["blosc_bits_per_value"] = {k: float(np.mean(v)) for k, v in bl.items()}
    # Spatial correlation of lpc errors: a pointwise model codes each cell
    # independently, so correlated errors are bits a joint (field) coder would
    # save. Gaussian estimate of that gap: -0.5 log2(1 - rho^2) per value.
    rr = np.full((te.t.max() + 1, H * W), np.nan)
    rr[te.t, te.cell] = te.y - full["lpc"][0]
    rr = rr.reshape(-1, H, W)
    e = np.roll(rr, -1, axis=2)
    ok = np.isfinite(rr) & np.isfinite(e)
    rho = float(np.corrcoef(rr[ok], e[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    res["lpc_error_zonal_corr"] = rho
    res["joint_coding_gap_bits_est"] = float(-0.5 * np.log2(max(1 - rho**2, 1e-12)))
    res["seconds"] = time.time() - t0
    maps = {
        f"{c}_gain": np.where(
            cnt_map > 0, gain_map / np.maximum(cnt_map, 1), np.nan
        ).reshape(H, W)
    }
    return res, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--eps", type=float, required=True, help="bin width / sigma_delta")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument(
        "--forcing", action="store_true", help="add tauuo, tauvo, hfds features"
    )
    ap.add_argument("--n-eval", type=int, default=8000)
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--channels", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = Cache(a.cache)
    rng = np.random.default_rng(a.seed)
    chans = a.channels.split(",") if a.channels else cache.meta["channels"]
    a.out.mkdir(parents=True, exist_ok=True)
    sig = {v: sigma_delta(cache, v) for v in dict.fromkeys(map(var_of, chans))}
    results, maps = [], {}
    for c in chans:
        res, mp = run_channel(
            cache,
            c,
            a.eps,
            sig[var_of(c)],
            a.n_eval,
            a.n_train,
            rng,
            diagnostics=not a.no_diagnostics,
            forcing=a.forcing,
        )
        results.append(res)
        if any(c == m for m in MAP_CHANNELS):
            maps.update(mp)
        b = res["bits"]
        print(
            f"[run] {c:10s} persist {b['persistence']:.2f}  lpc_ctx {b['lpc_ctx']:.2f}  lpc_analog {b['lpc_analog']:.2f}  mix {b['mix']:.2f} bits  ({res['seconds']:.0f}s)",
            flush=True,
        )
    meta = {k: v for k, v in cache.meta.items() if k != "splits"}
    meta["sigma_delta"] = sig
    meta["forcing_features"] = list(FORCING_VARS) if a.forcing else []
    meta["dz"] = {c: float(cache.dz[cache.meta["levels"][c]]) for c in chans}
    meta["wet_area"] = {
        c: float(cache.area[cache.masks[cache.meta["levels"][c]]].sum()) for c in chans
    }
    meta["split_ranges"] = {
        s: [v[0], v[-1], len(v)] for s, v in cache.meta["splits"].items()
    }
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "args": vars(a) | {"cache": str(a.cache), "out": str(a.out)},
                "channels": results,
            },
            indent=1,
        )
    )
    np.savez_compressed(a.out / "maps.npz", lat=cache.lat, **maps)


if __name__ == "__main__":
    main()
