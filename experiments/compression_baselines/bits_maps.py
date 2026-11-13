# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Per-cell maps of bits saved over persistence: the ladder beside an external model.

    python bits_maps.py CACHE LADDER_RESULTS_DIR PRED_DIR OUT [--name samudra2] [--full]

The ladder's maps come from its run (LADDER_RESULTS_DIR/maps.npz): bits per value
saved by `lpc_ctx` over persistence, averaged over test rows at each cell. This
script computes the same quantity for the external forecasts in PRED_DIR (from
one_step.py), scored exactly as score_external.py scores them: a Laplace around
the forecast with one spread per channel from validation errors, the ladder's
tolerance bins and floor, and persistence with the spread fit on training
tendencies. It is done for the channels the ladder's maps cover.

Positive (red) means the model puts more probability on the true bin than
persistence does at that cell; one bit is twice the probability. The ladder is
scored on every test step, the external model on the steps it forecast (every
other one for Samudra 2): the same period, different samples.

Writes OUT/bits-maps.png, OUT/bits-maps.csv (area-weighted means) and
OUT/bits_maps.npz.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from score_external import FLOOR, REF_MULT, laplace_interval, var_of


def cell_gain(
    cache: Path,
    meta: dict,
    c: str,
    pred,
    pmeta: dict,
    w: float,
    sig: float,
    b_model: float,
    b_pers: float,
) -> np.ndarray:
    A = np.load(cache / "test" / f"{c}.npy", mmap_mode="r")
    wet = np.load(cache / "static.npz")["masks"][meta["levels"][c]]
    stamps = meta["splits"]["test"]
    s = np.zeros(wet.shape)
    n = np.zeros(wet.shape)
    for k, t in enumerate(pmeta["target_times"]):
        j = stamps.index(t)
        x, y, f = (np.asarray(v, np.float64) for v in (A[j - 1], A[j], pred[k]))
        ok = wet & np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
        x, y, f = x[ok], y[ok], f[ok]
        idx = np.floor(y / w)
        lo, hi = idx * w - x, (idx + 1) * w - x
        q = laplace_interval(lo, hi, 0.0, REF_MULT * sig)

        def bits(p, q=q):
            return -np.log2((1 - FLOOR) * p + FLOOR * q)

        g = bits(laplace_interval(lo, hi, 0.0, b_pers)) - bits(
            laplace_interval(lo, hi, f - x, b_model)
        )
        s[ok] += g
        n[ok] += 1
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("ladder", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", default="samudra2")
    ap.add_argument("--full", action="store_true", help="200 dpi, unquantized")
    a = ap.parse_args()
    meta = json.loads((a.cache / "meta.json").read_text())
    L = json.loads((a.ladder / "results.json").read_text())
    lad = {c["channel"]: c for c in L["channels"]}
    lmaps = np.load(a.ladder / "maps.npz")
    chans = [k[: -len("_gain")] for k in lmaps.files if k.endswith("_gain")]
    P = {s: np.load(a.pred / s / "pred.npy", mmap_mode="r") for s in ("val", "test")}
    M = {s: json.loads((a.pred / s / "meta.json").read_text()) for s in ("val", "test")}
    for s in M:
        if not M[s]["complete"]:
            raise SystemExit(f"{s} forecasts are incomplete (smoke run?)")
    names = M["test"]["channels"]
    st = np.load(a.cache / "static.npz")
    area, lat = st["area"], st["lat"][:, 0]
    out: dict[str, np.ndarray] = {}
    rows = []
    for c in chans:
        i = names.index(c)
        w, sig = lad[c]["bin_width"], L["meta"]["sigma_delta"][var_of(c)]
        wet = st["masks"][meta["levels"][c]]
        # spreads exactly as score_external.py fits them
        V = np.load(a.cache / "val" / f"{c}.npy", mmap_mode="r")
        vst = meta["splits"]["val"]
        errs = []
        for k, t in enumerate(M["val"]["target_times"]):
            j = vst.index(t)
            if j == 0:
                continue
            y, f = np.asarray(V[j], np.float64), np.asarray(P["val"][k, i], np.float64)
            ok = (
                wet
                & np.isfinite(y)
                & np.isfinite(f)
                & np.isfinite(np.asarray(V[j - 1]))
            )
            errs.append(np.abs(y - f)[ok])
        b_model = float(np.mean(np.concatenate(errs)))
        T = np.load(a.cache / "train" / f"{c}.npy", mmap_mode="r")
        b_pers = float(
            np.nanmean(np.abs((np.asarray(T[1:]) - np.asarray(T[:-1]))[:, wet]))
        )
        g = cell_gain(
            a.cache, meta, c, P["test"][:, i], M["test"], w, sig, b_model, b_pers
        )
        out[f"{c}/{a.name}"] = g.astype(np.float32)
        out[f"{c}/ladder"] = lmaps[f"{c}_gain"].astype(np.float32)
        for m in ("ladder", a.name):
            v = out[f"{c}/{m}"]
            ok = np.isfinite(v)
            rows.append(
                {
                    "channel": c,
                    "depth_m": lad[c]["depth"],
                    "model": m,
                    "area_mean_bits_saved": float(np.average(v[ok], weights=area[ok])),
                    "share_of_area_worse_than_persistence": float(
                        np.sum(area[ok & (np.nan_to_num(v) < 0)]) / np.sum(area[ok])
                    ),
                }
            )
        print(f"[bits_maps] {c} done", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "bits_maps.npz", lat=lat, **out)
    with (a.out / "bits-maps.csv").open("w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onestep_figures
    from onestep_figures import save

    onestep_figures.FULL = a.full
    fig, axes = plt.subplots(
        len(chans), 2, figsize=(13, 3.1 * len(chans)), layout="constrained"
    )
    lab = {
        "ladder": "Ladder (lpc_ctx)",
        a.name: "Samudra 2" if a.name == "samudra2" else a.name,
    }
    for r, c in enumerate(chans):
        both = np.concatenate(
            [out[f"{c}/ladder"].ravel(), out[f"{c}/{a.name}"].ravel()]
        )
        lim = float(np.nanpercentile(np.abs(both), 98))
        for col, m in enumerate(("ladder", a.name)):
            ax = axes[r, col]
            v = out[f"{c}/{m}"]
            ok = np.isfinite(v)
            im = ax.imshow(
                v,
                origin="lower",
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_facecolor("#d9d9d9")
            ax.set_xticks([])
            ax.set_yticks([])
            mean = np.average(v[ok], weights=area[ok])
            ax.set_title(
                f"{lab[m]} · {c} ({lad[c]['depth']:g} m) · area mean {mean:+.2f} bits",
                fontsize=10,
            )
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, label="bits saved / value")
    fig.suptitle(
        "Bits per value saved over persistence, one step from the true state, 1° test window 2014–2022\n"
        "red = more probability on the true bin than persistence (1 bit = 2×) · blue = worse than persistence · grey = land",
        fontsize=12,
    )
    save(fig, a.out / "bits-maps.png", 80)


if __name__ == "__main__":
    main()
