#!/usr/bin/env python
"""Read-only sweep of coarsen_llc.py output over every finished hour, without the raw store.

Every hour marked done (samudra and native markers separately):
- every chunk file exists, and `repaired[t]` is 0 or 1 in each store;
- 1-degree fields, and the 1/4-degree and native 2D fields: NaNs equal `wet_area == 0`,
  values finite and inside LIMITS; a global wet-area mean and a checksum are kept.
Every --stride-th done hour, also:
- 1/4-degree and native 3D fields: the same NaN and range checks;
- 1 degree equals the wet-area mean of its 4x4 1/4-degree cells, for every variable;
- every rechunk tile file exists.
Afterwards: no field is bit-identical between two hours, and no global mean jumps
between consecutive hours by more than --jump-z robust deviations.
Safe while fills run: an hour is read only after its marker exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from coarsen_llc import (
    COARSE_ROOT, LIMITS, NATIVE, NATIVE_NAME, ONE, QUARTER, RECHUNK_NAME, RECHUNK_ROOT, SAMUDRA,
    SPLIT, STORE_NAMES, TILE, TILES, VARS_2D, VARS_3D,
)
from ocean_emulators.llc_coarse import box_sum, ratio

logger = logging.getLogger(__name__)


def markers(root: Path, product: str) -> list[int]:
    path = root / ("_progress" if product == SAMUDRA else "_progress_native")
    return sorted(int(n) for n in os.listdir(path))


def field_problems(name: str, x: np.ndarray, land: np.ndarray) -> list[str]:
    """NaN pattern against land, and range against LIMITS."""
    var = name.split("/")[-1]
    out = []
    if not np.array_equal(np.isnan(x), np.broadcast_to(land, x.shape)):
        out.append(f"{name}: {int((np.isnan(x) != land).sum())} cells with NaN-ness != land")
    wet = x[~np.broadcast_to(land, x.shape)]
    lo, hi = LIMITS[var]
    if wet.size and (np.isinf(wet).any() or wet.min() < lo or wet.max() > hi):
        out.append(f"{name}: range {wet.min():.4g}..{wet.max():.4g} outside {lo}..{hi}")
    return out


def wet_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.nansum(x * w, dtype=np.float64) / w.sum())


class Sweep:
    def __init__(self, args):
        self.root, self.rechunk_root, self.stride = args.coarse_root, args.rechunk_root, args.stride
        self.q = zarr.open_group(str(self.root / STORE_NAMES[QUARTER]), mode="r")
        self.one = zarr.open_group(str(self.root / STORE_NAMES[ONE]), mode="r")
        self.nat = zarr.open_group(str(self.root / NATIVE_NAME), mode="r")
        self.rechunk = zarr.open_group(str(self.rechunk_root / RECHUNK_NAME), mode="r")
        self.w = {name: g["wet_area"][:].astype(np.float64) for name, g in
                  (("q", self.q), ("one", self.one), ("nat", self.nat))}
        self.land = {k: w == 0 for k, w in self.w.items()}
        # All-land tiles are never written (they equal the fill value).
        wet = self.rechunk["hFacC_2d"][:] > 0
        self.ocean_tiles = sum(bool(wet[f, j:j + TILE, i:i + TILE].any()) for f, j, i in TILES)

    @staticmethod
    def chunks_missing(group: zarr.Group, store: Path, t: int) -> list[str]:
        names = {var: f"{t}" + ".0" * (group[var].ndim - 1) for var in VARS_3D + VARS_2D}
        return [f"{store.name}/{var}/{n} missing" for var, n in names.items() if not (store / var / n).exists()]

    def samudra(self, t: int) -> dict:
        problems = self.chunks_missing(self.q, self.root / STORE_NAMES[QUARTER], t)
        problems += self.chunks_missing(self.one, self.root / STORE_NAMES[ONE], t)
        repaired = dict(zip(VARS_3D + VARS_2D, self.one["repaired"][t].tolist()))
        # -1 is legal here: hours filled before `repaired` existed, rewritten only if a later pass repaired them.
        rows = {label: g["repaired"][t] for label, g in (("0p25", self.q), ("1deg", self.one), ("rechunk", self.rechunk))}
        legacy = all((r == -1).all() for r in rows.values())
        for label, r in rows.items():
            if not (legacy or np.isin(r, (0, 1)).all()):
                problems.append(f"{label}/repaired[{t}] = {r.tolist()}")
        means, sums, deep = {}, {}, t % self.stride == 0
        for var in VARS_3D + VARS_2D:
            is3 = var in VARS_3D
            one = self.one[var][t]
            land1 = self.land["one"] if is3 else self.land["one"][0]
            w1 = self.w["one"] if is3 else self.w["one"][0]
            problems += field_problems(f"1deg/{var}", one, land1)
            means[var] = wet_mean(one[0] if is3 else one, w1[0] if is3 else w1)
            sums[f"1deg/{var}"] = zlib.crc32(one.tobytes())
            if is3 and not deep:
                continue
            q = self.q[var][t]
            wq = self.w["q"] if is3 else self.w["q"][0]
            problems += field_problems(f"0p25/{var}", q, self.land["q"] if is3 else self.land["q"][0])
            sums[f"0p25/{var}"] = zlib.crc32(q.tobytes())
            # 1 degree must be the wet-area mean of its 4x4 quarter cells.
            want = ratio(box_sum(np.nan_to_num(q) * wq, SPLIT), box_sum(wq, SPLIT))
            ok = np.isnan(want) == np.isnan(one)
            scale = np.nanmax(np.abs(want)) if np.isfinite(want).any() else 1.0
            ok &= ~(np.abs(want - one) > 1e-5 * scale + 1e-5 * np.abs(want))
            if not ok.all():
                problems.append(f"1deg/{var} differs from the 4x4 mean of 0p25 at {int((~ok).sum())} cells")
        if deep:
            for var in VARS_2D:
                tdir = self.rechunk_root / RECHUNK_NAME / var / str(t)
                n = sum(len(files) for _, _, files in os.walk(tdir))
                if n != self.ocean_tiles:
                    problems.append(f"rechunk/{var}/{t}: {n} tiles, expected {self.ocean_tiles} ocean tiles")
        return {"t": t, "product": SAMUDRA, "deep": deep, "means": means, "sums": sums, "problems": problems,
                "repaired": repaired, "legacy": legacy}

    def native(self, t: int) -> dict:
        problems = self.chunks_missing(self.nat, self.root / NATIVE_NAME, t)
        r = self.nat["repaired"][t]
        if not np.isin(r, (0, 1)).all():
            problems.append(f"native/repaired[{t}] = {r.tolist()}")
        means, sums, deep = {}, {}, t % self.stride == 0
        for var in (VARS_3D + VARS_2D) if deep else VARS_2D:
            is3 = var in VARS_3D
            x = self.nat[var][t]
            land = self.land["nat"] if is3 else self.land["nat"][0]
            problems += field_problems(f"native/{var}", x, land)
            means[var] = wet_mean(x[0] if is3 else x, self.w["nat"][0])
            sums[f"native/{var}"] = zlib.crc32(x.tobytes())
        return {"t": t, "product": NATIVE, "deep": deep, "means": means, "sums": sums, "problems": problems,
                "repaired": dict(zip(VARS_3D + VARS_2D, r.tolist()))}


def jumps(records: list[dict], z: float) -> list[str]:
    """Hour-to-hour changes of each global mean, flagged against a robust spread."""
    out = []
    by_t = {r["t"]: r["means"] for r in records}
    for var in VARS_2D + VARS_3D:
        pairs = [(t, by_t[t + 1][var] - by_t[t][var]) for t in sorted(by_t)
                 if t + 1 in by_t and var in by_t[t] and var in by_t[t + 1]]
        if len(pairs) < 50:
            continue
        d = np.array([p[1] for p in pairs])
        mad = np.median(np.abs(d - np.median(d))) * 1.4826 or 1e-30
        for t, v in pairs:
            if abs(v - np.median(d)) > z * mad:
                out.append(f"{var}: global mean jumps {v:.3g} from t={t} to {t + 1} ({abs(v) / mad:.0f} sigma)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coarse-root", type=Path, default=COARSE_ROOT)
    p.add_argument("--rechunk-root", type=Path, default=RECHUNK_ROOT)
    p.add_argument("--stride", type=int, default=24, help="Full 3D checks on hours with t %% stride == 0")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--jump-z", type=float, default=12.0)
    p.add_argument("--out", type=Path, required=True, help="JSON lines, one record per hour and product")
    args = p.parse_args()
    sweep = Sweep(args)
    jobs = [(sweep.samudra, t) for t in markers(args.coarse_root, SAMUDRA)]
    jobs += [(sweep.native, t) for t in markers(args.coarse_root, NATIVE)]
    logger.info("%d samudra and %d native hours to sweep", *(sum(f == g for f, _ in jobs)
                                                              for g in (sweep.samudra, sweep.native)))
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.threads) as pool, args.out.open("w") as fh:
        for n, rec in enumerate(pool.map(lambda job: job[0](job[1]), jobs), 1):
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            for problem in rec["problems"]:
                logger.error("t=%d %s", rec["t"], problem)
            if n % 500 == 0:
                logger.info("%d/%d swept", n, len(jobs))
    # A repaired field may be a whole-field copy of a nearby hour, so it is exempt.
    late, seen = [], {}
    for r in records:
        for name, crc in r["sums"].items():
            if r["repaired"][name.split("/")[1]]:
                continue
            if (name, crc) in seen:
                late.append(f"{name}: hours {seen[name, crc]} and {r['t']} are bit-identical")
            seen[name, crc] = r["t"]
    for product in (SAMUDRA, NATIVE):
        late += [f"{product} {x}" for x in jumps([r for r in records if r["product"] == product], args.jump_z)]
    for problem in late:
        logger.error(problem)
    problems = [f"t={r['t']} {x}" for r in records for x in r["problems"]] + late
    for product in (SAMUDRA, NATIVE):
        recs = [r for r in records if r["product"] == product]
        for var in VARS_3D + VARS_2D:
            m = [r["means"][var] for r in recs if var in r["means"]]
            if m:
                logger.info("%s %s global wet mean: %.5g .. %.5g over %d hours", product, var, min(m), max(m), len(m))
    logger.info("%d hours swept (%d deep), %d problem(s)", len(records), sum(r["deep"] for r in records), len(problems))
    logger.info("%d samudra hours predate `repaired` (all -1)", sum(r.get("legacy", False) for r in records))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
