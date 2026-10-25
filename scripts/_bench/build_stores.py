"""Storage-precision bench on real face-1 data, against a float32 reference.

  errors: per format/variable/level, the state error and the error in the 1-hour change (x[t+1] - x[t]),
          over WINDOWS of consecutive hours across the training year. Writes only errors.csv.
  stores: 36 tiles x STORE_HOURS per window, written once per format x shuffle (zstd-3), for size and
          for scripts/_bench/throughput2.py (read speed and CPU decode time).

  .venv/bin/python scripts/_bench/build_stores.py --phase errors|stores|both [--root DIR] [--workers N]
"""
import argparse, functools, os, shutil, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numcodecs, numpy as np, torch, xarray as xr, zarr

numcodecs.blosc.use_threads = False
warnings.filterwarnings("ignore", message="All-NaN slice")
SRC = "/orcd/data/abodner/003/LLC4320/LLC4320"
STATS = "/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_{}.zarr"
FACE, TILE, NSIDE, LEVELS, FULL_HOURS = 1, 720, 6, 51, 10311
VARS = ["U", "V", "Theta", "Salt", "Eta"]  # the face cache's channel order
ORIGINS = [(j * TILE, i * TILE) for j in range(NSIDE) for i in range(NSIDE)]
FORMATS = ["fp16", "bf16", "fp16_k8", "fp16_k7", "fp16_k6", "fp16_k5", "fp8e4m3", "fp8e5m2"]  # fp32 is the reference
SHUFFLES = {"none": numcodecs.Blosc.NOSHUFFLE, "byte": numcodecs.Blosc.SHUFFLE, "bit": numcodecs.Blosc.BITSHUFFLE}
# Shuffle changes size and speed, never values: fp8 is one byte (byte shuffle is a no-op), fp32 is the size reference.
STORES = ([("fp32", "byte")] + [(f, s) for f in ("fp16", "bf16") for s in SHUFFLES]
          + [(f"fp16_k{k}", s) for k in (8, 7, 6, 5) for s in ("byte", "bit")]
          + [(f, s) for f in ("fp8e4m3", "fp8e5m2") for s in ("none", "bit")])
TORCH = {"bf16": (torch.bfloat16, torch.int16, np.uint16), "fp8e4m3": (torch.float8_e4m3fn, torch.uint8, np.uint8),
         "fp8e5m2": (torch.float8_e5m2, torch.uint8, np.uint8)}
FP8_MAX = {"fp8e4m3": 448.0, "fp8e5m2": 57344.0}  # saturate rather than overflow to NaN/inf


@functools.cache
def source():
    return zarr.open_group(SRC, mode="r")


@functools.cache
def stats():
    mean, std = (xr.open_zarr(STATS.format(k)) for k in ("means", "stds"))
    pick = lambda ds, v: np.float32(ds[v].values) if v == "Eta" else np.array(
        [ds[f"{v}_lev_{k}"].values.item() for k in range(LEVELS)], dtype=np.float32)
    return {v: (pick(mean, v), pick(std, v)) for v in VARS}


def bitround(h: np.ndarray, keep: int) -> np.ndarray:
    """Round-to-nearest-even onto `keep` of float16's 10 mantissa bits; NaN/inf untouched."""
    drop = 10 - keep
    u = h.view(np.uint16).astype(np.uint32)
    u = (u + (1 << (drop - 1)) - 1 + ((u >> drop) & 1)) & ~np.uint32((1 << drop) - 1)
    return np.where(np.isfinite(h), u.astype(np.uint16).view(np.float16), h)


def quantize(x: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """(stored array, its float32 decode) for z-scored float32 `x`; bf16/fp8 are stored as their raw bits."""
    if fmt == "fp32":
        return x, x
    if fmt.startswith("fp16"):
        h = x.astype(np.float16)
        if "_k" in fmt:
            h = bitround(h, int(fmt.split("_k")[1]))
        return h, h.astype(np.float32)
    dtype, bits, store = TORCH[fmt]
    t = torch.from_numpy(np.ascontiguousarray(x))
    q = (t.clamp(-FP8_MAX[fmt], FP8_MAX[fmt]) if fmt in FP8_MAX else t).to(dtype)  # clamp keeps NaN
    return q.view(bits).numpy().view(store), q.float().numpy()


def eta_path(root, start):
    return f"{root}/eta_face{FACE}_{start}.npy"


def cache_eta(root, windows, hours):
    """Face-1 Eta per window, z-scored: its raw chunk is the whole globe, so read each hour once, not per tile."""
    mu, sd = stats()["Eta"]
    for start in windows:
        if not os.path.exists(eta_path(root, start)):
            t0 = time.perf_counter()
            eta = (source()["Eta"][start:start + hours, FACE] - mu) / sd
            np.save(eta_path(root, start), eta.astype(np.float32))
            print(f"[eta] window {start}: {hours} hours cached in {time.perf_counter() - t0:.0f}s", flush=True)


def read(v, start, hours, origin, root):
    """(hours, L, 720, 720) z-scored float32, land NaN."""
    j0, i0 = origin
    if v == "Eta":
        return np.array(np.load(eta_path(root, start), mmap_mode="r")[:hours, None, j0:j0 + TILE, i0:i0 + TILE])
    mu, sd = stats()[v]
    blk = source()[v][start:start + hours, :, FACE, j0:j0 + TILE, i0:i0 + TILE]
    return ((blk - mu[:, None, None]) / sd[:, None, None]).astype(np.float32)


def error_job(job):
    """Sums for one (window, tile, variable) over the window's hours, 17 levels at a time to bound memory."""
    start, hours, origin, v, root = job
    x_all = read(v, start, hours, origin, root)
    out = {k: [] for k in ("n", "nt", "tend_ss")} | {f: {k: [] for k in ("ss", "max", "tend_err_ss", "lost")} for f in FORMATS}
    axes = (0, 2, 3)
    for l0 in range(0, x_all.shape[1], 17):
        x = x_all[:, l0:l0 + 17]
        dx = x[1:] - x[:-1]
        out["n"].append(np.isfinite(x).sum(axis=axes))
        out["nt"].append(np.isfinite(dx).sum(axis=axes))
        out["tend_ss"].append(np.nansum(dx * dx, axis=axes, dtype=np.float64))
        for f in FORMATS:
            q = quantize(x, f)[1]
            err = q - x
            dd = (q[1:] - q[:-1]) - dx
            out[f]["ss"].append(np.nansum(err * err, axis=axes, dtype=np.float64))
            out[f]["max"].append(np.nanmax(np.abs(err), axis=axes))
            out[f]["tend_err_ss"].append(np.nansum(dd * dd, axis=axes, dtype=np.float64))
            out[f]["lost"].append((np.isfinite(x) & ~np.isfinite(q)).sum(axis=axes))
    cat = lambda d: {k: np.concatenate(val) if isinstance(val, list) else cat(val) for k, val in d.items()}
    return v, cat(out)


def phase_errors(root, windows, hours, workers):
    cache_eta(root, windows, hours)
    jobs = [(s, hours, o, v, root) for s in windows for o in ORIGINS for v in VARS]
    acc = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for n, (v, out) in enumerate(pool.map(error_job, jobs), 1):
            if v not in acc:
                acc[v] = out
            else:
                a = acc[v]
                for k in ("n", "nt", "tend_ss"):
                    a[k] = a[k] + out[k]
                for f in FORMATS:
                    for k in ("ss", "tend_err_ss", "lost"):
                        a[f][k] = a[f][k] + out[f][k]
                    a[f]["max"] = np.fmax(a[f]["max"], out[f]["max"])
            if n % 40 == 0 or n == len(jobs):
                print(f"[errors] {n}/{len(jobs)} jobs ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for v in VARS:
            a = acc[v]
            tend = np.sqrt(a["tend_ss"] / a["nt"])
            for f in FORMATS:
                rms, tend_err = np.sqrt(a[f]["ss"] / a["n"]), np.sqrt(a[f]["tend_err_ss"] / a["nt"])
                rows += [(f, v, k, rms[k], a[f]["max"][k], tend[k], tend_err[k], tend_err[k] / tend[k], int(a[f]["lost"][k]))
                         for k in range(len(rms)) if a["n"][k] > 0]
    with open(f"{root}/errors.csv", "w") as fh:
        fh.write("format,var,level,rms_err,max_err,rms_tendency,rms_tendency_err,tendency_err_ratio,ocean_lost\n")
        fh.writelines(",".join(map(str, r)) + "\n" for r in rows)
    print(f"\n[errors] {root}/errors.csv, z units. ratio = rms(quantized 1-h change - true) / rms(true 1-h change), "
          f"median / worst level; {len(windows)} windows x {hours - 1} changes x {len(ORIGINS)} tiles")
    print(f"{'format':9s}" + "".join(f"{v + ' rms':>10s}{v + ' ratio':>13s}" for v in VARS))
    for f in FORMATS:
        line = f"{f:9s}"
        for v in VARS:
            r = [row for row in rows if row[0] == f and row[1] == v]
            ratios = sorted(row[7] for row in r)
            line += f"{np.median([row[3] for row in r]):10.1e}{np.median(ratios):8.3f}/{ratios[-1]:<5.2f}"
        lost = sum(row[8] for row in rows if row[0] == f)
        print(line + (f"  {lost} OCEAN VALUES LOST" if lost else ""), flush=True)


def store_job(job):
    """Write one (hour, tile) into every store; returns per-store write seconds."""
    root, slot, start, h, origin = job
    x = np.concatenate([read(v, start + h, 1, origin, root)[0] if v != "Eta" else
                        np.load(eta_path(root, start), mmap_mode="r")[h, None, origin[0]:origin[0] + TILE,
                                                                       origin[1]:origin[1] + TILE] for v in VARS])
    x = np.ascontiguousarray(x, dtype=np.float32)
    j0, i0 = origin
    seconds = {}
    for fmt, shuffle in STORES:
        stored = quantize(x, fmt)[0]
        arr = zarr.open_group(f"{root}/{fmt}__{shuffle}.zarr", mode="r+")["prognostic"]
        t0 = time.perf_counter()
        arr[slot, :, j0:j0 + TILE, i0:i0 + TILE] = stored
        seconds[(fmt, shuffle)] = time.perf_counter() - t0
    return seconds


def phase_stores(root, windows, hours, store_hours, workers):
    cache_eta(root, windows, hours)
    slots = [(s, h) for s in windows for h in range(store_hours)]
    for fmt, shuffle in STORES:
        path = f"{root}/{fmt}__{shuffle}.zarr"
        shutil.rmtree(path, ignore_errors=True)
        grp = zarr.open_group(path, mode="w")
        grp.attrs.update({"format": fmt, "shuffle": shuffle, "hours": [s + h for s, h in slots]})
        grp.create_dataset("prognostic", shape=(len(slots), 205, NSIDE * TILE, NSIDE * TILE),
                           chunks=(1, 205, TILE, TILE), dtype=quantize(np.zeros(1, np.float32), fmt)[0].dtype,
                           fill_value=None, compressor=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=SHUFFLES[shuffle]))
    jobs = [(root, slot, s, h, o) for slot, (s, h) in enumerate(slots) for o in ORIGINS]
    seconds = {s: 0.0 for s in STORES}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for n, out in enumerate(pool.map(store_job, jobs), 1):
            for s, dt in out.items():
                seconds[s] += dt
            if n % 36 == 0:
                print(f"[stores] {n}/{len(jobs)} tile-hours ({time.perf_counter() - t0:.0f}s)", flush=True)
    full = FULL_HOURS / len(slots) * NSIDE**2 / len(ORIGINS)  # the whole-face cache: 10311 hours x 36 tiles
    ref = None
    print(f"\n{'store':20s} {'sample GB':>10s} {'whole-face TB':>14s} {'vs fp32':>8s} {'write s/chunk':>14s}")
    for fmt, shuffle in STORES:
        path = f"{root}/{fmt}__{shuffle}.zarr"
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        ref = ref or size
        print(f"{fmt + '__' + shuffle:20s} {size / 1e9:10.2f} {size * full / 1e12:14.1f} {size / ref:8.3f} "
              f"{seconds[(fmt, shuffle)] / len(jobs):14.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("errors", "stores", "both"), default="both")
    p.add_argument("--root", default="/orcd/data/abodner/002/cody/LLC_patch/_precision_bench")
    p.add_argument("--windows", default="500,2700,4900,7100", help="first raw hour of each window (training year)")
    p.add_argument("--hours", type=int, default=13, help="consecutive hours per window for errors (hours-1 changes)")
    p.add_argument("--store-hours", type=int, default=2, help="hours per window written to the stores")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--tiles", type=int, default=NSIDE**2, help="first N of the 36 tiles (a quick test)")
    a = p.parse_args()
    global ORIGINS
    ORIGINS = ORIGINS[:a.tiles]  # workers fork after this, so they see it
    windows = [int(w) for w in a.windows.split(",")]
    os.makedirs(a.root, exist_ok=True)
    print(f"face {FACE}, {len(ORIGINS)} tiles, windows {windows}, root {a.root}", flush=True)
    if a.phase in ("errors", "both"):
        phase_errors(a.root, windows, a.hours, a.workers)
    if a.phase in ("stores", "both"):
        phase_stores(a.root, windows, a.hours, a.store_hours, a.workers)


if __name__ == "__main__":
    main()
