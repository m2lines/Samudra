"""Read speed and CPU decode time of the stores scripts/_bench/build_stores.py writes.

  cold: tiles/s with the store's page cache evicted, W reader processes; reps are the OUTER loop so a noisy
        filesystem minute is spread across every store, and each store is evicted once measured.
  warm: the same read with the files cached (decompress only).
  decode: single-thread CPU time per chunk, compressed bytes already in RAM, plus the cast to float32.

  .venv/bin/python scripts/_bench/throughput2.py [--root DIR] [--workers 16,32] [--reps 3]
"""
import argparse, os, random, statistics, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numcodecs, numpy as np, torch, zarr

numcodecs.blosc.use_threads = False
TILE, NSIDE = 720, 6
TORCH = {"bf16": (torch.int16, torch.bfloat16), "fp8e4m3": (torch.uint8, torch.float8_e4m3fn),
         "fp8e5m2": (torch.uint8, torch.float8_e5m2)}

_ARRAY = None


def _init(path):
    global _ARRAY
    _ARRAY = zarr.open_group(path, mode="r")["prognostic"]


def _read(job):
    slot, j0, i0 = job
    return np.asarray(_ARRAY[slot, :, j0:j0 + TILE, i0:i0 + TILE]).nbytes


def to_float32(stored: np.ndarray, fmt: str) -> torch.Tensor:
    if fmt in TORCH:
        bits, dtype = TORCH[fmt]
        return torch.from_numpy(stored.view(np.int16) if bits is torch.int16 else stored).view(dtype).float()
    return torch.from_numpy(stored).float()


def evict(path):
    files = [os.path.join(d, f) for d, _, x in os.walk(path) for f in x]

    def drop(p):
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(drop, files))


def run(path, jobs, workers):
    with Pool(workers, initializer=_init, initargs=(path,)) as pool:
        t0 = time.perf_counter()
        sum(pool.imap_unordered(_read, jobs, chunksize=1))
        return time.perf_counter() - t0


def chunks(path):
    """(slot, j0, i0) of every chunk file the store holds."""
    names = [k for k in os.listdir(f"{path}/prognostic") if not k.startswith(".")]
    return [(int(t), int(j) * TILE, int(i) * TILE) for t, _, j, i in (k.split(".") for k in names)]


def decode(path, fmt, n=8):
    """Median single-thread ms per chunk: decompress, then decompress + cast to float32."""
    arr = zarr.open_group(path, mode="r")["prognostic"]
    blobs = [arr.store[f"prognostic/{t}.0.{j // TILE}.{i // TILE}"] for t, j, i in sorted(chunks(path))[:n]]
    dz, dc = [], []
    for blob in blobs:
        t0 = time.perf_counter()
        raw = np.frombuffer(arr.compressor.decode(blob), dtype=arr.dtype).reshape(arr.chunks)
        t1 = time.perf_counter()
        to_float32(raw, fmt)
        t2 = time.perf_counter()
        dz.append(t1 - t0)
        dc.append(t2 - t0)
    return 1e3 * statistics.median(dz), 1e3 * statistics.median(dc), statistics.mean(len(b) for b in blobs) / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/orcd/data/abodner/002/cody/LLC_patch/_precision_bench")
    p.add_argument("--workers", default="16,32")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    ws = [int(w) for w in a.workers.split(",")]
    stores = sorted(d for d in os.listdir(a.root) if d.endswith(".zarr"))
    meta = {s: zarr.open_group(f"{a.root}/{s}", mode="r") for s in stores}
    host = os.uname().nodename
    csv = f"{a.root}/results.csv"
    if not os.path.exists(csv):
        with open(csv, "w") as f:
            f.write("host,phase,store,workers,seconds,tiles\n")

    def record(phase, store, w, dt, tiles):
        with open(csv, "a") as f:
            f.write(f"{host},{phase},{store},{w},{dt:.4f},{tiles}\n")

    jobs = {s: chunks(f"{a.root}/{s}") for s in stores}
    for s in stores:
        random.Random(0).shuffle(jobs[s])
    cold = {}
    for rep in range(a.reps):
        order = stores[:]
        random.Random(100 + rep).shuffle(order)
        for st in order:
            path = f"{a.root}/{st}"
            for w in ws:
                evict(path)
                dt = run(path, jobs[st], w)
                cold.setdefault((st, w), []).append(dt)
                record("cold", st, w, dt, len(jobs[st]))
            evict(path)
        print(f"[cold] rep {rep + 1}/{a.reps} done", flush=True)
    warm, dec = {}, {}
    for st in stores:
        path = f"{a.root}/{st}"
        run(path, jobs[st], ws[-1])  # prime
        warm[st] = min(run(path, jobs[st], ws[-1]) for _ in range(2))
        record("warm", st, ws[-1], warm[st], len(jobs[st]))
        dec[st] = decode(path, meta[st].attrs["format"])
        evict(path)  # bound the page cache, or cgroup v2 charges it to the job
        print(f"[warm] {st[:-5]}", flush=True)
    print(f"\n{host}: tiles/s (one tile = 205 x 720 x 720), median over {a.reps} reps (min-max); decode is single-thread")
    print(f"{'store':20s} {'MB/chunk':>9s}" + "".join(f"{'cold W=' + str(w):>22s}" for w in ws)
          + f"{'warm W=' + str(ws[-1]):>13s}{'decomp ms':>11s}{'+cast ms':>10s}")
    for st in stores:
        row = f"{st[:-5]:20s} {dec[st][2]:9.1f}"
        for w in ws:
            v = cold[(st, w)]
            n = len(jobs[st])
            row += f"   {n / statistics.median(v):7.1f} ({n / max(v):5.1f}-{n / min(v):5.1f})"
        row += f"{len(jobs[st]) / warm[st]:13.1f}{dec[st][0]:11.0f}{dec[st][1]:10.0f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
