"""Phase C: sustained multi-process read throughput, page cache evicted.

A "sample" is one tile-timestep: the contiguous (205,720,720) float16 buffer the
model consumes. A real training sample is hist+1+steps of these, so scale
accordingly. Each chunk is read exactly ONCE per run and the page cache is
dropped first, so this measures batches/second under load rather than the
latency of a warm single read.
"""
import itertools, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np, zarr

ROOT="/orcd/data/abodner/002/cody/LLC_patch/_layout_bench"
TILE=720; NSIDE=6; NT=4; VARS=["Theta","Salt","U","V"]
SAMPLE_BYTES=205*TILE*TILE*2
JOBS=[(t,j*TILE,i*TILE) for t in range(NT) for j in range(NSIDE) for i in range(NSIDE)]
random.Random(0).shuffle(JOBS)

_G=None; _LAYOUT=None
def _init(path,layout):
    global _G,_LAYOUT
    _G=zarr.open_group(path,mode="r"); _LAYOUT=layout
def _read(job):
    t,j0,i0=job
    if _LAYOUT=="packed":
        a=np.asarray(_G["prognostic"][t,:,j0:j0+TILE,i0:i0+TILE])
    else:
        parts=[np.asarray(_G[v][t,:,j0:j0+TILE,i0:i0+TILE]) for v in VARS]
        parts.append(np.asarray(_G["Eta"][t,j0:j0+TILE,i0:i0+TILE])[None])
        a=np.concatenate(parts,axis=0)
    assert a.shape==(205,TILE,TILE) and a.flags["C_CONTIGUOUS"]
    return a.nbytes

def evict(path):
    files=[os.path.join(d,f) for d,_,fs in os.walk(path) for f in fs]
    def drop(p):
        try:
            fd=os.open(p,os.O_RDONLY)
            try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally: os.close(fd)
        except OSError: pass
    with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(drop,files))

def run(path,layout,workers):
    with Pool(workers,initializer=_init,initargs=(path,layout)) as pool:
        t0=time.perf_counter()
        total=sum(pool.imap_unordered(_read,JOBS,chunksize=1))
        dt=time.perf_counter()-t0
    return dt,total

if __name__=="__main__":
    stores=sorted(d for d in os.listdir(ROOT) if d.endswith(".zarr"))
    WS=[8,16,32,64]; REP=2
    print(f"{len(JOBS)} samples/run, {SAMPLE_BYTES/1e6:.1f} MB logical each "
          f"({len(JOBS)*SAMPLE_BYTES/1e9:.1f} GB logical)\n",flush=True)
    print(f"{'store':38s} {'W':>3s} {'cold s':>7s} {'samp/s':>7s} {'GB/s':>6s}",flush=True)
    results={}
    for st in stores:
        path=f"{ROOT}/{st}"; layout="packed" if st.startswith("packed") else "compact"
        for w in WS:
            best=None
            for _ in range(REP):
                evict(path)
                dt,tot=run(path,layout,w)
                best=dt if best is None else min(best,dt)
            sps=len(JOBS)/best; gbs=len(JOBS)*SAMPLE_BYTES/best/1e9
            results[(st,w)]=(best,sps,gbs)
            print(f"{st[:-5]:38s} {w:3d} {best:7.2f} {sps:7.2f} {gbs:6.2f}",flush=True)

    # Warm cache at W=32: isolates decompress+assemble with no I/O, i.e. exactly
    # what the earlier flawed microbenchmark was measuring.
    print(f"\n--- WARM cache (no eviction), W=32: decompress+assemble only ---",flush=True)
    print(f"{'store':38s} {'warm s':>7s} {'samp/s':>7s} {'vs cold':>8s}",flush=True)
    for st in stores:
        path=f"{ROOT}/{st}"; layout="packed" if st.startswith("packed") else "compact"
        run(path,layout,32)                     # prime the cache
        dt,_=run(path,layout,32)
        cold=results[(st,32)][0]
        print(f"{st[:-5]:38s} {dt:7.2f} {len(JOBS)/dt:7.2f} {cold/dt:7.2f}x",flush=True)
