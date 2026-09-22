"""Phase C v2. Fixes two flaws in v1:

 * configs were measured in contiguous blocks, so shared-filesystem contention
   was confounded with the config under test. Now reps are the OUTER loop, so
   every rep sweeps all 12 configs and a noisy minute is spread across all of
   them. Report median across reps, with the full range.
 * the warm phase never evicted, so page cache accumulated across stores and
   cgroup v2 charged it to the job -- OOM at ~190 GB. Now each store is evicted
   once measured, bounding page cache to roughly one store.
"""
import os, random, statistics, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np, zarr

ROOT="/orcd/data/abodner/002/cody/LLC_patch/_layout_bench"
TILE=720; NSIDE=6; NT=4; VARS=["Theta","Salt","U","V"]
SAMPLE_BYTES=205*TILE*TILE*2
JOBS=[(t,j*TILE,i*TILE) for t in range(NT) for j in range(NSIDE) for i in range(NSIDE)]
random.Random(0).shuffle(JOBS)
WS=[16,32]; REPS=3

_G=None;_L=None
def _init(p,l):
    global _G,_L; _G=zarr.open_group(p,mode="r"); _L=l
def _read(job):
    t,j0,i0=job
    if _L=="packed":
        a=np.asarray(_G["prognostic"][t,:,j0:j0+TILE,i0:i0+TILE])
    else:
        ps=[np.asarray(_G[v][t,:,j0:j0+TILE,i0:i0+TILE]) for v in VARS]
        ps.append(np.asarray(_G["Eta"][t,j0:j0+TILE,i0:i0+TILE])[None])
        a=np.concatenate(ps,axis=0)
    return a.nbytes

def evict(path):
    fs=[os.path.join(d,f) for d,_,x in os.walk(path) for f in x]
    def drop(p):
        try:
            fd=os.open(p,os.O_RDONLY)
            try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally: os.close(fd)
        except OSError: pass
    with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(drop,fs))

def run(path,layout,w):
    with Pool(w,initializer=_init,initargs=(path,layout)) as pool:
        t0=time.perf_counter(); sum(pool.imap_unordered(_read,JOBS,chunksize=1))
        return time.perf_counter()-t0

stores=sorted(d for d in os.listdir(ROOT) if d.endswith(".zarr"))
lay=lambda s:"packed" if s.startswith("packed") else "compact"

CSV=f"{ROOT}/results.csv"
if not os.path.exists(CSV):
    with open(CSV,"w") as f: f.write("phase,store,workers,seconds\n")
def record(phase,store,w,dt):
    with open(CSV,"a") as f: f.write(f"{phase},{store},{w},{dt:.4f}\n"); f.flush()

cold={}
for rep in range(REPS):
    order=stores[:]; random.Random(100+rep).shuffle(order)
    for st in order:
        p=f"{ROOT}/{st}"
        for w in WS:
            evict(p); dt=run(p,lay(st),w)
            cold.setdefault((st,w),[]).append(dt); record("cold",st,w,dt)
        evict(p)
    print(f"[cold] rep {rep+1}/{REPS} done",flush=True)

warm={}
for st in stores:
    p=f"{ROOT}/{st}"
    run(p,lay(st),32)                      # prime
    warm[st]=min(run(p,lay(st),32) for _ in range(2)); record("warm",st,32,warm[st])
    evict(p)                               # bound page cache -> no OOM
    print(f"[warm] {st[:-5]}",flush=True)

print(f"\n{'store':38s} " + " ".join(f"{'cold W='+str(w):>16s}" for w in WS) + f" {'warm W=32':>12s} {'io share':>9s}")
for st in stores:
    row=f"{st[:-5]:38s} "
    for w in WS:
        v=cold[(st,w)]; med=statistics.median(v)
        row+=f" {len(JOBS)*SAMPLE_BYTES/med/1e9:5.2f} GB/s({min(v):4.1f}-{max(v):4.1f})"
    c32=statistics.median(cold[(st,32)])
    row+=f" {len(JOBS)*SAMPLE_BYTES/warm[st]/1e9:7.1f} GB/s {100*(1-warm[st]/c32):7.0f}%"
    print(row,flush=True)

# absolute cost of the compact->contiguous reorder, no I/O in the way
a=[np.empty((51,TILE,TILE),dtype=np.float16) for _ in range(4)]+[np.empty((1,TILE,TILE),dtype=np.float16)]
t0=time.perf_counter()
for _ in range(20): np.concatenate(a,axis=0)
print(f"\npure reorder (concatenate 5 -> (205,720,720)): {(time.perf_counter()-t0)/20*1000:.1f} ms/sample")
