"""Phase A/B: materialize 36 real face-1 tiles once, write them into 12 stores."""
import itertools, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import numcodecs, numpy as np, xarray as xr, zarr

numcodecs.blosc.use_threads = False
SRC="/orcd/data/abodner/003/LLC4320/LLC4320"
ROOT="/orcd/data/abodner/002/cody/LLC_patch/_layout_bench"
FACE=1; T=5000; NT=4; TILE=720; NSIDE=6; VARS=["Theta","Salt","U","V"]
B=numcodecs.Blosc
SHUF={"byteshuffle":B.SHUFFLE,"bitshuffle":B.BITSHUFFLE}
CODEC={"lz4hc-5":("lz4hc",5),"lz4-5":("lz4",5),"zstd-3":("zstd",3)}

g=zarr.open_group(SRC,mode="r")
mn=xr.open_zarr("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr")
sd=xr.open_zarr("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr")
MU={v:np.array([mn[f"{v}_lev_{k}"].values.item() for k in range(51)],dtype=np.float32) for v in VARS}
SD={v:np.array([sd[f"{v}_lev_{k}"].values.item() for k in range(51)],dtype=np.float32) for v in VARS}
EMU=np.float32(mn["Eta"].values); ESD=np.float32(sd["Eta"].values)

# ---- Phase A: read the real face once, normalized float16, land left NaN ----
t0=time.perf_counter()
tiles={}
def load(origin):
    j0,i0=origin
    a=np.empty((205,TILE,TILE),dtype=np.float16)
    for n,v in enumerate(VARS):
        blk=g[v][T,:,FACE,j0:j0+TILE,i0:i0+TILE]
        a[n*51:(n+1)*51]=((blk-MU[v][:,None,None])/SD[v][:,None,None]).astype(np.float16)
    e=g["Eta"][T,FACE,j0:j0+TILE,i0:i0+TILE]
    a[204]=((e-EMU)/ESD).astype(np.float16)
    return origin,a
origins=[(j*TILE,i*TILE) for j in range(NSIDE) for i in range(NSIDE)]
with ThreadPoolExecutor(max_workers=36) as ex:
    for origin,a in ex.map(load,origins): tiles[origin]=a
print(f"[A] 36 tiles in RAM ({sum(a.nbytes for a in tiles.values())/1e9:.2f} GB) in {time.perf_counter()-t0:.0f}s",flush=True)
nan_frac=float(np.isnan(tiles[(720,720)]).mean()), float(np.isnan(tiles[(2160,2160)]).mean())
print(f"[A] NaN fraction sample tiles: {nan_frac}",flush=True)

# ---- size check: does keeping NaN cost anything vs collapsing to 0? ----
probe=tiles[(2160,2160)]
c=numcodecs.Blosc(cname="lz4hc",clevel=5,shuffle=B.SHUFFLE)
n_nan=len(c.encode(probe)); n_zero=len(c.encode(np.nan_to_num(probe,nan=0.0)))
print(f"[A] land=NaN {n_nan/1e6:.1f} MB vs land=0 {n_zero/1e6:.1f} MB "
      f"({100*(n_nan-n_zero)/n_zero:+.2f}%)",flush=True)

# ---- Phase B: write the 12 stores ----
shutil.rmtree(ROOT,ignore_errors=True); os.makedirs(ROOT)
F=4320
for sname,layout,cname in itertools.product(SHUF,["packed","compact"],CODEC):
    tag=f"{layout}__{cname}__{sname}"
    cn,cl=CODEC[cname]
    comp=numcodecs.Blosc(cname=cn,clevel=cl,shuffle=SHUF[sname])
    grp=zarr.open_group(f"{ROOT}/{tag}.zarr",mode="w")
    t0=time.perf_counter()
    if layout=="packed":
        A=grp.create_dataset("prognostic",shape=(NT,205,F,F),chunks=(1,205,TILE,TILE),
                             dtype="f2",compressor=comp,fill_value=float("nan"))
        def wr(job):
            t,(j0,i0)=job; A[t,:,j0:j0+TILE,i0:i0+TILE]=tiles[(j0,i0)]
    else:
        Bs={v:grp.create_dataset(v,shape=(NT,51,F,F),chunks=(1,51,TILE,TILE),dtype="f2",
                                 compressor=comp,fill_value=float("nan")) for v in VARS}
        Bs["Eta"]=grp.create_dataset("Eta",shape=(NT,F,F),chunks=(1,TILE,TILE),dtype="f2",
                                     compressor=comp,fill_value=float("nan"))
        def wr(job):
            t,(j0,i0)=job; a=tiles[(j0,i0)]
            for n,v in enumerate(VARS): Bs[v][t,:,j0:j0+TILE,i0:i0+TILE]=a[n*51:(n+1)*51]
            Bs["Eta"][t,j0:j0+TILE,i0:i0+TILE]=a[204]
    jobs=[(t,o) for t in range(NT) for o in origins]
    with ThreadPoolExecutor(max_workers=48) as ex: list(ex.map(wr,jobs))
    sz=sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(f"{ROOT}/{tag}.zarr") for f in fs)
    nf=sum(len(fs) for _,_,fs in os.walk(f"{ROOT}/{tag}.zarr"))
    print(f"[B] {tag:38s} {sz/1e9:6.2f} GB  {nf:5d} files  {time.perf_counter()-t0:5.0f}s",flush=True)
print("[B] done",flush=True)
