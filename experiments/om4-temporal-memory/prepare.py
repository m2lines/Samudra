"""Pack exact OM4 inputs once on CPUs; never substitute forcing definitions."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from common import BOUND, DEPTHS, PROG


def canonical(name):
    for v in ("uo", "vo", "thetao", "so"):
        if name.startswith(v + "_lev_"):
            depth = float(name[len(v) + 5 :].replace("_", "."))
            return f"{v}_{DEPTHS.index(depth)}"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument(
        "--statistics",
        required=True,
        help="Frozen official-archive normalization JSON; local copies differ",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists():
        print("Already complete", flush=True)
        return
    ds = xr.open_zarr(str(Path(args.data) / "OM4.zarr"), chunks=None)
    z = zarr.open_consolidated(str(Path(args.data) / "OM4.zarr"), mode="r")
    means_ds = xr.open_zarr(str(Path(args.data) / "OM4_means.zarr"), chunks=None)
    stds_ds = xr.open_zarr(str(Path(args.data) / "OM4_stds.zarr"), chunks=None)
    means = {canonical(k): float(v.values) for k, v in means_ds.data_vars.items()}
    stds = {canonical(k): float(v.values) for k, v in stds_ds.data_vars.items()}
    frozen = json.loads(Path(args.statistics).read_text())
    means.update(frozen["OM4_means.zarr"])
    stds.update(frozen["OM4_stds.zarr"])
    # The archived evaluator computes climatology and anomaly mean/std over the
    # full source BEFORE selecting its inference window. Reproduce that here.
    print("Computing archived hfds anomaly definition", flush=True)
    heat = ds.hfds.load()
    climatology = heat.groupby("time.dayofyear").mean("time")
    anomaly = (heat - climatology.sel(dayofyear=heat.time.dt.dayofyear)).values
    means["hfds_anomalies"] = float(np.nanmean(anomaly))
    stds["hfds_anomalies"] = float(np.nanstd(anomaly))
    dates = [str(t) for t in ds.time.values]
    origins = [
        next(i for i, t in enumerate(dates) if t[:10] >= date)
        for date in [
            "2014-10-10",
            "2015-01-10",
            "2015-04-10",
            "2015-07-10",
            "2016-01-10",
            "2016-07-10",
        ]
    ]
    indices = (
        np.unique(np.concatenate([np.arange(i, i + 26) for i in origins]))
        if args.diagnostic_only
        else np.arange(len(dates))
    )
    mask = np.stack(
        [z[f"mask_{int(k.rsplit('_', 1)[1]) if k != 'zos' else 0}"][:] for k in PROG]
    ).astype(bool)
    surface = z["mask_0"][:].astype(bool)
    np.save(out / "mask.npy", mask)
    shape = (len(indices), 77, *mask.shape[-2:])
    p = np.lib.format.open_memmap(
        out / "prognostic.npy", mode="w+", dtype="float32", shape=shape
    )
    b = np.lib.format.open_memmap(
        out / "boundary.npy",
        mode="w+",
        dtype="float32",
        shape=(len(indices), 4, *mask.shape[-2:]),
    )

    def read(pair):
        k, ix = pair
        a = anomaly[ix] if k == "hfds_anomalies" else z[k].oindex[ix, :, :]
        valid = mask[PROG.index(k)] if k in PROG else surface
        return np.where(
            valid, np.nan_to_num((a - means[k]) / stds[k], nan=0.0), 0.0
        ).astype("float32")

    start = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        for j in range(0, len(indices), 32):
            ix = indices[j : j + 32]
            vals = list(pool.map(read, [(k, ix) for k in PROG + BOUND]))
            p[j : j + len(ix)] = np.stack(vals[:77], axis=1)
            b[j : j + len(ix)] = np.stack(vals[77:], axis=1)
            print(
                f"Packed {j + len(ix)}/{len(indices)} in {time.monotonic() - start:.1f}s",
                flush=True,
            )
    p.flush()
    b.flush()
    meta = {
        "data_root": args.data,
        "source_attrs": ds.attrs,
        "variables": PROG,
        "boundary_variables": BOUND,
        "means": [means[k] for k in PROG + BOUND],
        "stds": [stds[k] for k in PROG + BOUND],
        "times": [dates[i] for i in indices],
        "source_indices": indices.tolist(),
        "diagnostic_origins": [int(np.where(indices == i)[0][0]) for i in origins],
        "latitude": ds.y.values.tolist(),
        "longitude": ds.x.values.tolist(),
        "normalization": "archived scalar statistics; hfds anomalies recomputed over full source",
        "statistics_file": args.statistics,
        "diagnostic_only": args.diagnostic_only,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
