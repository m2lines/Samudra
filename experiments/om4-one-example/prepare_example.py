"""Pack four local OM4 frames with frozen, previously verified normalization.

Read only the required heat-flux day-of-year groups to reproduce the archived
full-source climatology without loading all 4,745 heat-flux frames into RAM.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.output / "COMPLETE").exists(), "Do not overwrite prepared data"
    meta = json.loads((args.reference / "metadata.json").read_text())
    ds = xr.open_zarr(args.source, chunks=None)
    assert ds.attrs == meta["source_attrs"]
    assert np.array_equal(ds.x.values, meta["longitude"])
    assert np.array_equal(ds.y.values, meta["latitude"])
    dates = [str(t) for t in ds.time.values]
    start = next(i for i, date in enumerate(dates) if date[:10] >= args.start_date)
    indices = np.arange(start, start + 4)
    assert indices[-1] < len(dates)
    variables = meta["variables"] + meta["boundary_variables"]
    masks = {i: ds[f"mask_{i}"].values.astype(bool) for i in range(19)}
    wet = np.stack([masks[0 if v == "zos" else int(v.rsplit("_", 1)[1])]
                    for v in meta["variables"]])
    assert np.array_equal(wet, np.load(args.reference / "mask.npy"))
    old_mean = np.asarray(meta["previous_local_means"], dtype=np.float32)
    old_std = np.asarray(meta["previous_local_stds"], dtype=np.float32)
    new_mean = np.asarray(meta["means"], dtype=np.float32)
    new_std = np.asarray(meta["stds"], dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "mask.npy", wet)
    prog = np.lib.format.open_memmap(args.output / "prognostic.npy", mode="w+",
                                    dtype="float32", shape=(4, 77, *wet.shape[-2:]))
    boundary = np.lib.format.open_memmap(args.output / "boundary.npy", mode="w+",
                                        dtype="float32", shape=(4, 4, *wet.shape[-2:]))
    doy = ds.time.dt.dayofyear.values
    anomaly = []
    groups = {}
    for index in indices:
        group = np.flatnonzero(doy == doy[index])
        groups[str(int(doy[index]))] = group.tolist()
        climatology = ds.hfds.isel(time=group).mean("time").values
        anomaly.append(ds.hfds.isel(time=int(index)).values - climatology)
    anomaly = np.stack(anomaly)
    for channel, name in enumerate(variables):
        raw = anomaly if name == "hfds_anomalies" else ds[name].isel(time=indices).values
        valid = wet[channel] if channel < 77 else wet[0]
        # Preserve the original prepare -> rescale float32 round trip.
        first = np.where(valid, np.nan_to_num((raw-old_mean[channel])/old_std[channel], nan=0), 0)
        restored = first * old_std[channel] + old_mean[channel]
        final = np.where(valid, (restored-new_mean[channel])/new_std[channel], 0).astype(np.float32)
        assert np.isfinite(final).all()
        if channel < 77:
            prog[:, channel] = final
        else:
            boundary[:, channel-77] = final
    prog.flush(); boundary.flush()
    meta.update(data_root=str(args.source.parent), times=[dates[i] for i in indices],
                source_indices=indices.tolist(), diagnostic_origins=[0], diagnostic_only=True,
                preparation="Selected-frame CPU packing; frozen statistics and original float32 normalization round trip",
                heat_climatology_groups=groups,
                normalization_reference=str(args.reference))
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.output / "COMPLETE").write_text("complete\n")
    print(json.dumps({"times": meta["times"], "indices": meta["source_indices"],
                      "heat_group_sizes": {k: len(v) for k, v in groups.items()}}))


if __name__ == "__main__":
    main()
