"""Stage only the public observation variables/dates used by the standard suite."""

import argparse
import json
from pathlib import Path

import dask
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
products = {"duacs": ["ugos", "vgos"], "oisst": ["sst"], "argo-iap": ["temp"]}
manifest = {}
with dask.config.set(scheduler="threads", num_workers=4):
    for name, variables in products.items():
        url = f"s3://m2lines-pubs/Samudra/v2026-07/obs/{name}.zarr"
        destination = args.output / f"{name}.zarr"
        marker = args.output / f"{name}.COMPLETE"
        if not marker.exists():
            source = xr.open_zarr(
                url,
                storage_options={
                    "anon": True,
                    "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"},
                },
                chunks={},
            )
            ds = source[variables].sel(time=slice("2014-10-01", "2022-12-31"))
            # Slicing can leave an offset first Dask chunk that overlaps encoded
            # Zarr chunks. Restore time alignment before the parallel write.
            ds = ds.chunk({"time": source.chunks["time"][0]})
            print(
                "Staging",
                name,
                dict(ds.sizes),
                "uncompressed bytes",
                ds.nbytes,
                flush=True,
            )
            ds.to_zarr(destination, mode="w", consolidated=True)
            check = xr.open_zarr(destination, chunks={})
            assert dict(check.sizes) == dict(ds.sizes)
            for v in variables:
                xr.testing.assert_allclose(
                    ds[v].isel(time=[0, -1]).compute(),
                    check[v].isel(time=[0, -1]).compute(),
                )
            marker.write_text(
                "Metadata and first/last samples verified; staging completed.\n"
            )
        staged = xr.open_zarr(destination, chunks={})
        manifest[name] = dict(
            source=url,
            endpoint="https://nyu1.osn.mghpcc.org",
            anonymous=True,
            variables=variables,
            sizes=dict(staged.sizes),
            first=str(staged.time.values[0]),
            last=str(staged.time.values[-1]),
            uncompressed_bytes=staged.nbytes,
        )
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
(args.output / "COMPLETE").write_text("All required public observations staged.\n")
