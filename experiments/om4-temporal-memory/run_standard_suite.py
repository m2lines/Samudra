"""Use repository observation metrics and spectra on the experimental rollout."""

import argparse
import json
import logging
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from common import Prepared, sha256

from samudra.constants import build_om4_layout
from samudra.metrics import observations, report
from samudra.viz.observations import spectra_figures


def dataset(values, data, times, name):
    layout = build_om4_layout()
    array = da.from_array(values, chunks=(10, 19, 180, 360), name=name, asarray=False)
    variables: dict[str, tuple[tuple[str, ...], da.Array]] = {}
    for i, key in enumerate(("uo", "vo", "thetao", "so")):
        block = slice(19 * i, 19 * (i + 1))
        physical = (
            array[:, block] * data.std[block, None, None] + data.mean[block, None, None]
        )
        physical = da.where(data.mask[block], physical, np.nan)
        variables[key] = (("time", "lev", "lat", "lon"), physical)
    ssh = da.where(data.mask[76], array[:, 76] * data.std[76] + data.mean[76], np.nan)
    variables["zos"] = (("time", "lat", "lon"), ssh)
    ds = xr.Dataset(
        variables,
        coords={
            "time": pd.to_datetime(times),
            "lev": np.asarray(layout.depth_levels),
            "lat": data.meta["latitude"],
            "lon": data.meta["longitude"],
        },
    )
    ds["thetao"].attrs["units"] = "degC"
    ds["so"].attrs["units"] = "psu"
    ds["zos"].attrs["units"] = "m"
    ds = ds.assign_coords(dz=("lev", np.asarray(layout.depth_thickness)))
    ds["areacello"] = (
        ("lat", "lon"),
        observations.spherical_cell_area(ds.lat.values, ds.lon.values),
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "COMPLETE").exists():
        return
    assert (args.rollout / "EXPORT_COMPLETE").exists()
    meta = json.loads((args.rollout / "metadata.json").read_text())
    data = Prepared(args.root / "training-data-archive")
    p = np.load(args.rollout / "normalized-prediction.npy", mmap_mode="r")
    model = dataset(p, data, meta["times"], meta["arm"] + "-" + meta["selection"])
    start = meta["start"] + 2
    gold = dataset(data.prog[start : start + len(p)], data, meta["times"], "om4-gold")
    assert model.sizes["time"] == len(meta["times"])
    assert np.array_equal(model.lev.values, np.array(build_om4_layout().depth_levels))
    # Source zero fill is restored to NaN before observation interpolation/reduction.
    for name, k in (("thetao", 38), ("so", 57), ("uo", 0), ("vo", 19)):
        plane = model[name].isel(time=0, lev=0).compute().values
        assert np.isnan(plane[~data.mask[k]]).all()
        np.testing.assert_allclose(
            plane[data.mask[k]],
            (p[0, k] * data.std[k] + data.mean[k])[data.mask[k]],
            rtol=0,
            atol=0,
        )
    if args.validate_only:
        print(
            "Validated times, depths, physical-unit conversion and wet masks",
            flush=True,
        )
        return
    assert (args.observations / "COMPLETE").exists()
    products = {
        name: observations.with_cell_area(
            observations.standardize(
                xr.open_zarr(args.observations / f"{store}.zarr", chunks={})
            )
        )
        for name, store in (
            ("duacs", "duacs"),
            ("oisst", "oisst"),
            ("argo", "argo-iap"),
        )
    }
    rollouts = {meta["arm"] + "-" + meta["selection"]: model, "OM4": gold}
    # Fixed standard suite window, identical across every checkpoint and arm.
    window = (pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"))
    with dask.config.set(scheduler="threads", num_workers=4):
        frame = report.compute_observation_metrics(
            rollouts,
            duacs=products["duacs"],
            oisst=products["oisst"],
            argo=products["argo"],
            model_dz={name: ds.dz for name, ds in rollouts.items()},
            window=window,
            velocity_kind="absolute",
        )
        frame.to_csv(args.output / "observation_metrics.csv", index=False)
        # Spectra use exactly the primary evaluation window on both model and products.
        spectral_models = {
            name: ds.sel(time=slice(*window)) for name, ds in rollouts.items()
        }
        spectral_products = {
            name: ds.sel(time=slice(*window)) for name, ds in products.items()
        }
        figures = spectra_figures(
            spectral_models,
            spectral_products,
            str(args.output / "spectra"),
            velocity_kind="absolute",
        )
    provenance = meta | {
        "metrics_window": ["2015-01-01", "2022-12-31"],
        "metrics_source": str(Path(report.__file__).resolve()),
        "metrics_source_sha256": sha256(Path(report.__file__)),
        "spectral_figures": figures,
        "area": "samudra.metrics.observations.spherical_cell_area on native Gaussian coordinates",
        "observation_manifest": json.loads(
            (args.observations / "manifest.json").read_text()
        ),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "COMPLETE").write_text(
        "Standard observation suite and spectra complete.\n"
    )


if __name__ == "__main__":
    main()
