# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stream native-cadence ERA5 and DUACS SSH into persistent local Zarr stores.

The public source metadata and selected coordinates are frozen before transfer.
Every block is decoded, written losslessly and read back before checkpointing.
Source bytes stay in memory; no raw archive is retained. Missing source chunks
are errors, never implicit all-fill arrays. Radiation/precipitation retain the
provider's accumulation units and timestamps without conversion to fluxes.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr
from numcodecs import Blosc

from ocean_preprocessing.obs_preprocessing import full_range as fr
from ocean_preprocessing.obs_preprocessing import prepare as prep

ERA5_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DUACS_URL = (
    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/"
    "SEALEVEL_GLO_PHY_L4_MY_008_047/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411/timeChunked.zarr"
)
ERA5_FIELDS = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
)
PRODUCTS = {"era5-surface": ERA5_FIELDS, "duacs-ssh": ("adt", "sla")}
BLOCK = 24
logger = logging.getLogger(__name__)
_local = threading.local()


def _fetch(url):
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    for attempt in range(3):
        try:
            response = _local.session.get(url, timeout=(10, 90))
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status != 429 and status < 500:
                raise OSError(f"Source HTTP {status}: {url}") from exc
            if attempt == 2:
                raise OSError(
                    f"Source unavailable after three attempts: {url}"
                ) from exc
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


class StrictHTTPStore(MutableMapping):
    """Read-only Zarr mapping: absent chunks must not silently become NaNs."""

    def __init__(self, url, metadata):
        self.url = url.rstrip("/")
        self.metadata = {k: json.dumps(v).encode() for k, v in metadata.items()}
        self.metadata[".zmetadata"] = json.dumps(
            dict(zarr_consolidated_format=1, metadata=metadata)
        ).encode()

    def __getitem__(self, key):
        if key in self.metadata:
            return self.metadata[key]
        if re.fullmatch(r"[A-Za-z0-9_]+/\d+(?:\.\d+)*", key):
            return _fetch(self.url + "/" + key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __setitem__(self, key, value):
        raise TypeError("Read-only source")

    def __delitem__(self, key):
        raise TypeError("Read-only source")

    def __contains__(self, key):
        return key in self.metadata or bool(
            re.fullmatch(r"[A-Za-z0-9_]+/\d+(?:\.\d+)*", key)
        )


def _open(plan):
    return xr.open_zarr(
        StrictHTTPStore(plan["source_uri"], plan["source_metadata"]),
        consolidated=True,
        chunks={},
    )[plan["variables"]]


def _array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _coordinates(ds):
    return {name: _array_hash(ds[name].values) for name in ds.coords}


def _code_hash():
    return fr._digest(
        {
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(fr.__file__), Path(prep.__file__))
        }
    )


def discover(product, manifest_path, start_date="1993-01-01", end_date=None):
    """Freeze metadata and native dates; ERA5 uses finalized coverage only."""
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product {product}")
    path = Path(manifest_path)
    if path.exists():
        raise FileExistsError(path)
    era = product == "era5-surface"
    url = ERA5_URL if era else DUACS_URL
    names = list(PRODUCTS[product]) + ["time", "latitude", "longitude"]
    keys = [".zgroup", ".zattrs"] + [
        f"{name}/{suffix}" for name in names for suffix in (".zarray", ".zattrs")
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda key: json.loads(_fetch(url + "/" + key)), keys))
    metadata = dict(zip(keys, values, strict=True))
    for name in names:
        a = metadata[f"{name}/.zarray"]
        if a["zarr_format"] != 2 or a.get("dimension_separator", ".") != ".":
            raise ValueError("Unsupported upstream Zarr format")
    plan = dict(
        schema_version=1,
        product=product,
        source_uri=url,
        source_metadata=metadata,
        variables=list(PRODUCTS[product]),
        discovered_at=datetime.datetime.now(datetime.UTC).isoformat(),
        frequency="h" if era else "D",
        provider_release="final ERA5 via ARCO" if era else "202411",
        processing="native cadence/grid/decoded values/masks; coordinate sorting; lossless compression",
        source_retention="none; each prepared block is read-back compared before checkpointing",
    )
    with _open(plan) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        first = pd.Timestamp(start_date)
        last = (
            pd.Timestamp(metadata[".zattrs"]["valid_time_stop"])
            + pd.Timedelta(hours=23)
            if era
            else times[-1]
        )
        if era and first < pd.Timestamp(metadata[".zattrs"]["valid_time_start"]):
            raise ValueError("Requested range precedes finalized ERA5")
        if end_date is not None:
            requested = pd.Timestamp(end_date)
            if requested > last:
                raise ValueError("Requested end exceeds finalized source coverage")
            last = requested
        expected = pd.date_range(first, last, freq=plan["frequency"])
        selected = ds.sel(time=slice(first, last))
        if expected.empty or not pd.DatetimeIndex(selected.time.values).equals(
            expected
        ):
            raise ValueError(
                "Source does not contain exact requested native timestamps"
            )
        reference = prep._standardize_daily(selected.isel(time=slice(0, 0)))
        shape = (721, 1440) if era else (1440, 2880)
        if (reference.sizes["lat"], reference.sizes["lon"]) != shape:
            raise ValueError("Unexpected source grid")
        plan.update(
            start_date=str(first),
            end_date=str(last),
            time_count=len(expected),
            coordinate_sha256=_coordinates(selected),
            sizes=dict(time=len(expected), lat=shape[0], lon=shape[1]),
            normalized_coordinates={
                k: _array_hash(reference[k].values) for k in ("lat", "lon")
            },
            decoded_dtypes={n: str(ds[n].dtype) for n in plan["variables"]},
        )
    fr._atomic_json(path, plan)
    return {k: v for k, v in plan.items() if k != "source_metadata"}


def _read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan["schema_version"] != 1 or plan["product"] not in PRODUCTS:
        raise ValueError("Unsupported surface manifest")
    dates = pd.date_range(plan["start_date"], plan["end_date"], freq=plan["frequency"])
    if len(dates) != plan["time_count"] or plan["variables"] != list(
        PRODUCTS[plan["product"]]
    ):
        raise ValueError("Inconsistent surface manifest")
    return plan


def _verify_source(plan):
    # Root update timestamps may advance. Array schemas/packing must not change.
    for key, expected in plan["source_metadata"].items():
        if key in (".zattrs", ".zgroup"):
            continue
        if json.loads(_fetch(plan["source_uri"] + "/" + key)) != expected:
            raise ValueError(f"Source metadata changed: {key}")


def _template(plan, first, staging):
    dates = pd.date_range(plan["start_date"], plan["end_date"], freq=plan["frequency"])
    template = xr.Dataset(
        coords={"time": dates, "lat": first.lat, "lon": first.lon},
        attrs={
            "m2lines/inventory_sha256": fr._digest(plan),
            "m2lines/processing_code_sha256": _code_hash(),
            "m2lines/source_uri": plan["source_uri"],
            "m2lines/native_cadence": plan["frequency"],
            "m2lines/processing": plan["processing"],
            "m2lines/source_history": plan["source_metadata"][".zattrs"],
            "m2lines/accumulations": "Provider units and valid times retained without flux conversion",
        },
    )
    for name, v in first.data_vars.items():
        if v.dims != ("time", "lat", "lon"):
            raise ValueError(f"Unexpected source dimensions: {name} {v.dims}")
        template[name] = xr.DataArray(
            da.empty(
                (len(dates), first.sizes["lat"], first.sizes["lon"]),
                chunks=(BLOCK, 180, 360),
                dtype=v.dtype,
            ),
            dims=v.dims,
            attrs=v.attrs,
        )
    encoding = {
        name: {"compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)}
        for name in template.data_vars
    }
    encoding["time"] = dict(
        units="hours since 1900-01-01", calendar="proleptic_gregorian", dtype="int64"
    )
    template.to_zarr(
        staging,
        mode="w-",
        compute=False,
        consolidated=False,
        encoding=encoding,
        zarr_format=2,
        write_empty_chunks=True,
    )


def run(manifest_path, output_root, workers=16, max_seconds=36000):
    """Resume blockwise transfer; return False at a clean wall-time checkpoint."""
    began = time.monotonic()
    plan = _read_plan(manifest_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    product = plan["product"]
    final = root / f"{product}.zarr"
    staging = root / f".{product}.partial.zarr"
    checkpoint = root / f".{product}.progress.json"
    receipts = root / f"{product}.blocks"
    if final.exists():
        validate(manifest_path, str(final))
        return True
    _verify_source(plan)
    identity = dict(
        inventory_sha256=fr._digest(plan),
        processing_code_sha256=_code_hash(),
        block_size=BLOCK,
    )
    with _open(plan) as source:
        source = source.sel(time=slice(plan["start_date"], plan["end_date"]))
        if _coordinates(source) != plan["coordinate_sha256"]:
            raise ValueError("Source coordinates changed after discovery")
        ds = prep._standardize_daily(source)
        for name in ds.variables:
            ds[name].encoding = {}
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if any(state.get(k) != v for k, v in identity.items()):
                raise ValueError("Checkpoint inventory or processing code mismatch")
            start = state["written"]
            if not 0 <= start <= plan["time_count"] or (
                start != plan["time_count"] and start % BLOCK
            ):
                raise ValueError("Invalid checkpoint boundary")
            with xr.open_zarr(staging, consolidated=False) as old:
                fr._schema_match(ds.isel(time=slice(0, 0)), old)
        else:
            if staging.exists():
                raise FileExistsError(f"Uncheckpointed staging store: {staging}")
            _template(plan, ds.isel(time=slice(0, 0)), staging)
            start = 0
            fr._atomic_json(checkpoint, dict(**identity, written=0))
        receipts.mkdir(exist_ok=True)
        with dask.config.set(scheduler="threads", num_workers=workers):
            for offset in range(start, plan["time_count"], BLOCK):
                if time.monotonic() - began >= max_seconds:
                    logger.info(
                        "Clean pause: %s at %s/%s", product, offset, plan["time_count"]
                    )
                    return False
                stop = min(offset + BLOCK, plan["time_count"])
                block = ds.isel(time=slice(offset, stop)).load()
                for name in plan["variables"]:
                    if (
                        not np.isfinite(block[name].values)
                        .reshape(stop - offset, -1)
                        .any(axis=1)
                        .all()
                    ):
                        raise ValueError(
                            f"All-missing source map: {name} block {offset}"
                        )
                block.drop_vars(list(block.coords)).chunk(
                    dict(time=BLOCK, lat=180, lon=360)
                ).to_zarr(
                    staging,
                    mode="r+",
                    region={"time": slice(offset, stop)},
                    consolidated=False,
                    write_empty_chunks=True,
                )
                with xr.open_zarr(staging, consolidated=False) as stored:
                    for name in plan["variables"]:
                        actual = stored[name].isel(time=slice(offset, stop)).values
                        if not np.array_equal(
                            actual, block[name].values, equal_nan=True
                        ):
                            raise ValueError(
                                f"Lossless read-back failed: {name} block {offset}"
                            )
                fr._atomic_json(
                    receipts / f"{offset:09d}.json",
                    dict(
                        **identity,
                        start=offset,
                        stop=stop,
                        decoded_sha256={
                            n: _array_hash(block[n].values) for n in plan["variables"]
                        },
                        verification="Every decoded value and mask read-back compared exactly",
                    ),
                )
                fr._atomic_json(checkpoint, dict(**identity, written=stop))
                logger.info(
                    "%s verified %s/%s timestamps (%.1f seconds)",
                    product,
                    stop,
                    plan["time_count"],
                    time.monotonic() - began,
                )
                del block
    zarr.consolidate_metadata(str(staging))
    validate(manifest_path, str(staging))
    staging.rename(final)
    checkpoint.unlink()
    return True


def validate(manifest_path, store):
    plan = _read_plan(manifest_path)
    path = Path(store)
    expected = pd.date_range(
        plan["start_date"], plan["end_date"], freq=plan["frequency"]
    )
    with xr.open_zarr(path, consolidated=True) as ds:
        if not pd.DatetimeIndex(ds.time.values).equals(expected):
            raise ValueError("Time axis mismatch")
        if dict(ds.sizes) != plan["sizes"] or set(ds.data_vars) != set(
            plan["variables"]
        ):
            raise ValueError("Schema mismatch")
        if ds.attrs.get("m2lines/inventory_sha256") != fr._digest(plan):
            raise ValueError("Inventory mismatch")
        for name in ("lat", "lon"):
            if _array_hash(ds[name].values) != plan["normalized_coordinates"][name]:
                raise ValueError(f"Coordinate mismatch: {name}")
            if not np.all(np.diff(ds[name].values) > 0):
                raise ValueError(f"Unsorted {name}")
        if float(ds.lon.min()) < 0 or float(ds.lon.max()) >= 360:
            raise ValueError("Invalid longitude")
        fr._check_chunks(path, plan["variables"])
        for name in plan["variables"]:
            if str(ds[name].dtype) != plan["decoded_dtypes"][name]:
                raise ValueError(f"Dtype mismatch: {name}")
            if ds[name].attrs.get("units") != plan["source_metadata"][
                f"{name}/.zattrs"
            ].get("units"):
                raise ValueError(f"Units mismatch: {name}")
            for t in sorted({0, len(expected) // 2, len(expected) - 1}):
                if not np.isfinite(ds[name].isel(time=t).values).any():
                    raise ValueError(f"All-missing sample: {name} {t}")
        code_hash = ds.attrs["m2lines/processing_code_sha256"]
    receipts = path.parent / f"{plan['product']}.blocks"
    for offset in range(0, plan["time_count"], BLOCK):
        receipt = json.loads((receipts / f"{offset:09d}.json").read_text())
        if (
            receipt["inventory_sha256"] != fr._digest(plan)
            or receipt["processing_code_sha256"] != code_hash
            or receipt["start"] != offset
            or receipt["stop"] != min(offset + BLOCK, plan["time_count"])
        ):
            raise ValueError("Block verification receipt mismatch")
    result = dict(
        product=plan["product"],
        time_count=len(expected),
        first=str(expected[0]),
        last=str(expected[-1]),
        sizes=plan["sizes"],
        inventory_sha256=fr._digest(plan),
        verification="All blocks source/output compared during transfer; all chunks present; metadata and decoded samples checked at validation",
    )
    fr._atomic_json(path.parent / f"{plan['product']}.validation.json", result)
    return result


def merge_duacs(
    manifest_path, extension_store, original_manifest, merged_manifest, store
):
    """Add verified SSH arrays by hardlink; publish consolidated metadata last.

    The four existing velocity arrays and coordinates are never rewritten.
    Caller must hold the DUACS processing/publication lock. Both stores must
    share a filesystem. Original and extension inventories remain immutable.
    """
    plan = _read_plan(manifest_path)
    if plan["product"] != "duacs-ssh":
        raise ValueError("Expected DUACS SSH extension")
    validate(manifest_path, extension_store)
    original = fr._read_manifest(original_manifest)
    if original["product"] != "duacs" or original["dataset_version"] != "202411":
        raise ValueError("Original DUACS release differs")
    target, extension = Path(store), Path(extension_store)
    combined = dict(original)
    combined["variables"] = original["variables"] + plan["variables"]
    if len(set(combined["variables"])) != len(combined["variables"]):
        raise ValueError("SSH variables already in original inventory")
    combined["ssh_extension"] = dict(
        inventory_sha256=fr._digest(plan),
        source_uri=plan["source_uri"],
        start_date=plan["start_date"],
        end_date=plan["end_date"],
        processing_code_sha256=_code_hash(),
    )
    combined_path = Path(merged_manifest)
    if combined_path.exists() and json.loads(combined_path.read_text()) != combined:
        raise ValueError("Existing combined manifest differs")
    with (
        xr.open_zarr(target, consolidated=True) as old,
        xr.open_zarr(extension, consolidated=True) as extra,
    ):
        for name in ("time", "lat", "lon"):
            xr.testing.assert_equal(old[name], extra[name])
        old_hash = old.attrs.get("m2lines/inventory_sha256")
        if old_hash == fr._digest(combined):
            fr.validate(merged_manifest, store)
            fr._atomic_json(target.parent / "duacs.inventory.json", combined)
            return
        if old_hash != fr._digest(original):
            raise ValueError("Existing DUACS inventory does not match original")
    fr.validate(original_manifest, store)
    # Hardlinks retain identical verified bytes without another 400 GB copy.
    for name in plan["variables"]:
        destination = target / name
        destination.mkdir(exist_ok=True)
        for source in (extension / name).iterdir():
            dest = destination / source.name
            if dest.exists():
                if not os.path.samefile(source, dest):
                    raise FileExistsError(f"Unexpected existing array object: {dest}")
            else:
                os.link(source, dest)
    fr._atomic_json(combined_path, combined)
    metadata = json.loads((target / ".zmetadata").read_text())
    added = json.loads((extension / ".zmetadata").read_text())["metadata"]
    for name in plan["variables"]:
        for suffix in (".zarray", ".zattrs"):
            key = f"{name}/{suffix}"
            metadata["metadata"][key] = added[key]
    attrs = dict(metadata["metadata"][".zattrs"])
    attrs.update(
        {
            "m2lines/inventory_sha256": fr._digest(combined),
            "m2lines/upstream_inventory": combined,
            "m2lines/ssh_extension": combined["ssh_extension"],
            "m2lines/original_velocity_inventory_sha256": fr._digest(original),
        }
    )
    metadata["metadata"][".zattrs"] = attrs
    fr._atomic_json(target / ".zattrs", attrs)
    fr._atomic_json(target / ".zmetadata", metadata)
    fr.validate(merged_manifest, store)
    fr._atomic_json(target.parent / "duacs.inventory.json", combined)


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    fire.Fire(
        dict(discover=discover, run=run, validate=validate, merge_duacs=merge_duacs)
    )
