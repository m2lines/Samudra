# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen upstream inventories and resumable, native-cadence observation stores.

No OM4 calendar, resampling, or time relabeling is used. Discovery is separate
from download so an upstream update cannot change the range halfway through a
job. Preparation writes disjoint time-chunk regions into a local staging store;
a checkpoint advances only after every variable in a region has been written.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import logging
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc

from ocean_preprocessing.obs_preprocessing import download as raw
from ocean_preprocessing.obs_preprocessing import prepare as prep

logger = logging.getLogger("ocean_preprocessing.obs")
PRODUCTS = ("duacs", "oisst", "argo-iap")
BLOCK = 24


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def _links(url: str) -> list[str]:
    parser = _Links()
    with urlopen(url, timeout=60) as response:
        parser.feed(response.read().decode())
    return parser.links


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _processing_digest() -> str:
    """Identify the actual implementation, including uncommitted edits."""
    modules = (Path(__file__), Path(prep.__file__), Path(raw.__file__))
    return _digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    )


def _continuous(times: pd.DatetimeIndex, frequency: str) -> None:
    if times.empty or not times.equals(
        pd.date_range(times[0], times[-1], freq=frequency)
    ):
        raise ValueError(f"Expected a nonempty, consecutive {frequency} time axis")


def _oisst_records() -> list[dict]:
    base = raw.OISST_BASE_URL + "/"
    months = sorted({v.strip("/") for v in _links(base) if re.fullmatch(r"\d{6}/?", v)})
    if not months:
        raise ValueError("NOAA inventory contains no monthly directories")

    def month_records(month):
        url = base + month + "/"
        # Preliminary filenames contain _preliminary and cannot match this.
        names = sorted(
            {
                v
                for v in _links(url)
                if re.fullmatch(r"oisst-avhrr-v02r01\.\d{8}\.nc", v)
            }
        )
        return [
            dict(
                date=str(pd.Timestamp(n.split(".")[1]).date()),
                path=n,
                url=urljoin(url, n),
            )
            for n in names
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return sorted(
            (r for records in pool.map(month_records, months) for r in records),
            key=lambda r: r["date"],
        )


def _iap_records() -> list[dict]:
    records = []
    dates_by_field = []
    for field, config in raw.ARGO_FIELDS.items():
        url = config.base_urls[0] + "/"
        pattern = rf"IAP_05_2000m_{field}_year_(\d{{4}})_month_(\d{{2}})\.nc"
        found = {}
        for name in _links(url):
            if match := re.fullmatch(pattern, name, flags=re.IGNORECASE):
                date = f"{match[1]}-{match[2]}-01"
                if date in found:
                    raise ValueError(f"Duplicate IAP {field} month {date}")
                found[date] = dict(
                    date=date, path=f"{field}/{name}", url=urljoin(url, name)
                )
        if not found:
            raise ValueError(f"No {field} files in {url}")
        dates_by_field.append(set(found))
        records.extend(found.values())
    if dates_by_field[0] != dates_by_field[1]:
        raise ValueError(
            "Upstream IAP temperature/salinity months differ; inspect before freezing a joint archive"
        )
    return sorted(records, key=lambda r: (r["date"], r["path"]))


def _catalogue_time(coord: dict, key: str) -> pd.Timestamp:
    value = coord[key]
    if isinstance(value, (float, int)):
        unit = coord["coordinate_unit"]
        if not unit.startswith("milliseconds since 1970-01-01"):
            raise ValueError(f"Unsupported numeric Copernicus time unit: {unit}")
        return pd.Timestamp(value, unit="ms")
    return pd.Timestamp(value).tz_localize(None)


def _duacs_inventory(include_ssh: bool) -> dict:
    import copernicusmarine

    catalogue = copernicusmarine.describe(
        dataset_id=raw.DUACS_DATASET_ID, disable_progress_bar=True
    ).model_dump(mode="json")
    datasets = [
        d
        for p in catalogue["products"]
        for d in p["datasets"]
        if d["dataset_id"] == raw.DUACS_DATASET_ID
    ]
    if len(datasets) != 1 or len(datasets[0]["versions"]) != 1:
        raise ValueError(
            "Expected one current DUACS dataset version; inspect the catalogue"
        )
    version = datasets[0]["versions"][0]
    services = [
        s
        for p in version["parts"]
        if p["name"] == "default"
        for s in p["services"]
        if s["service_short_name"] == "geoseries"
    ]
    if len(services) != 1:
        raise ValueError("Expected one DUACS geoseries service")
    variables = list(raw.DUACS_VELOCITY_VARIABLES) + (
        list(raw.DUACS_SSH_VARIABLES) if include_ssh else []
    )
    ranges = []
    for name in variables:
        coords = [
            c
            for v in services[0]["variables"]
            if v["short_name"] == name
            for c in v["coordinates"]
            if c["coordinate_id"] == "time"
        ]
        if len(coords) != 1:
            raise ValueError(f"Missing or ambiguous DUACS time coordinate for {name}")
        first, last = (
            _catalogue_time(coords[0], k) for k in ("minimum_value", "maximum_value")
        )
        ranges.append((first, last))
    if len(set(ranges)) != 1:
        raise ValueError("DUACS variables have different time coverage")
    first, last = ranges[0]
    # This product is midnight-labelled daily. Refuse a changed convention.
    if first != first.normalize() or last != last.normalize():
        raise ValueError(
            "DUACS timestamp convention changed; inspect before downloading"
        )
    return dict(
        start_date=str(first.date()),
        end_date=str(last.date()),
        dataset_id=raw.DUACS_DATASET_ID,
        dataset_version=version["label"],
        source_uri=services[0]["uri"],
        variables=variables,
    )


def discover(product: str, manifest_path: str, include_ssh: bool = False) -> dict:
    """Freeze the complete *current* upstream inventory; never overwrite a plan."""
    if product not in PRODUCTS:
        raise ValueError(f"Expected one of {PRODUCTS}, got {product}")
    path = Path(manifest_path)
    if path.exists():
        raise FileExistsError(f"Frozen inventory already exists: {path}")
    result = dict(
        schema_version=1,
        product=product,
        discovered_at=datetime.datetime.now(datetime.UTC).isoformat(),
        frequency="MS" if product == "argo-iap" else "D",
    )
    if product == "duacs":
        result.update(_duacs_inventory(include_ssh))
        dates = pd.date_range(result["start_date"], result["end_date"], freq="D")
    else:
        records = _oisst_records() if product == "oisst" else _iap_records()
        # IAP has two files per month; OISST must have exactly one per day.
        dates = pd.DatetimeIndex(sorted({r["date"] for r in records}))
        if product == "oisst" and len(dates) != len(records):
            raise ValueError("Duplicate NOAA daily files")
        _continuous(dates, result["frequency"])
        result.update(
            records=records,
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
        )
    result["time_count"] = len(dates)
    result["variables"] = result.get(
        "variables", ["sst"] if product == "oisst" else ["temp", "salt"]
    )
    _atomic_json(path, result)
    logger.info(
        "Frozen %s: %s..%s, %d timestamps -> %s",
        product,
        result["start_date"],
        result["end_date"],
        len(dates),
        path,
    )
    return {k: v for k, v in result.items() if k != "records"}


def _read_manifest(path: str) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan["schema_version"] != 1 or plan["product"] not in PRODUCTS:
        raise ValueError("Unsupported observation manifest")
    dates = pd.date_range(plan["start_date"], plan["end_date"], freq=plan["frequency"])
    if len(dates) != plan["time_count"]:
        raise ValueError("Manifest time count disagrees with date range")
    if plan["product"] != "duacs":
        for field in (
            ["temperature", "salinity"] if plan["product"] == "argo-iap" else [None]
        ):
            records = [
                r
                for r in plan["records"]
                if field is None or r["path"].startswith(field + "/")
            ]
            if not pd.DatetimeIndex([r["date"] for r in records]).equals(dates):
                raise ValueError(
                    f"Manifest contains missing, duplicate or unordered {field or 'OISST'} dates"
                )
    return plan


def _validate_netcdf(path: Path, record: dict, product: str) -> None:
    with xr.open_dataset(path, decode_times=(product == "oisst")) as ds:
        if product == "oisst":
            if (
                pd.Timestamp(ds.time.values[0]).normalize()
                != pd.Timestamp(record["date"])
                or ds.sizes["time"] != 1
            ):
                raise ValueError(f"OISST filename/time disagreement: {path}")
            if "preliminary" in str(ds.attrs.get("title", "")).lower():
                raise ValueError(f"Preliminary OISST file in final inventory: {path}")
            name = "sst"
        else:
            aliases = (
                prep.ARGO_TEMP_ALIASES
                if record["path"].startswith("temperature/")
                else prep.ARGO_SALT_ALIASES
            )
            name = prep._find_var(ds, aliases)
            if name is None:
                raise ValueError(f"Missing IAP variable: {path}")
        # Read the entire retained field, not just its header: a truncated file
        # above the size floor is still a failed download. One file at a time.
        if not np.isfinite(ds[name].values).any():
            raise ValueError(f"No finite data in {path}:{name}")


def download(
    manifest_path: str,
    raw_root: str,
    workers: int = 4,
    credentials_file: str | None = None,
) -> None:
    """Fetch a frozen inventory on scratch, promoting only completed files."""
    plan = _read_manifest(manifest_path)
    product = plan["product"]
    root = Path(raw_root) / product
    root.mkdir(parents=True, exist_ok=True)
    identity = root / "inventory.json"
    if identity.exists() and json.loads(identity.read_text()) != plan:
        raise ValueError(f"{root} belongs to a different inventory; use a new raw_root")
    _atomic_json(identity, plan)
    if product == "duacs":
        raw.duacs(
            root,
            start_date=plan["start_date"],
            end_date=plan["end_date"],
            variables=tuple(plan["variables"]),
            dataset_version=plan["dataset_version"],
            credentials_file=credentials_file,
        )
        for first, last in raw._yearly_ranges(plan["start_date"], plan["end_date"]):
            _check_chunks(
                root / f"duacs_{first:%Y%m%d}_{last:%Y%m%d}.zarr", plan["variables"]
            )
        return
    floor = raw.MIN_BYTES_OISST if product == "oisst" else raw.MIN_BYTES_ARGO
    for offset in range(0, len(plan["records"]), 64):
        records = plan["records"][offset : offset + 64]
        pending = []
        for record in records:
            target = root / record["path"]
            if raw._is_complete(target, floor):
                try:
                    _validate_netcdf(target, record, product)
                except (OSError, ValueError, KeyError):
                    logger.warning("Replacing unreadable/incomplete input: %s", target)
                    pending.append(record)
            else:
                pending.append(record)
        with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as staging:
            tasks = [(r["url"], Path(staging) / r["path"]) for r in pending]
            failed = raw._download_batch(
                tasks,
                workers=workers,
                segments=1,
                retries=4,
                min_bytes=floor,
                label=product,
            )
            for record in pending:
                if record["url"] in failed:
                    continue
                temporary = Path(staging) / record["path"]
                _validate_netcdf(temporary, record, product)
                target = root / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(target)
            if failed:
                raise RuntimeError(
                    f"{len(failed)} downloads failed; rerun the same inventory"
                )
        logger.info(
            "%s: checked/downloaded %d/%d files",
            product,
            min(offset + 64, len(plan["records"])),
            len(plan["records"]),
        )


@contextlib.contextmanager
def _open_block(plan: dict, root: Path, dates: pd.DatetimeIndex):
    with contextlib.ExitStack() as stack:
        if plan["product"] == "duacs":
            datasets = []
            for first, last in raw._yearly_ranges(plan["start_date"], plan["end_date"]):
                if first > dates[-1].date() or last < dates[0].date():
                    continue
                path = root / f"duacs_{first:%Y%m%d}_{last:%Y%m%d}.zarr"
                ds = stack.enter_context(xr.open_zarr(path, chunks={}))
                expected = pd.date_range(first, last, freq="D")
                if not pd.DatetimeIndex(ds.time.values).equals(expected):
                    raise ValueError(f"Incomplete DUACS year: {path}")
                datasets.append(
                    ds[plan["variables"]].sel(
                        time=slice(str(dates[0].date()), str(dates[-1].date()))
                    )
                )
            ds = xr.concat(
                datasets,
                dim="time",
                join="exact",
                data_vars="minimal",
                coords="minimal",
                compat="equals",
            )
        else:
            records = [
                r
                for r in plan["records"]
                if str(dates[0].date()) <= r["date"] <= str(dates[-1].date())
            ]
            if plan["product"] == "oisst":
                ds = stack.enter_context(
                    xr.open_mfdataset(
                        [root / r["path"] for r in records],
                        combine="by_coords",
                        join="exact",
                    )
                )[["sst"]]
                if "zlev" in ds.dims:
                    ds = ds.squeeze("zlev", drop=True)
            else:
                fields = []
                for field, output, aliases in [
                    ("temperature", "temp", prep.ARGO_TEMP_ALIASES),
                    ("salinity", "salt", prep.ARGO_SALT_ALIASES),
                ]:
                    paths = [
                        root / r["path"]
                        for r in records
                        if r["path"].startswith(field + "/")
                    ]
                    source = stack.enter_context(
                        xr.open_mfdataset(
                            paths,
                            combine="nested",
                            concat_dim="time",
                            decode_times=False,
                            join="exact",
                        )
                    )
                    name = prep._find_var(source, aliases)
                    if name is None:
                        raise ValueError(f"Missing IAP {field} variable")
                    source = source.assign_coords(time=dates)
                    fields.append(
                        source[[name]].rename({name: output}).astype("float32")
                    )
                ds = xr.merge(fields, join="exact", compat="equals")
        ds = prep._standardize_daily(ds)
        actual = pd.DatetimeIndex(ds.time.values)
        expected = dates + (
            pd.Timedelta(hours=12) if plan["product"] == "oisst" else pd.Timedelta(0)
        )
        if not actual.equals(expected):
            raise ValueError(
                "Source timestamps differ from the frozen native-cadence inventory"
            )
        # Source encodings describe input packing/chunks, not the output layout.
        # Preserve decoded values, units and masks; do not quantize DUACS floats.
        ds = ds.copy()
        for name in ds.variables:
            ds[name].encoding = {}
        yield ds


def _layout(ds: xr.Dataset) -> dict:
    return {
        dim: min(size, dict(time=BLOCK, lat=180, lon=360, depth_std=16).get(dim, size))
        for dim, size in ds.sizes.items()
    }


def _schema_match(ds: xr.Dataset, reference: xr.Dataset) -> None:
    if set(ds.data_vars) != set(reference.data_vars):
        raise ValueError("Observation variables changed between blocks")
    for name in ds.data_vars:
        if (
            ds[name].dims != reference[name].dims
            or ds[name].dtype != reference[name].dtype
        ):
            raise ValueError(f"Observation schema changed: {name}")
        if ds[name].attrs.get("units") != reference[name].attrs.get("units"):
            raise ValueError(f"Observation units changed: {name}")
    xr.testing.assert_equal(ds.drop_dims("time"), reference.drop_dims("time"))


def prepare(
    manifest_path: str, raw_root: str, output_root: str, threads: int = 4
) -> None:
    """Stream one native-cadence Zarr v2 store; restart from the last full block.

    Output must be local scratch; transfer to OSN from the DTN after validation.
    Only one preparation process may write a given output_root/product.
    """
    if "://" in output_root:
        raise ValueError("Prepare on local scratch, then publish with the DTN script")
    plan = _read_manifest(manifest_path)
    product = plan["product"]
    root = Path(raw_root) / product
    if json.loads((root / "inventory.json").read_text()) != plan:
        raise ValueError("Raw archive does not match the frozen manifest")
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    final, staging = out / f"{product}.zarr", out / f".{product}.partial.zarr"
    state_path = out / f".{product}.progress.json"
    fingerprint = _digest(plan)
    processing_digest = _processing_digest()
    if final.exists():
        validate(manifest_path, str(final))
        _atomic_json(out / f"{product}.inventory.json", plan)
        logger.info("Already complete: %s", final)
        return
    if product == "duacs":
        for first, last in raw._yearly_ranges(plan["start_date"], plan["end_date"]):
            _check_chunks(
                root / f"duacs_{first:%Y%m%d}_{last:%Y%m%d}.zarr", plan["variables"]
            )
    dates = pd.date_range(plan["start_date"], plan["end_date"], freq=plan["frequency"])
    with _open_block(plan, root, dates[:BLOCK]) as first:
        times = dates + (
            pd.Timedelta(hours=12) if product == "oisst" else pd.Timedelta(0)
        )
        reference = first.isel(time=slice(0, 0)).load()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (
                state["inventory_sha256"] != fingerprint
                or state["block_size"] != BLOCK
                or state.get("processing_code_sha256") != processing_digest
            ):
                raise ValueError(
                    "Partial output belongs to a different inventory/layout"
                )
            start = state["written"]
            if not 0 <= start <= len(dates) or (start != len(dates) and start % BLOCK):
                raise ValueError("Invalid preparation checkpoint boundary")
            with xr.open_zarr(staging, consolidated=False) as stored:
                _schema_match(first, stored)
        else:
            if staging.exists():
                raise FileExistsError(
                    f"Uncheckpointed staging store: {staging}; inspect/remove it before retrying"
                )
            coords = {
                name: value
                for name, value in first.coords.items()
                if "time" not in value.dims
            }
            coords["time"] = times
            template = xr.Dataset(coords=coords, attrs=first.attrs)
            chunks = _layout(first)
            chunks["time"] = BLOCK
            for name, variable in first.data_vars.items():
                shape = tuple(
                    len(times) if d == "time" else first.sizes[d] for d in variable.dims
                )
                template[name] = xr.DataArray(
                    da.empty(
                        shape,
                        chunks=tuple(chunks[d] for d in variable.dims),
                        dtype=variable.dtype,
                    ),
                    dims=variable.dims,
                    attrs=variable.attrs,
                )
            template = prep._with_provenance(
                template,
                product,
                alignment=prep.MONTHLY if product == "argo-iap" else prep.RAW_DAILY,
            )
            template.attrs.update(
                {
                    "m2lines/inventory_sha256": fingerprint,
                    "m2lines/processing_code_sha256": processing_digest,
                    "m2lines/upstream_inventory": {
                        k: v for k, v in plan.items() if k != "records"
                    },
                    "m2lines/source_history": first.attrs.get("history", ""),
                    "m2lines/native_time_labels": "provider daily labels retained; IAP calendar months from filenames",
                }
            )
            # Fixed time encoding avoids first-block-derived units on later writes.
            encoding = {
                name: {
                    "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
                }
                for name in template.data_vars
            }
            encoding["time"] = {
                "units": "days since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
            }
            template.to_zarr(
                staging,
                mode="w-",
                compute=False,
                consolidated=False,
                zarr_format=2,
                encoding=encoding,
                write_empty_chunks=True,
            )
            start = 0
            _atomic_json(
                state_path,
                dict(
                    inventory_sha256=fingerprint,
                    processing_code_sha256=processing_digest,
                    block_size=BLOCK,
                    written=0,
                ),
            )
    with dask.config.set(scheduler="threads", num_workers=threads):
        for offset in range(start, len(dates), BLOCK):
            stop = min(offset + BLOCK, len(dates))
            with _open_block(plan, root, dates[offset:stop]) as ds:
                _schema_match(ds, reference)
                ds = ds.chunk(_layout(ds))
                # Static coords and the full time coordinate were written once.
                payload = ds.drop_vars(list(ds.coords))
                payload.to_zarr(
                    staging,
                    mode="r+",
                    region={"time": slice(offset, stop)},
                    consolidated=False,
                    write_empty_chunks=True,
                )
            _atomic_json(
                state_path,
                dict(
                    inventory_sha256=fingerprint,
                    processing_code_sha256=processing_digest,
                    block_size=BLOCK,
                    written=stop,
                ),
            )
            logger.info("%s: wrote %d/%d timestamps", product, stop, len(dates))
    zarr.consolidate_metadata(str(staging))
    validate(manifest_path, str(staging))
    staging.rename(final)
    state_path.unlink()
    _atomic_json(out / f"{product}.inventory.json", plan)
    logger.info("Complete: %s", final)


def _check_chunks(path: Path, variables: list[str]) -> None:
    import itertools
    import math

    for name in variables:
        a = json.loads((path / name / ".zarray").read_text())
        separator = a.get("dimension_separator", ".")
        for index in itertools.product(
            *(
                range(math.ceil(s / c))
                for s, c in zip(a["shape"], a["chunks"], strict=True)
            )
        ):
            if not (path / name / separator.join(map(str, index))).is_file():
                raise ValueError(f"Missing chunk: {path / name}/{index}")


def validate(manifest_path: str, store: str) -> dict:
    """Check time/schema/provenance, all chunk objects, and sample decoded data."""
    plan = _read_manifest(manifest_path)
    path = Path(store)
    with xr.open_zarr(path, consolidated=True) as ds:
        expected = pd.date_range(
            plan["start_date"], plan["end_date"], freq=plan["frequency"]
        )
        if plan["product"] == "oisst":
            expected += pd.Timedelta(hours=12)
        if not pd.DatetimeIndex(ds.time.values).equals(expected):
            raise ValueError("Prepared time axis disagrees with inventory")
        if ds.attrs.get("m2lines/inventory_sha256") != _digest(plan):
            raise ValueError("Prepared provenance disagrees with inventory")
        if set(ds.data_vars) != set(plan["variables"]):
            raise ValueError("Prepared variables disagree with inventory")
        for dim in ["lat", "lon"]:
            if not np.all(np.diff(ds[dim].values) > 0):
                raise ValueError(f"Non-monotone {dim}")
        if float(ds.lon.min()) < 0 or float(ds.lon.max()) >= 360:
            raise ValueError("Longitude is outside [0, 360)")
        _check_chunks(path, list(ds.data_vars))
        for name in ds.data_vars:
            for t in sorted({0, len(expected) // 2, len(expected) - 1}):
                sample = ds[name].isel(time=t)
                if "depth_std" in sample.dims:
                    sample = sample.isel(depth_std=0)
                if not np.isfinite(sample.values).any():
                    raise ValueError(f"No finite values in {name} at sample {t}")
        result = dict(
            product=plan["product"],
            first=str(expected[0]),
            last=str(expected[-1]),
            time_count=len(expected),
            sizes=dict(ds.sizes),
            inventory_sha256=_digest(plan),
        )
    logger.info("Validated %s", result)
    return result


class FullRange:
    """Discover, download, prepare and validate without temporal averaging."""

    discover = staticmethod(discover)
    download = staticmethod(download)
    prepare = staticmethod(prepare)
    validate = staticmethod(validate)
