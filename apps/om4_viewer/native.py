# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded-cache reads and geographic sampling of native OM4."""

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

RAW_URL = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily.zarr"
SNAPSHOT_URL = (
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily_snapshots.zarr"
)
GRID_URL = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr"
HGRID_URL = (
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr"
)
DIVERGING = {"uo", "vo", "zos", "hfds", "wfo", "tauuo", "tauvo"}
PUBLISHED_RAW_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/"


class SourceReadError(OSError):
    """A data-source failure whose message is safe to return to the browser."""


def authenticated_published_read(url):
    """Read the identical published object with the bucket's local rclone keys."""
    if not url.startswith(PUBLISHED_RAW_ROOT):
        raise ValueError("Authenticated fallback is restricted to published OM4 inputs")
    location = "nyu-osn-m2lines-pubs:m2lines-pubs/Samudra/raw/" + url.removeprefix(
        PUBLISHED_RAW_ROOT
    )
    try:
        result = subprocess.run(
            ["rclone", "cat", location, "--retries", "2", "--low-level-retries", "2"],
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise SourceReadError(
            f"OSN denied public access and the server-side authenticated read failed: {url}"
        ) from None
    if result.returncode:
        # Never include rclone's authentication/config diagnostics in a response.
        raise SourceReadError(
            f"OSN denied public access and the server-side authenticated read failed: {url}"
        )
    return result.stdout


def field_limits(base, samples):
    values = np.concatenate([sample.ravel() for sample in samples])
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    if base in DIVERGING:
        high = max(abs(low), abs(high), 1e-6)
        low = -high
    if high <= low:
        high = low + 1.0
    return {
        "min": float(low),
        "max": float(high),
        "palette": "balance" if base in DIVERGING else "thermal",
    }


def time_labels(values, offset_hours=0):
    """Format CF dates without dropping the half-day offset of mean records."""
    values = np.asarray(values)
    if offset_hours:
        if np.issubdtype(values.dtype, np.datetime64):
            values = values + np.timedelta64(offset_hours, "h")
        else:
            values = [value + timedelta(hours=offset_hours) for value in values]
    return [str(value).replace("T", " ").split(".")[0] for value in values]


class DiskCache:
    """Cache compressed objects atomically; concurrent readers share downloads."""

    def __init__(self, root, max_bytes=2 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.key_locks = {}

    def get(self, url, read=None):
        digest = hashlib.sha256(url.encode()).hexdigest()
        target = self.root / digest
        with self.lock:
            key_lock = self.key_locks.setdefault(digest, threading.Lock())
        with key_lock:
            with self.lock:
                if target.exists():
                    content = target.read_bytes()
                    target.touch()
                    return content
            for attempt in range(3):
                try:
                    if read is None:
                        with urlopen(url, timeout=60) as response:
                            content = response.read()
                    else:
                        content = read()
                    break
                except HTTPError as exc:
                    if exc.code == 403 and url.startswith(PUBLISHED_RAW_ROOT):
                        content = authenticated_published_read(url)
                        break
                    if exc.code == 404:
                        # Zarr must not turn an absent remote data chunk into a
                        # believable all-fill frame. Only optional metadata may miss.
                        if url.rsplit("/", 1)[-1].startswith("."):
                            raise KeyError(url) from exc
                        raise SourceReadError(
                            f"Remote data chunk is missing: {url}"
                        ) from exc
                    if exc.code < 500 or attempt == 2:
                        raise SourceReadError(
                            f"OSN read failed ({exc.code}): {url}"
                        ) from exc
                    time.sleep(0.2 * (attempt + 1))
                except (URLError, TimeoutError, ConnectionError) as exc:
                    if attempt == 2:
                        raise SourceReadError(f"OSN read failed: {url}: {exc}") from exc
                    time.sleep(0.2 * (attempt + 1))
            with self.lock:
                fd, temporary = tempfile.mkstemp(prefix=".download-", dir=self.root)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(content)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                files = sorted(
                    (p for p in self.root.iterdir() if not p.name.startswith(".")),
                    key=lambda p: p.stat().st_mtime,
                )
                size = sum(p.stat().st_size for p in files)
                for path in files:
                    if size <= self.max_bytes:
                        break
                    size -= path.stat().st_size
                    path.unlink()
            return content


class PublicStore(MutableMapping):
    def __init__(self, url, cache):
        self.url = url.rstrip("/")
        self.cache = cache

    def __getitem__(self, key):
        if key.startswith("/") or ".." in key.split("/"):
            raise KeyError(key)
        return self.cache.get(self.url + "/" + key)

    def __setitem__(self, key, value):
        raise TypeError("Read-only OSN store")

    def __delitem__(self, key):
        raise TypeError("Read-only OSN store")

    def __iter__(self):
        metadata = json.loads(self[".zmetadata"])["metadata"]
        return iter(metadata)

    def __len__(self):
        return len(json.loads(self[".zmetadata"])["metadata"])

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True


def xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack(
        (
            np.cos(lat).ravel() * np.cos(lon).ravel(),
            np.cos(lat).ravel() * np.sin(lon).ravel(),
            np.sin(lat).ravel(),
        )
    )


class NativeDataset:
    """Keep source cells; rasterize nearest cells only at the requested viewport."""

    def __init__(
        self,
        cache,
        names,
        url=RAW_URL,
        grid_url=GRID_URL,
        hgrid_url=HGRID_URL,
        snapshot=False,
        geometry_source=None,
    ):
        self.cache = cache
        self.data = xr.open_zarr(
            PublicStore(url, cache), chunks=None, consolidated=True
        )
        if "z_l" in self.data.dims:
            self.data = self.data.rename({"z_l": "lev"})
        self.path = url
        self.grid_url, self.hgrid_url = grid_url, hgrid_url
        self.snapshot = snapshot
        self.geometry_source = geometry_source
        self.name = (
            "OM4 native ¼° snapshots · OSN" if snapshot else "OM4 native ¼° means · OSN"
        )
        self.times = time_labels(self.data.time.values)
        if snapshot:
            self.interval_ends = self.times
        elif "time_bnds" in self.data:
            self.interval_ends = time_labels(self.data.time_bnds.isel(nv=1).values)
        else:
            self.interval_ends = time_labels(self.data.time.values, offset_hours=60)
        self.fields = []
        self.keys = {}
        levels = self.data.lev.values
        for base, name in names.items():
            if base not in self.data:
                continue
            array = self.data[base]
            entries = []
            for level in range(len(levels)) if "lev" in array.dims else [None]:
                key = f"{base}_{level}" if level is not None else base
                self.keys[key] = (base, level)
                depth = float(levels[level]) if level is not None else None
                entries.append(
                    {
                        "key": key,
                        "level": level,
                        "label": f"{depth:g} m" if depth is not None else "Surface",
                        "units": str(array.attrs.get("units", "")),
                        "sampling": "instantaneous"
                        if "time: point" in str(array.attrs.get("cell_methods", ""))
                        else "five-day mean",
                    }
                )
            self.fields.append({"id": base, "name": name, "levels": entries})
        self.geometry = None
        self.raw_cache = OrderedDict()
        self.frames = OrderedDict()
        self.lookups = OrderedDict()
        self.limits = lru_cache(maxsize=128)(self._limits)

    def _limits(self, key):
        return field_limits(
            self.keys[key][0],
            [
                self.frame(key, index)
                for index in sorted({0, len(self.times) // 2, len(self.times) - 1})
            ],
        )

    def metadata(self):
        return {
            "name": self.name,
            "path": self.path,
            "times": self.times,
            "interval_end_times": self.interval_ends,
            "sampling": "snapshot" if self.snapshot else "five-day mean",
            "source_run": self.data.attrs.get("title"),
            "calendar": self.data.time.encoding.get("calendar", "standard"),
            "grid": "native",
            "shape": [int(self.data.sizes["yh"]), int(self.data.sizes["xh"])],
            "fields": self.fields,
            "note": "Native tripolar cells · geographic nearest-cell display · currents centered and rotated east/north",
        }

    def _geometry(self):
        if self.geometry_source is not None:
            return self.geometry_source._geometry()
        if self.geometry is None:
            grid = xr.open_zarr(
                PublicStore(self.grid_url, self.cache), chunks=None, consolidated=True
            )
            mosaic = xr.open_zarr(
                PublicStore(self.hgrid_url, self.cache), chunks=None, consolidated=True
            )
            with ThreadPoolExecutor(max_workers=3) as pool:
                lon, lat, wet = list(
                    pool.map(
                        lambda key: np.asarray(grid[key]), ["geolon", "geolat", "wet"]
                    )
                )
            angle = np.asarray(
                mosaic.angle_dx.isel(nyp=slice(1, None, 2), nxp=slice(1, None, 2))
            )
            if lon.shape != angle.shape:
                raise ValueError("Native geometry and rotation shapes differ")
            self.geometry = {
                "lon": (lon + 180) % 360 - 180,
                "lat": lat,
                "wet": wet > 0,
                "cos": np.cos(np.deg2rad(angle)),
                "sin": np.sin(np.deg2rad(angle)),
                "tree": KDTree(xyz(lon, lat), leafsize=16),
            }
            grid.close()
            mosaic.close()
        return self.geometry

    def _raw(self, base, index, level=None):
        source = self.data[base]
        chunks = dict(zip(source.dims, source.encoding["chunks"]))
        start = index // chunks["time"] * chunks["time"]
        depth_start = (
            level // chunks["lev"] * chunks["lev"] if level is not None else None
        )
        key = (base, start, depth_start)
        if key not in self.raw_cache:
            slices = {"time": slice(start, start + chunks["time"])}
            if level is not None:
                slices["lev"] = slice(depth_start, depth_start + chunks["lev"])
            array = np.asarray(source.isel(slices), dtype=np.float32)
            self.raw_cache[key] = array
            # Means chunk one time/all depths; snapshots chunk ten times/one
            # depth. Cache the actual source chunk, never every snapshot depth.
            while (
                sum(value.nbytes for value in self.raw_cache.values()) > 384 * 1024**2
                and len(self.raw_cache) > 1
            ):
                self.raw_cache.popitem(last=False)
        self.raw_cache.move_to_end(key)
        chunk = self.raw_cache[key][index - start]
        return chunk[level - depth_start] if level is not None else chunk

    @staticmethod
    def center(array, shape):
        """Average face values to tracer centers, matching the preparation pipeline."""
        y, x = shape
        array = np.nan_to_num(array, nan=0.0)
        if array.shape[-1] == x + 1:
            return (array[..., :-1] + array[..., 1:]) * 0.5
        if array.shape[-2] == y + 1:
            return (array[..., :-1, :] + array[..., 1:, :]) * 0.5
        if array.shape[-2:] != shape:
            raise ValueError("Unexpected native variable shape")
        return array

    def frame(self, key, index):
        if key not in self.keys or not 0 <= index < len(self.times):
            raise ValueError("Unknown field or time index")
        cache_key = (key, index)
        if cache_key not in self.frames:
            base, level = self.keys[key]
            geo = self._geometry()
            # The tracer's finite mask is authoritative at each level. Surface
            # fields use the native wet mask, rather than interpreting zero as land.
            if level not in (None, 0):
                tracer = self._raw("thetao", index, level)
                wet = np.isfinite(tracer) & geo["wet"]
            else:
                wet = geo["wet"]
            if base in {"uo", "vo", "tauuo", "tauvo"}:
                ubase, vbase = (
                    ("uo", "vo") if base in {"uo", "vo"} else ("tauuo", "tauvo")
                )
                u = self._raw(ubase, index, level)
                # Copy the selected centered slice before another chunk evicts it.
                u = self.center(u, geo["lat"].shape)
                v = self._raw(vbase, index, level)
                v = self.center(v, geo["lat"].shape)
                a = (
                    u * geo["cos"] - v * geo["sin"]
                    if base == ubase
                    else u * geo["sin"] + v * geo["cos"]
                )
            else:
                a = self._raw(base, index, level)
                a = a.copy()
            a[~wet] = -np.inf
            self.frames[cache_key] = a.astype("<f4")
            while len(self.frames) > 12:
                self.frames.popitem(last=False)
        self.frames.move_to_end(cache_key)
        return self.frames[cache_key]

    def raster(self, key, index, bounds, width, height):
        geo = self._geometry()
        cache_key = (*bounds, width, height)
        if cache_key not in self.lookups:
            west, south, east, north = bounds
            lon, lat = np.meshgrid(
                np.linspace(west, east, width, endpoint=False)
                + (east - west) / (2 * width),
                np.linspace(north, south, height, endpoint=False)
                - (north - south) / (2 * height),
            )
            distance, indices = geo["tree"].query(xyz(lon, lat), workers=2)
            self.lookups[cache_key] = (indices, distance < np.deg2rad(0.8))
            while len(self.lookups) > 6:
                self.lookups.popitem(last=False)
        indices, valid = self.lookups[cache_key]
        result = self.frame(key, index).ravel()[indices].copy()
        result[~valid] = -np.inf
        return result.reshape(height, width)

    def close(self):
        self.limits.cache_clear()
        self.data.close()
        self.frames.clear()
        self.raw_cache.clear()
        self.lookups.clear()
