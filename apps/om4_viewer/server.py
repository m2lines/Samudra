# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only, on-demand viewer for rectilinear ocean Zarr datasets."""

import argparse
import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from aiohttp import web
from native import (
    SNAPSHOT_URL,
    DiskCache,
    NativeDataset,
    SourceReadError,
    field_limits,
    time_labels,
)

# Layer centers from samudra.constants.build_om4_layout. The prepared OM4
# stores flatten the depth dimension into field names and omit this coordinate.
OM4_DEPTHS = (
    2.5,
    10,
    22.5,
    40,
    65,
    105,
    165,
    250,
    375,
    550,
    775,
    1050,
    1400,
    1850,
    2400,
    3100,
    4000,
    5000,
    6000,
)
FIELD_NAMES = {
    "thetao": "Temperature",
    "so": "Salinity",
    "uo": "Zonal velocity",
    "vo": "Meridional velocity",
    "zos": "Sea surface height",
    "hfds": "Surface heat flux",
    "wfo": "Freshwater flux into ocean",
    "tauuo": "Zonal wind stress",
    "tauvo": "Meridional wind stress",
}
STATIC = Path(__file__).parent


class Dataset:
    def __init__(self, path: str, *, data=None, name=None):
        root = Path(path).expanduser().resolve()
        if (root / "OM4.zarr").is_dir():
            root = root / "OM4.zarr"
        self.path = root
        self.name = name or (
            "OM4 1° · local"
            if root.parent.name == "om4_onedeg"
            else root.parent.name
            if root.name == "OM4.zarr"
            else root.name
        )
        self.data = xr.open_zarr(root, chunks=None) if data is None else data
        rename = {
            old: new
            for old, new in (("lat", "y"), ("lon", "x"))
            if old in self.data.dims and new not in self.data.dims
        }
        if rename:
            self.data = self.data.rename(rename)
        for coord in ("time", "x", "y"):
            if coord not in self.data.coords or self.data[coord].dims != (coord,):
                raise ValueError(
                    f"{root}: expected one-dimensional {coord!r} coordinate"
                )
        self.lon = (np.asarray(self.data.x, dtype=float) + 180) % 360 - 180
        self.lat = np.asarray(self.data.y, dtype=float)
        self.x_order, self.y_order = np.argsort(self.lon), np.argsort(self.lat)[::-1]
        self.lon, self.lat = self.lon[self.x_order], self.lat[self.y_order]
        if (
            not np.isfinite(self.lon).all()
            or not np.isfinite(self.lat).all()
            or np.any(np.diff(self.lon) <= 0)
            or np.any(np.diff(self.lat) >= 0)
            or len(self.lon) < 2
            or len(self.lat) < 2
        ):
            raise ValueError(f"{root}: coordinates must be finite and unique")
        if not np.allclose(np.diff(self.lon), np.diff(self.lon)[0]):
            raise ValueError(f"{root}: longitude spacing must be uniform")
        self.fields: dict[str, list[dict]] = {}
        self.keys: dict[str, tuple[str, str | None]] = {}
        for key, array in self.data.data_vars.items():
            if set(array.dims) != {"time", "y", "x"}:
                continue
            match = re.fullmatch(r"(.+)_(\d+)", str(key))
            base, level = (match[1], int(match[2])) if match else (str(key), None)
            source_attrs = self.data.attrs.get(base, {})
            inherited_units = (
                source_attrs.get("units", "") if isinstance(source_attrs, dict) else ""
            )
            mask = f"mask_{level if level is not None else 0}"
            self.keys[str(key)] = (base, mask if mask in self.data else None)
            depth = (
                OM4_DEPTHS[level]
                if base in {"thetao", "so", "uo", "vo"}
                and level is not None
                and level < len(OM4_DEPTHS)
                else None
            )
            self.fields.setdefault(base, []).append(
                {
                    "key": str(key),
                    "level": level,
                    "label": f"{depth:g} m"
                    if depth is not None
                    else (f"Level {level}" if level is not None else "Surface"),
                    "units": str(array.attrs.get("units") or inherited_units),
                    "sampling": "five-day mean",
                }
            )
        if not self.fields or not self.data.sizes["time"]:
            raise ValueError(f"{root}: no time-varying fields on (time, y, x)")
        for levels in self.fields.values():
            levels.sort(
                key=lambda item: item["level"] if item["level"] is not None else -1
            )
        self.times = [
            str(t).replace("T", " ").split(".")[0] for t in self.data.time.values
        ]
        # Instance-owned caches: bounded memory and no retained closed datasets.
        frame_count = max(
            4, min(128, 128 * 1024**2 // (len(self.lat) * len(self.lon) * 4))
        )
        self.frame = lru_cache(maxsize=frame_count)(self._frame)
        self.limits = lru_cache(maxsize=128)(self._limits)
        self.mask = lru_cache(maxsize=24)(self._mask)

    def _mask(self, key: str) -> np.ndarray:
        return np.asarray(self.data[key].transpose("y", "x"), dtype=bool)[
            np.ix_(self.y_order, self.x_order)
        ]

    def _frame(self, key: str, index: int) -> np.ndarray:
        if key not in self.keys:
            raise ValueError("Unknown field")
        if not 0 <= index < len(self.times):
            raise ValueError("Time index out of range")
        array = np.asarray(
            self.data[key].isel(time=index).transpose("y", "x"), dtype="<f4"
        )
        array = array[np.ix_(self.y_order, self.x_order)].copy()
        array[~np.isfinite(array)] = np.nan
        mask = self.keys[key][1]
        if mask is not None:
            array[
                ~self.mask(mask)
            ] = -np.inf  # Distinguish land from missing wet cells.
        array.setflags(write=False)
        return array

    def _limits(self, key: str) -> dict:
        samples = [
            self.frame(key, i).ravel()
            for i in sorted({0, len(self.times) // 2, len(self.times) - 1})
        ]
        return field_limits(self.keys[key][0], samples)

    def metadata(self) -> dict:
        return {
            "grid": "rectilinear",
            "shape": [len(self.lat), len(self.lon)],
            "name": self.name,
            "path": str(self.path),
            "times": self.times,
            "interval_end_times": time_labels(self.data.time.values, offset_hours=60),
            "sampling": "five-day mean",
            "source_run": self.data.attrs.get("title"),
            "calendar": self.data.time.encoding.get("calendar", "standard"),
            "longitude": self.lon.tolist(),
            "latitude": self.lat.tolist(),
            "fields": [
                {"id": key, "name": FIELD_NAMES.get(key, key), "levels": levels}
                for key, levels in self.fields.items()
            ],
        }

    def raster(self, key, index, bounds, width, height):
        west, south, east, north = bounds
        lon = (
            np.linspace(west, east, width, endpoint=False)
            + (east - west) / (2 * width)
            + 180
        ) % 360 - 180
        lat = np.linspace(north, south, height, endpoint=False) - (north - south) / (
            2 * height
        )
        # Cell edges preserve the irregular Gaussian latitude spacing.
        xedges = (self.lon[:-1] + self.lon[1:]) / 2
        yedges = (self.lat[::-1][:-1] + self.lat[::-1][1:]) / 2
        xi = np.searchsorted(xedges, lon)
        yi = len(self.lat) - 1 - np.searchsorted(yedges, lat)
        return self.frame(key, index)[np.ix_(yi, xi)]

    def close(self):
        self.frame.cache_clear()
        self.mask.cache_clear()
        self.limits.cache_clear()
        self.data.close()


CM4_LOCATION = "nyu-osn-m2lines-pubs:m2lines-pubs/Samudra/v2026-09/cm4_onedeg/CM4.zarr"


class CM4Dataset(Dataset):
    """Published Gaussian CM4 means, retaining their noleap interval-end dates."""

    def __init__(self, cache):
        from rollouts import ArchiveStore

        source = xr.open_zarr(
            ArchiveStore(CM4_LOCATION, cache), chunks=None, consolidated=True
        )
        # Expose the same field/depth keys as prepared OM4 without rewriting or
        # regridding any samples. These slices stay lazy until a frame is read.
        fields = {}
        for key, array in source.data_vars.items():
            if set(array.dims) == {"time", "lev", "y", "x"}:
                for level in range(source.sizes["lev"]):
                    fields[f"{key}_{level}"] = array.isel(lev=level).reset_coords(
                        drop=True
                    )
            elif set(array.dims) == {"time", "y", "x"}:
                fields[key] = array.reset_coords(drop=True)
        for level in range(source.sizes["lev"]):
            fields[f"mask_{level}"] = source.wetmask.isel(lev=level).reset_coords(
                drop=True
            )
        super().__init__(
            "CM4.zarr",
            data=xr.Dataset(fields, attrs=source.attrs),
            name="CM4 1° · piControl · OSN",
        )
        self.source = source
        # Use the recorded bounds rather than OM4's midpoint + 60 hours rule.
        self.interval_ends = time_labels(source.time_bnds.isel(nv=-1).values)
        depths = np.asarray(source.lev)
        for entries in self.fields.values():
            for entry in entries:
                if entry["level"] is not None:
                    entry["label"] = f"{depths[entry['level']]:g} m"

    def metadata(self):
        result = super().metadata()
        result.update(
            path=CM4_LOCATION,
            interval_end_times=self.interval_ends,
            time_label="interval end",
        )
        return result

    def close(self):
        super().close()
        self.source.close()


class RolloutDataset(Dataset):
    rollout = True

    def __init__(self, cache, resolution, run_id, run_name, truth_directory):
        from rollouts import ARCHIVE_ROOT, TRUTH_ROOT, ArchiveStore

        self.location = f"{ARCHIVE_ROOT}/{run_name}/predictions.zarr"
        self.run_id, self.run_name = run_id, run_name
        data = xr.open_zarr(
            ArchiveStore(self.location, cache), chunks=None, consolidated=True
        )
        super().__init__(
            run_name,
            data=data,
            name=f"Samudra 2 {resolution} · official rollout",
        )
        self.truth_location = f"{TRUTH_ROOT}/{truth_directory}/OM4.zarr"
        self.truth = Dataset(
            truth_directory,
            data=xr.open_zarr(
                ArchiveStore(self.truth_location, cache), chunks=None, consolidated=True
            ),
        )
        if not np.array_equal(self.lon, self.truth.lon) or not np.array_equal(
            self.lat, self.truth.lat
        ):
            raise ValueError(f"Rollout grid does not match its OM4 source: {run_id}")
        units = {
            entry["key"]: entry["units"]
            for field in self.truth.metadata()["fields"]
            for entry in field["levels"]
        }
        for entries in self.fields.values():
            for entry in entries:
                entry["units"] = entry["units"] or units.get(entry["key"], "")
                entry["sampling"] = "forecast · five-day mean"
                key = entry["key"]
                if key not in self.truth.keys or self.truth.keys[key][1] is None:
                    raise ValueError(f"No source ocean mask for {run_id}/{key}")
                self.keys[key] = self.truth.keys[key]

    def _mask(self, key):
        return self.truth.mask(key)

    def close(self):
        super().close()
        self.truth.close()

    def metadata(self):
        result = super().metadata()
        result.update(
            path=self.location,
            source_run=self.run_name,
            source_link=f"https://wandb.ai/ocean_emulators/default/runs/{self.run_id}",
            truth_path=self.truth_location,
            rollout=True,
        )
        return result


class LocalRolloutDataset(Dataset):
    """A prepared physical rollout with its own explicit masks and provenance."""

    rollout = True

    def __init__(self, path):
        super().__init__(path)
        self.name = self.data.attrs["viewer_name"]
        for levels in self.fields.values():
            for entry in levels:
                if self.keys[entry["key"]][1] is None:
                    raise ValueError(f"Local rollout has no ocean mask: {entry['key']}")
                entry["sampling"] = "forecast · five-day mean"

    def metadata(self):
        result = super().metadata()
        result.update(
            rollout=True,
            source_link=self.data.attrs.get("source_link"),
            source_run=self.data.attrs.get("title"),
        )
        return result


def create_app(
    paths: list[str],
    native=False,
    cache_dir=None,
    snapshots=False,
    rollouts=False,
    cm4=False,
    local_rollouts=(),
) -> web.Application:
    datasets: list[Dataset | NativeDataset] = [Dataset(path) for path in paths]
    cache = (
        DiskCache(cache_dir or STATIC / ".cache")
        if native or snapshots or rollouts or cm4
        else None
    )
    means = None
    if native:
        means = NativeDataset(cache, FIELD_NAMES)
        datasets.append(means)
    if snapshots:
        datasets.append(
            NativeDataset(
                cache,
                FIELD_NAMES,
                url=SNAPSHOT_URL,
                snapshot=True,
                geometry_source=means,
            )
        )
    if rollouts:
        from rollouts import OFFICIAL_EVALS

        datasets.extend(RolloutDataset(cache, *entry) for entry in OFFICIAL_EVALS)
    if cm4:
        datasets.append(CM4Dataset(cache))
    datasets.extend(LocalRolloutDataset(path) for path in local_rollouts)
    # Serialize disk reads/cache mutations; leave the event loop free to serve UI.
    read_lock = asyncio.Lock()

    def selected(request):
        index = int(request.query.get("dataset", "0"))
        if not 0 <= index < len(datasets):
            raise ValueError("Unknown dataset")
        return datasets[index]

    @web.middleware
    async def errors(request, handler):
        try:
            return await handler(request)
        except (ValueError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except SourceReadError as exc:
            logging.getLogger(__name__).exception("Source data read failed")
            return web.json_response({"error": str(exc)}, status=502)

    async def metadata(request):
        return web.json_response({"datasets": [d.metadata() for d in datasets]})

    async def raster(request):
        dataset = selected(request)
        key, index = request.query["field"], int(request.query["time"])
        bounds = tuple(
            float(v) for v in request.query.get("bounds", "-180,-90,180,90").split(",")
        )
        width, height = (
            int(request.query.get("width", "1024")),
            int(request.query.get("height", "512")),
        )
        if (
            len(bounds) != 4
            or not np.isfinite(bounds).all()
            or not -90 <= bounds[1] < bounds[3] <= 90
            or not 0 < bounds[2] - bounds[0] <= 360
            or not 16 <= width <= 1440
            or not 16 <= height <= 900
        ):
            raise ValueError("Invalid map bounds or dimensions")
        async with read_lock:
            values = await asyncio.to_thread(
                dataset.raster, key, index, bounds, width, height
            )
        return web.Response(
            body=values.astype("<f4").tobytes(), content_type="application/octet-stream"
        )

    async def frame(request):
        dataset = selected(request)
        key, index = request.query["field"], int(request.query["time"])
        async with read_lock:
            values = await asyncio.to_thread(dataset.frame, key, index)
        return web.Response(
            body=values.tobytes(),
            content_type="application/octet-stream",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    async def limits(request):
        dataset, key = selected(request), request.query["field"]
        if key not in dataset.keys:
            raise ValueError("Unknown field")
        async with read_lock:
            if isinstance(dataset, NativeDataset) or getattr(dataset, "rollout", False):
                match = next(
                    (
                        d
                        for d in datasets
                        if isinstance(d, Dataset)
                        and not getattr(d, "rollout", False)
                        and not isinstance(d, CM4Dataset)
                        and key in d.keys
                    ),
                    None,
                )
                if match is not None:
                    result = await asyncio.to_thread(match.limits, key)
                    result["reference"] = "local 1° reference"
                else:
                    result = await asyncio.to_thread(dataset.limits, key)
                    result["reference"] = "selected source reference"
            else:
                result = await asyncio.to_thread(dataset.limits, key)
                result["reference"] = (
                    "selected CM4 source"
                    if isinstance(dataset, CM4Dataset)
                    else "local reference"
                )
        return web.json_response(result)

    async def index(request):
        return web.FileResponse(STATIC / "index.html")

    async def javascript(request):
        return web.FileResponse(STATIC / "viewer.js")

    async def cleanup(app):
        for dataset in datasets:
            dataset.close()

    app = web.Application(middlewares=[errors])
    app.router.add_get("/", index)
    app.router.add_get("/viewer.js", javascript)
    app.router.add_get("/api/metadata", metadata)
    app.router.add_get("/api/frame", frame)
    app.router.add_get("/api/map", raster)
    app.router.add_get("/api/limits", limits)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="append",
        help="Zarr store or directory containing OM4.zarr; repeat for a dataset selector",
    )
    parser.add_argument(
        "--local-rollout",
        action="append",
        default=[],
        help="Prepared local physical rollout with masks and provenance; repeatable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include public native quarter-degree OM4 (default: on)",
    )
    parser.add_argument(
        "--snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also include native tripolar snapshots (default: on; --no-native disables both)",
    )
    parser.add_argument(
        "--rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the three archived official Samudra 2 evals via local rclone credentials",
    )
    parser.add_argument(
        "--cm4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include published CM4 1-degree piControl means (noleap calendar)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(STATIC / ".cache"),
        help="OSN compressed-chunk cache (bounded to 2 GiB)",
    )
    args = parser.parse_args()
    web.run_app(
        create_app(
            args.data or ["~/data/om4_onedeg"],
            native=args.native,
            cache_dir=args.cache_dir,
            snapshots=args.native and args.snapshots,
            rollouts=args.rollouts,
            cm4=args.cm4,
            local_rollouts=args.local_rollout,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
