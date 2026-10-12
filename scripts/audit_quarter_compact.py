#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU qualification of the actual ARM CompactOm4Reader.

Reads one all-channel frame and four tiny indexed windows, never a full period.
The independent reference is raw xarray, without Samudra canonicalization or its
Rust adapter. Static masks and numeric time coordinates also get a direct Zarr
comparison. This qualifies decoding/indexing, not normalization, model training,
IAP observation operators, or the six-tensor QuarterOm4Windows integration.
Native/reference timings are ordered, cache-affected diagnostics, not benchmarks.
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]


def reference_read(dataset, indices, selectors):
    """Group depth selections so xarray decompresses each packed field once."""
    fields = {}
    for base in dict.fromkeys(base for base, _ in selectors):
        array = dataset[base].isel(time=indices)
        levels = list(dict.fromkeys(level for name, level in selectors if name == base))
        if levels == [None]:
            fields[base] = array.transpose("time", "y", "x").values[:, None]
        else:
            fields[base] = (
                array.isel(lev=levels).transpose("time", "lev", "y", "x").values
            )
    channels = []
    for base, level in selectors:
        levels = list(dict.fromkeys(v for name, v in selectors if name == base))
        channels.append(fields[base][:, levels.index(level)])
    return np.stack(channels, axis=1)


def audit(path, native, *, concurrency=4):
    started = time.perf_counter()
    dataset = xr.open_zarr(path, consolidated=True, chunks=None, decode_times=False)
    raw = zarr.open_group(str(path), mode="r")
    try:
        if tuple(dataset.sizes[d] for d in ("lev", "y", "x")) != (19, 720, 1440):
            raise ValueError("Expected the averaged compact 19-level 720x1440 store")
        times = dataset.time.values
        np.testing.assert_array_equal(times, raw["time"][:])
        dates = cftime.num2date(
            times, **{key: dataset.time.attrs[key] for key in ("units", "calendar")}
        )
        if not np.all(np.diff(times) > 0):
            raise ValueError("Time coordinates must be strictly increasing")
        recent = [i for i, date in enumerate(dates) if 1975 <= date.year <= 2013]
        early = [i for i, date in enumerate(dates) if date.year < 1975]
        if len(recent) < 2 or len(early) < 2:
            raise ValueError("Expected both early and recent source coverage")
        masks = {}
        for level in range(19):
            if "wetmask" in dataset:
                mask = dataset.wetmask.isel(lev=level).transpose("y", "x").values
                dims = list(raw["wetmask"].attrs["_ARRAY_DIMENSIONS"])
                index = tuple(level if d == "lev" else slice(None) for d in dims)
                direct = raw["wetmask"][index]
            else:
                mask = dataset[f"mask_{level}"].transpose("y", "x").values
                direct = raw[f"mask_{level}"][:]
            np.testing.assert_array_equal(mask, direct)
            if not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid wet mask at level {level}")
            masks[level] = mask.astype(bool)
        all_channels: list[tuple[str, int | None]] = [
            (base, lev) for base in ("uo", "vo", "thetao", "so") for lev in range(19)
        ]
        all_channels += [(base, None) for base in ("zos", "tauuo", "tauvo", "hfds")]
        # Deliberately permute both levels and variables and duplicate channels.
        subset = [all_channels[i] for i in (56, 0, 79, 38, 76, 19, 57, 56)]
        cases = [
            ("first-all-channels", [0], all_channels),
            ("early-overlap", [early[1], early[0], early[1]], subset),
            ("recent-overlap", [recent[1], recent[0], recent[1]], subset),
            ("recent-end", [recent[-1], recent[-2]], subset),
            ("physical-last", [len(times) - 1, len(times) - 2], subset),
        ]
        pool = native.FlatOm4ReadPool(concurrency)
        reader = native.CompactOm4Reader(str(path), all_channels, pool)
        if tuple(reader.shape) != (len(times), 720, 1440):
            raise ValueError("Native reader shape differs from xarray")
        results = []
        for name, indices, selectors in cases:
            output = np.empty((len(indices), len(selectors), 720, 1440), np.float32)
            before = time.perf_counter()
            reader.read_into(indices, selectors, output)
            native_seconds = time.perf_counter() - before
            before = time.perf_counter()
            reference = reference_read(dataset, indices, selectors)
            reference_seconds = time.perf_counter() - before
            np.testing.assert_array_equal(output, reference, err_msg=name)
            for channel, (base, channel_level) in enumerate(selectors):
                wet = masks[0 if channel_level is None else channel_level]
                if not np.isfinite(output[:, channel, wet]).all():
                    raise ValueError(
                        f"Nonfinite wet data in {name}: {base}/{channel_level}"
                    )
            results.append(
                dict(
                    name=name,
                    indices=indices,
                    dates=[str(dates[i]) for i in indices],
                    selectors=selectors,
                    output_bytes=output.nbytes,
                    native_seconds=native_seconds,
                    xarray_seconds=reference_seconds,
                    sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                    equal_including_nan=True,
                )
            )
            del output, reference
        return dict(
            status="passed",
            scope=__doc__,
            source=str(path),
            machine=platform.machine(),
            module_file=native.__file__,
            numpy=np.__version__,
            xarray=xr.__version__,
            zarr=zarr.__version__,
            concurrency=concurrency,
            shape=list(reader.shape),
            time_attributes=dict(dataset.time.attrs),
            first=str(dates[0]),
            last=str(dates[-1]),
            numeric_time_delta_counts={
                str(v): int(n)
                for v, n in zip(*np.unique(np.diff(times), return_counts=True))
            },
            mask_wet_counts=[int(masks[i].sum()) for i in range(19)],
            source_metadata_sha256=hashlib.sha256(
                (path / ".zmetadata").read_bytes()
            ).hexdigest(),
            cases=results,
            wall_seconds=time.perf_counter() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import samudra_rust_loader as native  # type: ignore[import-not-found]

    if platform.machine() != "aarch64":
        raise RuntimeError("Qualification must run on ARM, matching Beta")
    if not Path(native.__file__).resolve().is_relative_to(args.runtime.resolve()):
        raise RuntimeError("Loaded native extension outside the isolated runtime")
    result = audit(args.store, native)
    # Keep binary identity valid when the qualified sidecar is mounted at a
    # different path in training. Record the package entrypoint separately.
    result["module_file_sha256"] = hashlib.sha256(
        Path(native.__file__).read_bytes()
    ).hexdigest()
    result["module_binary_sha256"] = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            (name == native.__name__ or name.startswith(native.__name__ + "."))
            and filename is not None
            and Path(filename).suffix == ".so"
        ):
            result["module_binary_sha256"][Path(filename).name] = hashlib.sha256(
                Path(filename).read_bytes()
            ).hexdigest()
    if not result["module_binary_sha256"]:
        raise RuntimeError("No loaded native reader binary found for audit identity")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
