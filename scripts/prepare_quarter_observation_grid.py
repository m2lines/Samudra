#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the quarter observation grid using only OM4 metadata and static arrays.

Dynamic field chunks are forbidden by the storage wrapper, including accidental
reads introduced by canonical-source changes. Reuse verifies both provenance
and exact freshly read static values; changed artifacts are never overwritten.
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]
from scipy.special import roots_legendre

from samudra.config import DataConfig
from samudra.utils.data import CanonicalSource

STORES = ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class StaticOnlyStore(zarr.storage.DirectoryStore):
    """Read-only Zarr-v2 store refusing time-dependent field payloads."""

    def __init__(self, path):
        super().__init__(str(path))
        metadata = json.loads((Path(path) / ".zmetadata").read_text())["metadata"]
        self.dynamic = {
            key.removesuffix("/.zattrs")
            for key, value in metadata.items()
            if key.endswith("/.zattrs")
            and "time" in value.get("_ARRAY_DIMENSIONS", [])
            and key != "time/.zattrs"
        }
        self.payload_arrays = set()

    def __getitem__(self, key):
        parts = key.split("/")
        if len(parts) > 1 and not parts[-1].startswith("."):
            if parts[0] in self.dynamic:
                raise RuntimeError(f"Forbidden temporal field payload read: {key}")
            self.payload_arrays.add(parts[0])
        return super().__getitem__(key)

    def getitems(self, keys, *, contexts=None):
        return {key: self[key] for key in keys if key in self}

    def __setitem__(self, key, value):
        raise TypeError("StaticOnlyStore is read-only")

    def __delitem__(self, key):
        raise TypeError("StaticOnlyStore is read-only")


def source_config():
    # Only canonicalization is used: no split reads, training backend, or CUDA.
    return DataConfig.model_validate(
        {
            "loading": {"type": "cpu"},
            "sources": [
                {
                    "type": "om4",
                    "grid_type": "gaussian",
                    "prognostic_vars_key": "thermo_dynamic_all",
                    "boundary_vars_key": "tau_hfds",
                    "data_location": STORES[0],
                    "data_means_location": STORES[1],
                    "data_stds_location": STORES[2],
                    "train_time": {"start": "1975-01-03", "end": "2013-10-04"},
                    "val_time": {"start": "2013-10-05", "end": "2014-10-05"},
                }
            ],
        }
    ).sources[0]


def read_grid(source):
    """Canonical static export; caller validates the production grid contract."""
    stores = [StaticOnlyStore(Path(source) / name) for name in STORES]
    datasets = [
        xr.open_zarr(store, consolidated=True, chunks=None, decode_times=False)
        for store in stores
    ]
    try:
        raw, means, stds = datasets
        # Canonical mask construction casts to boolean; reject invalid raw
        # masks before that conversion can turn NaNs into apparent ocean.
        masks = [
            name
            for name in raw.variables
            if str(name).startswith("mask_") or name == "wetmask"
        ]
        if not masks:
            raise ValueError("Missing source wet masks")
        for name in masks:
            if "time" in raw[name].dims:
                raise ValueError("Wet masks must be static")
            values = raw[name].load().values
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"Nonfinite or nonbinary source mask: {name}")
        config = source_config()
        data, means, stds, layout = config.canonicalize_datasets(raw, means, stds)
        canonical = CanonicalSource.from_datasets(
            data,
            means,
            stds,
            data_layout=layout,
            prognostic_var_names=layout.prognostic_var_names,
            boundary_var_names=layout.boundary_var_names,
        )
        names = layout.prognostic_var_names
        statistics = canonical.statistics(names)
        lat, lon = canonical.resolution
        arrays = dict(
            lat=lat.numpy(),
            lon=lon.numpy(),
            mask=canonical.masks.prognostic.numpy(),
            names=np.asarray(names),
            mean=statistics.mean,
            std=statistics.std,
            depth=np.asarray(layout.depth_levels),
            depth_thickness=np.asarray(layout.depth_thickness),
        )
        if "lev" not in raw:
            raise ValueError("Missing source depth coordinate")
        np.testing.assert_array_equal(raw["lev"].values, arrays["depth"])
        optional = {}
        for name in ("lat_b", "lon_b", "areacello", "ocean_fraction", "dz"):
            if name in raw:
                if "time" in raw[name].dims:
                    raise ValueError(f"Expected static source geometry: {name}")
                arrays[name] = raw[name].values.copy()
                optional[name] = dict(
                    dimensions=list(raw[name].dims), shape=list(raw[name].shape)
                )
        producer = {
            key: value
            for key, value in raw.attrs.items()
            if key
            in (
                "m2lines/samudra_git_hash",
                "m2lines/ocean_emulators_git_hash",
                "m2lines/cli_args",
                "m2lines/date_created",
                "grid_type",
                "regrid_method",
                "m2lines/wfo_surgery_source",
                "m2lines/wfo_surgery_alignment",
            )
        }
        evidence = dict(
            source_producer=producer,
            optional_geometry=optional,
            payload_arrays_read={
                name: sorted(store.payload_arrays)
                for name, store in zip(STORES, stores, strict=True)
            },
            temporal_field_payloads_read=0,
        )
        return arrays, evidence
    finally:
        for dataset in datasets:
            dataset.close()


def validate_grid(arrays):
    expected_names = tuple(
        f"{base}_{level}"
        for base in ("uo", "vo", "thetao", "so")
        for level in range(19)
    ) + ("zos",)
    if tuple(arrays["names"]) != expected_names:
        raise ValueError(
            "Canonical prognostic channel order differs from the 77-channel contract"
        )
    if (
        arrays["lat"].shape != (720,)
        or arrays["lon"].shape != (1440,)
        or arrays["mask"].shape != (77, 720, 1440)
    ):
        raise ValueError("Expected a global Gaussian 720x1440 grid with 77 masks")
    expected_lat = np.rad2deg(np.arcsin(roots_legendre(720)[0]))
    np.testing.assert_allclose(arrays["lat"], expected_lat, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        arrays["lon"], (np.arange(1440) + 0.5) / 4, rtol=0, atol=1e-10
    )
    if arrays["mask"].dtype != np.bool_ or not arrays["mask"].any(axis=(-2, -1)).all():
        raise ValueError("Every channel must have a nonempty boolean wet mask")
    for name in ("mean", "std"):
        if arrays[name].shape != (77,) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid canonical {name} statistics")
    if not (arrays["std"] > 0).all():
        raise ValueError("All canonical standard deviations must be positive")
    if (
        not np.isfinite(arrays["depth"]).all()
        or not (np.diff(arrays["depth"]) > 0).all()
    ):
        raise ValueError("Invalid physical depths")
    if "areacello" in arrays:
        area = arrays["areacello"]
        if (
            area.shape != (720, 1440)
            or not np.isfinite(area[arrays["mask"][0]]).all()
            or not (area[arrays["mask"][0]] > 0).all()
        ):
            raise ValueError("Invalid source cell areas on wet support")


def publish_grid(output, arrays, evidence):
    """Verify exact reusable state or publish once without replacing any artifact."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid, ready = output / "grid.npz", output / "GRID_READY.json"
    if grid.exists() or ready.exists():
        if not grid.exists() or not ready.exists():
            raise ValueError("Incomplete grid publication; refusing to overwrite")
        receipt = json.loads(ready.read_text())
        if receipt.get("provenance") != evidence or receipt.get(
            "grid_sha256"
        ) != digest(grid):
            raise ValueError("Existing grid provenance or content hash differs")
        with np.load(grid, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("Existing grid array inventory differs")
            for name, values in arrays.items():
                if saved[name].dtype != values.dtype:
                    raise ValueError(f"Existing grid dtype differs: {name}")
                np.testing.assert_array_equal(
                    saved[name], values, err_msg=f"Source static values changed: {name}"
                )
        return receipt
    with tempfile.TemporaryDirectory(prefix=".grid-", dir=output) as temporary:
        candidate = Path(temporary) / "grid.npz"
        np.savez_compressed(candidate, **arrays)
        with np.load(candidate, allow_pickle=False) as saved:
            for name, values in arrays.items():
                np.testing.assert_array_equal(saved[name], values)
        receipt = dict(
            provenance=evidence,
            grid_sha256=digest(candidate),
            verification="Full NPZ read-back; every array equals canonical source static values",
        )
        # Exclusive publication also protects concurrent callers from replacing
        # an artifact created after the initial existence check.
        os.link(candidate, grid)
        with ready.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def prepare(source, output, producer):
    source = Path(source).resolve()
    complete = source / "COMPLETE.json"
    conversion_hash = digest(complete)
    conversion = json.loads(complete.read_text())
    if Path(
        conversion["destination"]
    ).resolve() != source or "SHA256" not in conversion.get("verification", ""):
        raise ValueError("Conversion receipt does not verify this source")
    metadata_hashes = {name: digest(source / name / ".zmetadata") for name in STORES}
    arrays, evidence = read_grid(source)
    validate_grid(arrays)
    # Refuse a concurrently replaced metadata/receipt set.
    if metadata_hashes != {
        name: digest(source / name / ".zmetadata") for name in STORES
    }:
        raise ValueError("Source metadata changed during grid preparation")
    if conversion_hash != digest(complete):
        raise ValueError("Conversion receipt changed during grid preparation")
    evidence.update(
        source=str(source),
        metadata_sha256=metadata_hashes,
        conversion_receipt_sha256=conversion_hash,
        conversion_receipt=conversion,
        grid_producer=producer,
        code_sha256=digest(__file__),
        canonical_builder_sha256={
            cls.__module__: digest(inspect.getfile(cls))
            for cls in (DataConfig, CanonicalSource)
        },
        schema_version=1,
        normalization="Existing source means/stds in canonical 77-channel order; not refitted",
    )
    return publish_grid(output, arrays, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--producer", default=os.environ.get("SAMUDRA_CODE_COMMIT"))
    args = parser.parse_args()
    producer = (
        args.producer
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    )
    receipt = prepare(args.source, args.output, producer)
    print(
        json.dumps(
            dict(
                grid_sha256=receipt["grid_sha256"],
                output=args.output,
                temporal_field_payloads_read=0,
            )
        )
    )


if __name__ == "__main__":
    main()
