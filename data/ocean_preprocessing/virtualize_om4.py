# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Index native NetCDF chunks as a Parquet Kerchunk store, without rewriting data.

Run with ``python -m ocean_preprocessing.virtualize_om4 --help``.
The local output directory can subsequently be published using rclone.
"""

import ctypes
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import fire
import fsspec
import obstore.store
import xarray as xr
from kerchunk.df import refs_to_dataframe
from obspec_utils.glob import glob
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import HDFParser, NetCDF3Parser

LOG = logging.getLogger(__name__)
OSN = "https://nyu1.osn.mghpcc.org"


@dataclass(frozen=True)
class LocalURIStore:
    """Read URI-escaped object keys through a native obstore LocalStore.

    Both obspec-utils 0.9 and VirtualiZarr 2.8 strip store prefixes without
    decoding file URIs. Keep that namespace encoded, and decode each object key
    exactly once at the local storage boundary. VirtualiZarr reads coordinates
    through its own async store calls, so decoding only our explicit reads is
    insufficient. The wrapped store still performs all filesystem I/O.
    """

    store: obstore.store.LocalStore

    @property
    def prefix(self) -> str:
        return quote(str(self.store.prefix), safe="/")

    def head(self, path):
        return self.store.head(unquote(path))

    def get(self, path, **options):
        return self.store.get(unquote(path), **options)

    def get_range(self, path, **options):
        return self.store.get_range(unquote(path), **options)

    def get_ranges(self, path, **options):
        return self.store.get_ranges(unquote(path), **options)

    async def get_range_async(self, path, **options):
        return await self.store.get_range_async(unquote(path), **options)


@dataclass(frozen=True)
class SourceArchive:
    """Selected source objects and the storage connection that resolves their URLs."""

    urls: tuple[str, ...]
    registry: ObjectStoreRegistry
    pattern: str
    endpoint_url: str | None
    anonymous: bool

    @classmethod
    def discover(
        cls, root: str, pattern: str, endpoint_url: str, anonymous: bool
    ) -> "SourceArchive":
        """Select files in filename order from a local directory or S3 prefix."""
        if root.startswith("s3://"):
            parsed = urlsplit(root)
            base = "s3://" + parsed.netloc
            store = obstore.store.from_url(
                base,
                endpoint=endpoint_url,
                region="us-east-1",
                skip_signature=anonymous,
            )
            prefix = parsed.path.strip("/")
            matches = glob(store, f"{prefix}/{pattern}" if prefix else pattern)
            urls = tuple(f"{base}/{path}" for path in sorted(matches))
        else:
            if "://" in root:
                raise ValueError("Source must be a local directory or s3:// prefix")
            local_root = Path(root).resolve()
            base = local_root.as_uri()
            store = obstore.store.LocalStore(local_root)
            urls = tuple(
                (local_root / path).as_uri() for path in sorted(glob(store, pattern))
            )
            store = LocalURIStore(store)
            endpoint_url = None
        if not urls:
            raise ValueError("No source files matched the pattern")
        return cls(
            urls, ObjectStoreRegistry({base: store}), pattern, endpoint_url, anonymous
        )

    @property
    def protocol(self) -> str:
        """Storage protocol shared by the selected source URLs."""
        return urlsplit(self.urls[0]).scheme

    def snapshot(self) -> list[dict]:
        """Read fresh object metadata for provenance and source-change detection."""
        objects = []
        for url in self.urls:
            store, path = self.registry.resolve(url)
            metadata = store.head(path)
            objects.append(
                {
                    "url": url,
                    "size": metadata["size"],
                    "e_tag": metadata["e_tag"],
                    "version": metadata["version"],
                    "last_modified": metadata["last_modified"].isoformat(),
                }
            )
        return objects


def open_virtual_netcdf(
    url: str, registry: ObjectStoreRegistry, concat_dim: str | None
) -> xr.Dataset:
    """Parse a NetCDF file once, loading only metadata needed for combination."""
    store, path = registry.resolve(url)
    magic = bytes(store.get_range(path, start=0, end=8))
    if magic[:4] in (b"CDF\x01", b"CDF\x02", b"CDF\x05"):
        parser = NetCDF3Parser()
    elif magic == b"\x89HDF\r\n\x1a\n":
        parser = HDFParser()
    else:
        raise ValueError(f"Unrecognized NetCDF signature: {url}")
    manifest = parser(url=url, registry=registry)
    virtual = manifest.to_virtual_dataset(loadable_variables=[], decode_times=False)
    if concat_dim is None:
        return virtual
    if concat_dim not in virtual.coords:
        raise ValueError(f"Missing {concat_dim} coordinate: {url}")
    bounds = {coordinate.attrs.get("bounds") for coordinate in virtual.coords.values()}
    loadable = [
        name
        for name, variable in virtual.variables.items()
        if concat_dim not in variable.dims or name == concat_dim or name in bounds
    ]
    dataset = manifest.to_virtual_dataset(
        loadable_variables=loadable, decode_times=False
    )
    for name in loadable:
        dataset[name].load()
    return dataset


def combine_virtual_datasets(
    datasets: Sequence[xr.Dataset],
    concat_dim: str | None,
    source_urls: Sequence[str] = (),
) -> xr.Dataset:
    """Combine compatible virtual arrays without aligning or filling their grids."""
    if concat_dim is None:
        if len(datasets) != 1:
            raise ValueError("Static virtualization requires exactly one source file")
        return datasets[0]
    reference = datasets[0]
    for candidate in datasets[1:]:
        if set(candidate.variables) != set(reference.variables):
            raise ValueError("Source variable inventories differ")
        for name in reference.variables:
            if candidate[name].dims != reference[name].dims:
                raise ValueError(f"Source dimensions differ for {name}")
            # Xarray compares array-valued attributes as well as scalar attributes.
            if not xr.Dataset(attrs=candidate[name].attrs).identical(
                xr.Dataset(attrs=reference[name].attrs)
            ):
                raise ValueError(f"Source attributes differ for {name}")
    for name, variable in reference.variables.items():
        if concat_dim not in variable.dims or not isinstance(
            variable.data, ManifestArray
        ):
            continue
        axis = variable.dims.index(concat_dim)
        chunks = variable.data.metadata.chunk_grid.chunk_shape
        for index, dataset in enumerate(datasets):
            current = dataset[name].data
            if not isinstance(current, ManifestArray):
                continue
            current_chunks = current.metadata.chunk_grid.chunk_shape
            location = source_urls[index] if source_urls else f"source {index}"
            if current_chunks != chunks:
                raise ValueError(
                    f"Incompatible native chunk shapes for {name!r} in {location}: "
                    f"{current_chunks}, expected {chunks}. Rechunk the inputs before virtualization."
                )
            if (
                index < len(datasets) - 1
                and dataset.sizes[concat_dim] % current_chunks[axis]
            ):
                raise ValueError(
                    f"Partial native chunk for {name!r} in {location} at a non-final file boundary. "
                    "Rechunk the inputs before virtualization."
                )
    return xr.concat(
        datasets,
        dim=concat_dim,
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="override",
    )


def rename_directory_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename a local directory without replacing any destination.

    On POSIX, os.rename replaces an existing empty directory; shutil.move may
    nest the source inside it. Checking existence first leaves a race with other
    writers. Python 3.12 has no standard-library exclusive directory rename, so
    use the OS no-replace operation to preserve any competing writer's output.
    """
    if sys.platform == "win32":
        os.rename(
            source, destination
        )  # Windows rename does not replace existing paths.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        arguments = (
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        raise NotImplementedError(
            "Exclusive directory publication requires Linux, macOS, or Windows"
        )
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def staged_reference_directory(output: Path) -> Iterator[Path]:
    """Publish a complete local directory only when all work inside succeeds."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as tmp:
        directory = Path(tmp) / "references"
        yield directory
        rename_directory_exclusive(directory, output)


def write_parquet_references(
    dataset: xr.Dataset, directory: Path, record_size: int, source: SourceArchive
) -> None:
    """Serialize Zarr v2 chunk references, including scalars and local paths."""
    for name, variable in dataset.variables.items():
        if not isinstance(variable.data, ManifestArray):
            continue
        array = variable.data
        grid_size = math.prod(
            math.ceil(size / chunk)
            for size, chunk in zip(array.shape, array.metadata.chunk_grid.chunk_shape)
        )
        # Missing Parquet rows can decode to Pandas NA/NaN rather than None,
        # which reference readers mistake for paths even with a declared fill.
        # With no CF fill, Zarr v2 null-fill reads can also be uninitialized.
        if sum(1 for _ in array.manifest.iter_refs()) < grid_size:
            raise ValueError(
                f"{name!r} has unwritten native chunks; "
                "sparse Kerchunk Parquet references are not supported. "
                "Materialize those chunks before virtualization."
            )
    references = dataset.vz.to_kerchunk(format="dict", zarr_format=2)
    refs = references["refs"]
    # VirtualiZarr 2.8 emits an empty scalar chunk key. Zarr v2 uses "0";
    # correcting the key also places array metadata before chunks in the writer.
    for name, variable in dataset.variables.items():
        if variable.ndim == 0:
            refs[f"{name}/0"] = refs.pop(f"{name}/")
    # Zarr v2 readers use the storage fill value as a CF missing-data marker.
    # NetCDF variables without _FillValue must retain valid zeros, not mask them.
    for name, variable in dataset.variables.items():
        if "_FillValue" in variable.attrs:
            continue
        metadata = json.loads(refs[f"{name}/.zarray"])
        metadata["fill_value"] = None
        refs[f"{name}/.zarray"] = json.dumps(metadata)
    # VirtualiZarr strips file:// but leaves URI escapes in local reference paths.
    # Kerchunk's local reader expects native paths, not percent-encoded URI paths.
    if source.protocol == "file":
        for reference in refs.values():
            if isinstance(reference, list):
                reference[0] = unquote(reference[0])
    refs_to_dataframe(references, url=str(directory), record_size=record_size)


def validate_parquet_references(
    directory: Path, expected: xr.Dataset, source: SourceArchive
) -> None:
    """Reopen the serialized reference store and check its dataset structure."""
    remote_options = (
        {
            "anon": source.anonymous,
            "endpoint_url": source.endpoint_url,
            "asynchronous": True,
        }
        if source.protocol == "s3"
        else {}
    )
    mapper = fsspec.filesystem(
        "reference",
        fo=str(directory),
        asynchronous=True,
        remote_protocol=source.protocol,
        remote_options=remote_options,
    ).get_mapper("")
    with xr.open_zarr(
        mapper, consolidated=False, chunks=None, decode_times=False
    ) as actual:
        if dict(actual.sizes) != dict(expected.sizes) or set(actual.variables) != set(
            expected.variables
        ):
            raise ValueError(
                "Parquet round-trip changed dimensions or variable inventory"
            )


def reference_provenance(
    source: SourceArchive,
    snapshot: list[dict],
    datasets: Sequence[xr.Dataset],
    combined: xr.Dataset,
    concat_dim: str | None,
    record_size: int,
) -> dict:
    """Describe the inputs, environment, and reference-store construction."""
    return {
        "source_files": snapshot,
        "source_endpoint": source.endpoint_url,
        "pattern": source.pattern,
        "static": concat_dim is None,
        "record_size": record_size,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "virtualizarr",
                "kerchunk",
                "xarray",
                "zarr",
                "fsspec",
                "s3fs",
                "obstore",
                "obspec-utils",
                "fastparquet",
            )
        },
        "dimensions": dict(combined.sizes),
        "variables": {
            name: {
                "dimensions": list(variable.dims),
                "dtype": str(variable.dtype),
                "attributes": variable.attrs,
            }
            for name, variable in combined.variables.items()
        },
        "source_attributes": [dataset.attrs for dataset in datasets],
        "validation": "metadata round-trip and unchanged selected source objects; selection frozen at discovery; no full payload scan",
    }


def build(
    source_root: str,
    output: str | Path,
    *,
    pattern: str = "[0-9]*.ocean_5daily.nc",
    expected_files: int | None = None,
    endpoint_url: str = OSN,
    anonymous: bool = True,
    static: bool = False,
    record_size: int = 100_000,
) -> Path:
    """Combine NetCDF files into a local Kerchunk Parquet reference store.

    Preserve source time values in filename order; consumers validate chronology.
    Native chunk shapes must agree and non-final files must end on chunk boundaries.
    Unwritten native chunks must be materialized before virtualization, even
    with a declared _FillValue; sparse Parquet references are not supported.

    Args:
        source_root: Local source directory or s3:// bucket/prefix. Selection is
            frozen at discovery; files added later are excluded.
        output: New local directory for the reference store.
        pattern: Glob selecting source filenames, sorted before concatenation.
        expected_files: Optional expected number of matching files.
        endpoint_url: S3-compatible endpoint for source reads.
        anonymous: Read public sources without credentials. Use --anonymous=False
            for authenticated reads.
        static: Index one static/grid file without concatenating along time.
        record_size: Number of references per Parquet partition.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if record_size < 1 or (expected_files is not None and expected_files < 1):
        raise ValueError("record_size and expected_files must be positive")
    source = SourceArchive.discover(source_root, pattern, endpoint_url, anonymous)
    if expected_files is not None and len(source.urls) != expected_files:
        raise ValueError(
            f"Expected {expected_files} source files, found {len(source.urls)}"
        )
    if static and len(source.urls) != 1:
        raise ValueError("--static requires exactly one source file")
    snapshot = source.snapshot()
    concat_dim = None if static else "time"
    datasets = []
    for url in source.urls:
        LOG.info("Parsing %s", url)
        datasets.append(open_virtual_netcdf(url, source.registry, concat_dim))
    combined = combine_virtual_datasets(datasets, concat_dim, source.urls)
    provenance = reference_provenance(
        source, snapshot, datasets, combined, concat_dim, record_size
    )
    with staged_reference_directory(output) as directory:
        write_parquet_references(combined, directory, record_size, source)
        validate_parquet_references(directory, combined, source)
        if source.snapshot() != snapshot:
            raise ValueError("Source objects changed while building references")
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )
    LOG.info("Wrote %s (%d sources)", output, len(source.urls))
    return output


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    fire.Fire(build, name="ocean_preprocessing.virtualize_om4", serialize=str)


if __name__ == "__main__":
    main()
