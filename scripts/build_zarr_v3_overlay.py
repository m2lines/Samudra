#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr>=3,<4",
# ]
# ///
"""Build a root-only Zarr v3 overlay on top of an existing local Zarr v2 store.

The output store contains:
- a single root ``zarr.json`` with inline consolidated metadata for the full tree
- symlinks to each top-level child directory from the source store
- v3 ``dimension_names`` translated from xarray's v2 ``_ARRAY_DIMENSIONS`` attrs

No chunk payloads are copied or rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.dtype.common import HasEndianness
from zarr.core.group import ArrayV2Metadata, ArrayV3Metadata, Group, GroupMetadata
from zarr.metadata.migrate_v3 import _convert_compressor, _convert_filters


@dataclass
class FillValueRewrite:
    """Tracks lossy v2 -> v3 fill value rewrites."""

    path: str
    dtype: str
    source_fill_value: Any
    replacement_fill_value: Any
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the source Zarr v2 store")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output path for the overlay store (default: ./<source>.v3.overlay.zarr)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory",
    )
    return parser.parse_args()


def main() -> int:
    """Build the overlay store."""
    args = parse_args()
    source = args.source.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else Path.cwd() / f"{source.name}.v3.overlay.zarr"
    )

    validate_paths(source, output, force=args.force)

    root = zarr.open_group(store=str(source), mode="r", zarr_format=2)
    rewrites: list[FillValueRewrite] = []

    with tempfile.TemporaryDirectory(prefix="zarr-v3-overlay-meta-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        write_metadata_tree(root, tmpdir, rewrites)
        zarr.consolidate_metadata(tmpdir, zarr_format=3)
        materialize_overlay(source, output, tmpdir / "zarr.json", root)

    print(f"Created overlay: {output}")
    print(f"Source store: {source}")
    print(f"Top-level symlinks: {len(tuple(root.keys()))}")
    if rewrites:
        print("Fill value rewrites:")
        for rewrite in rewrites:
            payload = asdict(rewrite)
            print(json.dumps(jsonable(payload), sort_keys=True))

    return 0


def validate_paths(source: Path, output: Path, *, force: bool) -> None:
    """Validate source/output paths before writing."""
    if not source.exists():
        raise FileNotFoundError(f"Source store does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source store is not a directory: {source}")
    if output.exists():
        if not force:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --force to replace it."
            )
        remove_existing_path(output)


def write_metadata_tree(
    root: Group, metadata_root: Path, rewrites: list[FillValueRewrite]
) -> None:
    """Write a metadata-only Zarr v3 tree that mirrors the source hierarchy."""
    write_group_metadata(metadata_root, root.metadata.attributes)

    for rel_path, node in root.members(max_depth=None):
        node_path = metadata_root / rel_path
        if isinstance(node, Group):
            write_group_metadata(node_path, node.metadata.attributes)
            continue

        metadata_v2 = cast(ArrayV2Metadata, node.metadata)
        metadata_v3 = convert_array_metadata(rel_path, metadata_v2, rewrites)
        write_json(node_path / "zarr.json", metadata_v3.to_dict())


def convert_array_metadata(
    rel_path: str, metadata_v2: ArrayV2Metadata, rewrites: list[FillValueRewrite]
) -> ArrayV3Metadata:
    """Translate one v2 array metadata document to v3."""
    codecs: list[Any] = []
    dimension_names = extract_dimension_names(metadata_v2.attributes)

    if metadata_v2.order == "F":
        codecs.append(
            TransposeCodec(order=list(range(len(metadata_v2.shape) - 1, -1, -1)))
        )

    if metadata_v2.filters:
        codecs.extend(_convert_filters(tuple(metadata_v2.filters)))

    if isinstance(metadata_v2.dtype, HasEndianness):
        codecs.append(BytesCodec(endian=metadata_v2.dtype.endianness))
    else:
        codecs.append(BytesCodec(endian=None))

    if metadata_v2.compressor is not None:
        codecs.append(_convert_compressor(metadata_v2.compressor, metadata_v2.dtype))

    fill_value: Any = metadata_v2.fill_value
    if fill_value is None:
        fill_value = metadata_v2.dtype.default_scalar()
        rewrites.append(
            FillValueRewrite(
                path=rel_path,
                dtype=str(metadata_v2.dtype),
                source_fill_value=None,
                replacement_fill_value=fill_value,
                reason=(
                    "Zarr v3 metadata requires a dtype-compatible fill value here; "
                    "the v2 metadata stored null."
                ),
            )
        )

    return ArrayV3Metadata(
        shape=metadata_v2.shape,
        data_type=metadata_v2.dtype,
        chunk_grid=metadata_v2.chunk_grid,
        chunk_key_encoding=V2ChunkKeyEncoding(
            separator=metadata_v2.dimension_separator
        ),
        fill_value=fill_value,
        codecs=codecs,
        attributes=metadata_v2.attributes,
        dimension_names=dimension_names,
        storage_transformers=None,
    )


def extract_dimension_names(attributes: dict[str, Any]) -> tuple[str, ...] | None:
    """Translate xarray's v2 dimension metadata into the v3 field."""
    raw_dimension_names = attributes.get("_ARRAY_DIMENSIONS")
    if raw_dimension_names is None:
        return None
    if not isinstance(raw_dimension_names, list | tuple):
        raise TypeError(
            "_ARRAY_DIMENSIONS must be a list or tuple of strings; "
            f"got {type(raw_dimension_names).__name__}"
        )
    if not all(isinstance(name, str) for name in raw_dimension_names):
        raise TypeError("_ARRAY_DIMENSIONS must contain only strings")
    return tuple(raw_dimension_names)


def materialize_overlay(
    source: Path, output: Path, root_zarr_json: Path, root: Group
) -> None:
    """Create the final overlay directory with root metadata and symlinks."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root_zarr_json, output / "zarr.json")

    for name in root.keys():
        src_child = source / name
        dst_child = output / name
        if not src_child.exists():
            raise FileNotFoundError(
                f"Top-level child is missing from source store: {src_child}"
            )
        if not src_child.is_dir():
            raise ValueError(
                f"Top-level child {src_child} is not a directory. Root-array overlays are not supported."
            )
        os.symlink(src_child, dst_child, target_is_directory=True)


def write_group_metadata(path: Path, attributes: dict[str, Any]) -> None:
    """Write a group zarr.json file."""
    metadata = GroupMetadata(attributes=attributes, zarr_format=3)
    write_json(path / "zarr.json", metadata.to_dict())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def remove_existing_path(path: Path) -> None:
    """Remove an existing file, symlink, or directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def jsonable(value: Any) -> Any:
    """Convert values to plain JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return repr(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
