#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Inspect a Zarr v2 store for metadata-only Zarr v3 overlay feasibility.

This reads only metadata files from a local filesystem-backed Zarr v2 store and
emits a JSON report. The report is designed to answer one specific question:
can a separate local Zarr v3 store plausibly reuse the existing chunk payloads
via symlinks or another filesystem overlay, without rewriting chunk data?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LIKELY_CODEC_IDS = {
    "adler32",
    "asciitables",
    "bitround",
    "blosc",
    "bz2",
    "crc32",
    "crc32c",
    "delta",
    "fixedscaleoffset",
    "fletcher32",
    "gzip",
    "jenkins_lookup3",
    "json2",
    "lz4",
    "lzma",
    "msgpack2",
    "packbits",
    "pcodec",
    "quantize",
    "shuffle",
    "vlen-array",
    "vlen-bytes",
    "zfpy",
    "zlib",
    "zstd",
}


@dataclass
class NodeInfo:
    """Metadata for a single node in a Zarr hierarchy."""

    path: str
    kind: Literal["array", "group"] | None = None
    metadata: dict[str, Any] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="Path to a local Zarr v2 store")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2; use 0 for compact output)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the inspector."""
    args = parse_args()
    report = build_report(args.store.resolve())
    indent = None if args.indent == 0 else args.indent
    json.dump(report, sys.stdout, indent=indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def build_report(store_path: Path) -> dict[str, Any]:
    """Build a JSON-serializable inspection report."""
    if not store_path.exists():
        raise FileNotFoundError(f"Store does not exist: {store_path}")
    if not store_path.is_dir():
        raise NotADirectoryError(f"Store path is not a directory: {store_path}")

    metadata_source = "consolidated" if (store_path / ".zmetadata").exists() else "tree"
    nodes = (
        load_nodes_from_consolidated(store_path)
        if metadata_source == "consolidated"
        else scan_nodes_from_tree(store_path)
    )

    root = nodes.get("")
    if root is None or root.kind is None:
        raise ValueError(f"Could not determine root node metadata in {store_path}")

    arrays = [
        node_to_array_report(node) for node in nodes.values() if node.kind == "array"
    ]
    groups = [node for node in nodes.values() if node.kind == "group"]
    arrays.sort(key=lambda item: item["path"])
    groups.sort(key=lambda item: item.path)

    dimension_separators = sorted(
        {
            array["dimension_separator"]
            for array in arrays
            if array["dimension_separator"] is not None
        }
    )
    codec_counter = Counter()
    warnings: list[str] = []
    reasons: list[str] = []

    if root.kind != "group":
        reasons.append(
            "The store root is an array, so a v3 overlay is possible but awkward because "
            "root-array chunks live directly at the store root."
        )

    for array in arrays:
        path_label = array["path"] or "."
        if array["dtype"].endswith("O") or array["dtype"] == "object":
            reasons.append(
                f"{path_label}: object dtype requires special handling in v3."
            )
        if array["object_codec"] is not None:
            reasons.append(
                f"{path_label}: object codec {codec_name(array['object_codec'])!r} requires manual v3 codec mapping."
            )
        if array["order"] == "F":
            reasons.append(
                f"{path_label}: Fortran-order chunks require careful v3 codec translation."
            )

        if codec := array["compressor"]:
            codec_counter[codec_name(codec)] += 1
            maybe_add_unknown_codec_warning(warnings, path_label, codec, "compressor")
        for filter_codec in array["filters"]:
            codec_counter[codec_name(filter_codec)] += 1
            maybe_add_unknown_codec_warning(
                warnings, path_label, filter_codec, "filter"
            )
        if object_codec := array["object_codec"]:
            codec_counter[codec_name(object_codec)] += 1
            maybe_add_unknown_codec_warning(
                warnings, path_label, object_codec, "object codec"
            )

    if len(dimension_separators) > 1:
        warnings.append(
            "Arrays use mixed v2 dimension separators, so the v3 overlay must set "
            "chunk_key_encoding per array rather than assuming one global separator."
        )

    verdict = "likely_feasible" if not reasons else "needs_manual_review"
    next_checks = [
        "Generate v3 consolidated metadata at the new local store root only.",
        "For each array, map v2 dimension_separator to v3 chunk_key_encoding=name:v2.",
        "Translate v2 compressor/filters/object_codec to an equivalent v3 codecs pipeline.",
        "Test one representative array with the actual zarr v3/xarray reader you plan to use.",
    ]
    if root.kind != "group":
        next_checks.insert(
            0,
            "Confirm you really have a root-array store; xarray dataset stores are usually root groups.",
        )

    return {
        "store_path": str(store_path),
        "source_zarr_format": 2,
        "metadata_source": metadata_source,
        "root_kind": root.kind,
        "root_attrs_keys": sorted(root.attrs),
        "summary": {
            "array_count": len(arrays),
            "group_count": len(groups),
            "dimension_separators": dimension_separators,
            "codec_ids": dict(sorted(codec_counter.items())),
        },
        "feasibility": {
            "verdict": verdict,
            "reasons": dedupe(reasons),
            "warnings": dedupe(warnings),
            "next_checks": next_checks,
        },
        "groups": [group_to_report(node) for node in groups],
        "arrays": arrays,
    }


def load_nodes_from_consolidated(store_path: Path) -> dict[str, NodeInfo]:
    """Load node metadata from .zmetadata."""
    zmetadata = json.loads((store_path / ".zmetadata").read_text())
    metadata = zmetadata.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid .zmetadata in {store_path}: missing 'metadata' object"
        )

    nodes: dict[str, NodeInfo] = {}
    for key, value in metadata.items():
        path, suffix = split_metadata_key(key)
        node = nodes.setdefault(path, NodeInfo(path=path))
        if suffix == ".zgroup":
            node.kind = "group"
            node.metadata = expect_dict(value, f"{key} metadata")
        elif suffix == ".zarray":
            node.kind = "array"
            node.metadata = expect_dict(value, f"{key} metadata")
        elif suffix == ".zattrs":
            node.attrs = expect_dict(value, f"{key} attrs")

    return nodes


def split_metadata_key(key: str) -> tuple[str, str]:
    """Split a consolidated metadata key into logical node path and suffix."""
    for suffix in (".zgroup", ".zarray", ".zattrs"):
        if key == suffix:
            return "", suffix
        trailer = f"/{suffix}"
        if key.endswith(trailer):
            return key[: -len(trailer)], suffix
    raise ValueError(f"Unrecognized consolidated metadata key: {key}")


def scan_nodes_from_tree(store_path: Path) -> dict[str, NodeInfo]:
    """Discover node metadata by walking a filesystem store."""
    nodes: dict[str, NodeInfo] = {}
    stack = [("", store_path)]
    visited_dirs: set[Path] = set()

    while stack:
        rel_path, abs_path = stack.pop()
        real_path = abs_path.resolve()
        if real_path in visited_dirs:
            continue
        visited_dirs.add(real_path)

        with os.scandir(abs_path) as it:
            entries = {entry.name: entry for entry in it}

        if ".zarray" in entries:
            nodes[rel_path] = NodeInfo(
                path=rel_path,
                kind="array",
                metadata=read_json(abs_path / ".zarray"),
                attrs=read_json_if_exists(abs_path / ".zattrs"),
            )
            continue

        if ".zgroup" not in entries:
            raise ValueError(
                f"Directory is missing .zgroup/.zarray metadata: {abs_path}"
            )

        nodes[rel_path] = NodeInfo(
            path=rel_path,
            kind="group",
            metadata=read_json(abs_path / ".zgroup"),
            attrs=read_json_if_exists(abs_path / ".zattrs"),
        )

        for entry in entries.values():
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=True):
                child_rel = entry.name if not rel_path else f"{rel_path}/{entry.name}"
                stack.append((child_rel, Path(entry.path)))

    return nodes


def node_to_array_report(node: NodeInfo) -> dict[str, Any]:
    """Convert an array node to JSON-ready output."""
    if node.metadata is None:
        raise ValueError(f"Array node {node.path!r} is missing metadata")

    metadata = node.metadata
    return {
        "path": node.path,
        "shape": as_int_list(metadata.get("shape")),
        "chunks": as_int_list(metadata.get("chunks")),
        "dtype": str(metadata.get("dtype")),
        "fill_value": jsonable(metadata.get("fill_value")),
        "order": metadata.get("order"),
        "dimension_separator": metadata.get("dimension_separator", "."),
        "compressor": jsonable(metadata.get("compressor")),
        "filters": jsonable(metadata.get("filters") or []),
        "object_codec": jsonable(metadata.get("object_codec")),
        "attrs_keys": sorted(node.attrs),
    }


def group_to_report(node: NodeInfo) -> dict[str, Any]:
    """Convert a group node to JSON-ready output."""
    return {
        "path": node.path,
        "attrs_keys": sorted(node.attrs),
    }


def maybe_add_unknown_codec_warning(
    warnings: list[str], path_label: str, codec: dict[str, Any], codec_role: str
) -> None:
    """Add a warning when the codec identifier is absent or unusual."""
    name = codec_name(codec)
    if name == "<missing-id>":
        warnings.append(
            f"{path_label}: {codec_role} metadata is missing an 'id' field."
        )
    elif name not in LIKELY_CODEC_IDS:
        warnings.append(
            f"{path_label}: {codec_role} {name!r} is not in the known numcodecs-style set; "
            "runtime compatibility should be checked explicitly."
        )


def codec_name(codec: dict[str, Any] | None) -> str:
    """Return the codec id, if present."""
    if not codec:
        return "<none>"
    codec_id = codec.get("id")
    return str(codec_id) if codec_id is not None else "<missing-id>"


def expect_dict(value: Any, label: str) -> dict[str, Any]:
    """Assert a metadata object is a dict."""
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected {label} to be an object, got {type(value).__name__}"
        )
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    payload = json.loads(path.read_text())
    return expect_dict(payload, str(path))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Read JSON if present, otherwise return an empty dict."""
    return read_json(path) if path.exists() else {}


def as_int_list(value: Any) -> list[int] | None:
    """Normalize integer tuples/lists for JSON output."""
    if value is None:
        return None
    return [int(item) for item in value]


def dedupe(values: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def jsonable(value: Any) -> Any:
    """Convert metadata values into JSON-friendly equivalents."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
