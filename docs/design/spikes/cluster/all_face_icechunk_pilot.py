# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build a sparse all-face, all-in-scope-array LLC archive pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]
from schema_bakeoff import CODEC_CANDIDATES, codec, tree_size

VOLUME = ("U", "V", "Theta", "Salt")
SURFACE = ("Eta", "oceQnet", "oceTAUX", "oceTAUY")
GEOMETRY = ("XC", "YC")
MASKS = ("mask_c", "mask_w", "mask_s")
SMALL = (
    "time",
    "k",
    "k_l",
    "k_u",
    "k_p1",
    "face",
    "j",
    "i",
    "j_g",
    "i_g",
    "Z",
    "Zl",
    "Zu",
    "Zp1",
    "drC",
    "drF",
)
CORNERS = (
    (0, 720, 0, 720),
    (0, 720, 3600, 4320),
    (3600, 4320, 0, 720),
    (3600, 4320, 3600, 4320),
)


def attrs(source: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.attrs.asdict().items()
        if key != "_ARRAY_DIMENSIONS"
    }


def create_like(
    root: Any,
    source: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    compressor: Any,
) -> Any:
    dimensions = tuple(source.attrs.asdict()["_ARRAY_DIMENSIONS"])
    kwargs = {
        "name": name,
        "shape": shape,
        "chunks": chunks,
        "dtype": source.dtype,
        "fill_value": source.fill_value,
        "dimension_names": dimensions,
        "attributes": attrs(source),
    }
    if shards is not None:
        kwargs["shards"] = shards
        kwargs["compressors"] = [compressor]
    return root.create_array(**kwargs)


def validate_corners(source: Any, target: Any, *, time_dependent: bool) -> None:
    for face in range(13):
        for j0, j1, i0, i1 in CORNERS:
            if time_dependent:
                np.testing.assert_equal(
                    target[:, :, face, j0:j1, i0:i1],
                    source[0:2, :, face, j0:j1, i0:i1],
                )
            else:
                np.testing.assert_equal(
                    target[:, face, j0:j1, i0:i1],
                    source[:, face, j0:j1, i0:i1],
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("codec_decision", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    decision = json.loads(args.codec_decision.read_text())["winner"]
    compressor = codec(CODEC_CANDIDATES[decision["candidate_index"]])
    zarr.config.set({"async.concurrency": 2})

    split_config = ic.ManifestSplittingConfig.from_dict(
        {
            ic.ManifestSplitCondition.AnyArray(): {
                ic.ManifestSplitDimCondition.DimensionName("time"): 2
            }
        }
    )
    config = ic.RepositoryConfig(
        manifest=ic.ManifestConfig(splitting=split_config),
    )
    storage = ic.local_filesystem_storage(str(args.output))
    repo = ic.Repository.create(storage, config=config)
    repo.save_config()
    session = repo.writable_session("main")
    root = zarr.create_group(store=session.store, zarr_format=3)
    source_root = zarr.open_group(str(args.source), mode="r")
    root.attrs.update(source_root.attrs.asdict())
    root.attrs.update(
        {
            "archive_role": "sparse all-face engineering pilot, not a release",
            "pilot_populated_regions": "four 720x720 corners per face",
            "pilot_source_times": [0, 1],
        }
    )
    started = time.perf_counter()
    targets: dict[str, Any] = {}

    for name in SMALL:
        source = source_root[name]
        values = np.asarray(source[0:2] if name == "time" else source[:])
        target = create_like(
            root,
            source,
            name,
            shape=values.shape,
            chunks=values.shape,
            shards=None,
            compressor=compressor,
        )
        target[:] = values

    for name in VOLUME:
        source = source_root[name]
        target = create_like(
            root,
            source,
            name,
            shape=(2, *source.shape[1:]),
            chunks=(1, 17, 1, 120, 120),
            shards=(2, 51, 1, 1440, 1440),
            compressor=compressor,
        )
        for face in range(13):
            for j0, j1, i0, i1 in CORNERS:
                target[:, :, face, j0:j1, i0:i1] = source[0:2, :, face, j0:j1, i0:i1]
        targets[name] = target

    for name in SURFACE:
        source = source_root[name]
        target = create_like(
            root,
            source,
            name,
            shape=(2, *source.shape[1:]),
            chunks=(1, 1, 360, 360),
            shards=(24, 1, 1080, 1080),
            compressor=compressor,
        )
        for time_index in range(2):
            # One source object contains all faces. Decode it once, then scatter.
            plane = np.asarray(source[time_index])
            for face in range(13):
                for j0, j1, i0, i1 in CORNERS:
                    target[time_index, face, j0:j1, i0:i1] = plane[face, j0:j1, i0:i1]
        targets[name] = target

    for name in (*GEOMETRY, *MASKS):
        source = source_root[name]
        is_mask = name in MASKS
        target = create_like(
            root,
            source,
            name,
            shape=source.shape,
            chunks=(17, 1, 120, 120) if is_mask else (1, 360, 360),
            shards=(51, 1, 1440, 1440) if is_mask else (1, 1080, 1080),
            compressor=compressor,
        )
        for face in range(13):
            for j0, j1, i0, i1 in CORNERS:
                if is_mask:
                    target[:, face, j0:j1, i0:i1] = source[:, face, j0:j1, i0:i1]
                else:
                    target[face, j0:j1, i0:i1] = source[face, j0:j1, i0:i1]
        targets[name] = target

    encode_seconds = time.perf_counter() - started
    snapshot = session.commit(
        "sparse all-face LLC pilot",
        metadata={
            "volume_logical_chunk": [1, 17, 1, 120, 120],
            "volume_physical_shard": [2, 51, 1, 1440, 1440],
            "surface_logical_chunk": [1, 1, 360, 360],
            "surface_physical_shard": [24, 1, 1080, 1080],
        },
    )

    pinned = repo.readonly_session(snapshot_id=snapshot)
    reopened = zarr.open_group(pinned.store, mode="r")
    for name in VOLUME:
        validate_corners(source_root[name], reopened[name], time_dependent=True)
    for name in SURFACE:
        for time_index in range(2):
            # Preserve the source's global-chunk access unit during validation.
            plane = np.asarray(source_root[name][time_index])
            for face in range(13):
                for j0, j1, i0, i1 in CORNERS:
                    np.testing.assert_equal(
                        reopened[name][time_index, face, j0:j1, i0:i1],
                        plane[face, j0:j1, i0:i1],
                    )
    for name in MASKS:
        validate_corners(source_root[name], reopened[name], time_dependent=False)
    for name in GEOMETRY:
        for face in range(13):
            for j0, j1, i0, i1 in CORNERS:
                np.testing.assert_equal(
                    reopened[name][face, j0:j1, i0:i1],
                    source_root[name][face, j0:j1, i0:i1],
                )

    dataset = xr.open_zarr(pinned.store, consolidated=False)
    expected_dims = {"time", "k", "face", "j", "i", "j_g", "i_g"}
    if not expected_dims.issubset(dataset.sizes):
        raise AssertionError(
            f"missing dimensions: {expected_dims - set(dataset.sizes)}"
        )
    physical_objects, physical_bytes = tree_size(args.output)
    result = {
        "icechunk": ic.__version__,
        "xarray": xr.__version__,
        "snapshot": str(snapshot),
        "arrays": sorted(dataset.data_vars),
        "dimensions": dict(dataset.sizes),
        "faces_validated": 13,
        "corners_per_face": 4,
        "source_times": [0, 1],
        "encode_seconds": encode_seconds,
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "exact_corner_validation": True,
        "staggered_dimensions_preserved": True,
        "pinned_xarray_open": True,
        "limitation": "Cross-face halo rotation is not validated by this sparse archive pilot.",
    }
    (args.output.parent / "all-face-pilot-result.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
