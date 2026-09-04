# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build and validate a pinned real-data Xarray/Icechunk pilot snapshot."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]
from schema_bakeoff import CODEC_CANDIDATES, codec, tree_size
from timespace_bakeoff import TILES, UNION


def timed_xarray_read(
    data: xr.DataArray, fixture: np.ndarray, selection: dict
) -> float:  # noqa: ANN001
    started = time.perf_counter()
    actual = data.isel(selection).values
    expected = fixture[
        selection["time"],
        selection["k"],
        selection["j"],
        selection["i"],
    ]
    np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("codec_decision", type=Path)
    parser.add_argument("logical_decision", type=Path)
    parser.add_argument("physical_decision", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    codec_decision = json.loads(args.codec_decision.read_text())["winner"]
    logical = json.loads(args.logical_decision.read_text())["winner"]
    physical = json.loads(args.physical_decision.read_text())["winner"]
    candidate = CODEC_CANDIDATES[codec_decision["candidate_index"]]
    fixture = np.load(args.fixture / "Theta.npy", mmap_mode="r")

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
    root.attrs.update(
        {
            "title": "Bounded LLC4320 sharding pilot",
            "source": "/orcd/data/abodner/003/LLC4320/LLC4320",
            "source_variable": "Theta",
            "source_face": 1,
            "source_j_start": 720,
            "source_i_start": 720,
            "archive_role": "engineering pilot, not a public release",
        }
    )
    data = root.create_array(
        "Theta",
        shape=fixture.shape,
        chunks=(1, logical["depth_inner"], logical["inner"], logical["inner"]),
        shards=(physical["time_shard"], 51, physical["outer"], physical["outer"]),
        dtype=fixture.dtype,
        compressors=[codec(candidate)],
        dimension_names=("time", "k", "j", "i"),
        attributes={"long_name": "potential temperature", "units": "degree_C"},
    )
    coordinates = {
        "time": np.arange(fixture.shape[0], dtype="int64"),
        "k": np.arange(51, dtype="int64"),
        "j": np.arange(720, 3600, dtype="int64"),
        "i": np.arange(720, 3600, dtype="int64"),
    }
    for name, values in coordinates.items():
        coordinate = root.create_array(
            name,
            shape=values.shape,
            chunks=values.shape,
            dtype=values.dtype,
            dimension_names=(name,),
        )
        coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]
        coordinate[:] = values
    started = time.perf_counter()
    data[:] = fixture
    encode_seconds = time.perf_counter() - started
    snapshot = session.commit(
        "bounded real-data Xarray pilot",
        metadata={
            "codec": candidate[0],
            "logical_chunk": [
                1,
                logical["depth_inner"],
                logical["inner"],
                logical["inner"],
            ],
            "physical_shard": [
                physical["time_shard"],
                51,
                physical["outer"],
                physical["outer"],
            ],
        },
    )

    pinned = repo.readonly_session(snapshot_id=snapshot)
    dataset = xr.open_zarr(pinned.store, consolidated=False)
    np.testing.assert_equal(dataset.Theta.values, fixture)
    tile_runs = []
    union_runs = []
    for repeat in range(3):
        time_index = repeat
        tile_runs.append(
            sum(
                timed_xarray_read(
                    dataset.Theta,
                    fixture,
                    {
                        "time": time_index,
                        "k": slice(None),
                        "j": slice(tile[0], tile[1]),
                        "i": slice(tile[2], tile[3]),
                    },
                )
                for tile in TILES
            )
        )
        union_runs.append(
            timed_xarray_read(
                dataset.Theta,
                fixture,
                {
                    "time": time_index,
                    "k": slice(None),
                    "j": slice(UNION[0], UNION[1]),
                    "i": slice(UNION[2], UNION[3]),
                },
            )
        )
    point_series = timed_xarray_read(
        dataset.Theta,
        fixture,
        {
            "time": slice(None),
            "k": slice(20, 21),
            "j": slice(1000, 1001),
            "i": slice(1000, 1001),
        },
    )
    physical_objects, physical_bytes = tree_size(args.output)
    result = {
        "icechunk": ic.__version__,
        "xarray": xr.__version__,
        "snapshot": str(snapshot),
        "codec": candidate[0],
        "logical_chunk": [
            1,
            logical["depth_inner"],
            logical["inner"],
            logical["inner"],
        ],
        "physical_shard": [
            physical["time_shard"],
            51,
            physical["outer"],
            physical["outer"],
        ],
        "encode_seconds": encode_seconds,
        "tile_median_seconds": statistics.median(tile_runs),
        "union_median_seconds": statistics.median(union_runs),
        "point_series_seconds": point_series,
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "exact_full_array_validation": True,
        "pinned_snapshot_validation": True,
        "xarray_dimensions": dict(dataset.sizes),
    }
    (args.output.parent / "xarray-pilot-result.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
