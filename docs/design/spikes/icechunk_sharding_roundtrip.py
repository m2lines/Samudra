# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Confirm Python and Rust-backed readers can read a sharded Icechunk array.

Run with:

    uv run --isolated --with 'icechunk>=2.1,<3' \
        --with 'zarr>=3.1,<4' --with 'zarrista>=0.1,<1' --with numpy \
        python docs/design/spikes/icechunk_sharding_roundtrip.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]
from zarrista import AsyncArray  # type: ignore[import-not-found]


async def read_with_zarrista(session, source: np.ndarray) -> None:  # noqa: ANN001
    """Validate the Icechunk session through zarrs-backed Zarrista."""
    array = await AsyncArray.open(session, "/field")
    actual = (await array[:, :, 68:148, 68:148]).to_numpy()
    np.testing.assert_equal(actual, source[:, :, 68:148, 68:148])


def main() -> None:
    with tempfile.TemporaryDirectory() as path:
        repo = ic.Repository.create(ic.local_filesystem_storage(path))
        session = repo.writable_session("main")
        array = zarr.create_array(
            store=session.store,
            name="field",
            shape=(1, 4, 216, 216),
            chunks=(1, 4, 18, 18),
            shards=(1, 4, 72, 72),
            dtype="float32",
            compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
        )
        source = np.arange(array.size, dtype="float32").reshape(array.shape)
        array[:] = source
        snapshot = session.commit("sharded test")

        readonly = repo.readonly_session(snapshot_id=snapshot)
        actual = zarr.open_array(readonly.store, path="field", mode="r")[
            :, :, 68:148, 68:148
        ]
        np.testing.assert_equal(actual, source[:, :, 68:148, 68:148])
        asyncio.run(read_with_zarrista(readonly, source))

        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
        ]
        print(f"icechunk={ic.__version__}")
        print(f"snapshot={snapshot}")
        print(f"roundtrip_shape={actual.shape}")
        print("zarrista_icechunk_read=OK")
        print(f"repository_files={len(files)}")
        print(
            f"repository_bytes={sum(os.path.getsize(file_path) for file_path in files)}"
        )


if __name__ == "__main__":
    main()
