# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Archived rollout access, physical fields, and linked timestamps."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native import DiskCache
from rollouts import ArchiveStore
from server import create_app


class ArchiveTests(unittest.TestCase):
    def test_authenticated_reads_are_cached_and_missing_chunks_fail(self):
        payloads = {
            ".zmetadata": json.dumps(
                {"metadata": {".zgroup": {"zarr_format": 2}}}
            ).encode(),
            "thetao_0/0.0.0": b"data",
        }

        def read(args, **kwargs):
            key = args[2].removeprefix("remote:bucket/predictions.zarr/")
            return SimpleNamespace(
                returncode=0 if key in payloads else 1,
                stdout=payloads.get(key, b""),
                stderr=b"private auth diagnostic",
            )

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("rollouts.subprocess.run", side_effect=read) as command,
        ):
            store = ArchiveStore("remote:bucket/predictions.zarr", DiskCache(temp))
            self.assertEqual(store["thetao_0/0.0.0"], b"data")
            self.assertEqual(store["thetao_0/0.0.0"], b"data")
            self.assertEqual(command.call_count, 2)
            self.assertEqual(json.loads(store[".zgroup"]), {"zarr_format": 2})
            with self.assertRaises(KeyError):
                store["../secret"]
            with self.assertRaises(KeyError):
                store["absent/.zattrs"]
            with self.assertRaisesRegex(OSError, "Could not read OSN object"):
                store["thetao_0/1.0.0"]


class RolloutApiTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        values = np.arange(24, dtype="f4").reshape(2, 3, 4)
        # The historical rollout writer emitted lat/lon dimensions and omitted units.
        truth = xr.Dataset(
            {
                "thetao_0": (("time", "lat", "lon"), values),
                "mask_0": (
                    ("lat", "lon"),
                    np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool),
                ),
            },
            attrs={"thetao": {"units": "degC"}},
            coords={
                "time": np.array(
                    ["2014-10-15T12:00", "2014-10-20T12:00"], dtype="datetime64[ns]"
                ),
                "lat": [-60.0, 0.0, 60.0],
                "lon": [0.5, 90.5, 180.5, 270.5],
            },
        )
        truth.to_zarr(root / "truth", consolidated=True)
        predictions = truth[["thetao_0"]].isel(time=[1]).copy()
        predictions.thetao_0.attrs = {}
        predictions.thetao_0.values += 2
        predictions.to_zarr(root / "predictions", consolidated=True)
        with patch(
            "rollouts.ArchiveStore",
            side_effect=lambda path, cache: (
                root / ("truth" if path.endswith("OM4.zarr") else "predictions")
            ),
        ):
            return create_app(
                [str(root / "truth")], rollouts=True, cache_dir=root / "cache"
            )

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temp.cleanup()

    async def test_official_rollouts_metadata_and_reference_ranges(self):
        response = await self.client.get("/api/metadata")
        datasets = (await response.json())["datasets"]
        self.assertEqual(len(datasets), 4)
        rollout = datasets[1]
        self.assertTrue(rollout["rollout"])
        self.assertTrue(rollout["source_link"].endswith("/uw49dzuq"))
        self.assertEqual(rollout["times"], ["2014-10-20 12:00:00"])
        self.assertEqual(rollout["interval_end_times"], ["2014-10-23 00:00:00"])
        level = rollout["fields"][0]["levels"][0]
        self.assertEqual(level["units"], "degC")
        self.assertEqual(level["sampling"], "forecast · five-day mean")
        self.assertEqual(
            set(rollout["times"]) & set(datasets[0]["times"]), set(rollout["times"])
        )
        truth = await (
            await self.client.get("/api/limits?dataset=0&field=thetao_0")
        ).json()
        prediction = await (
            await self.client.get("/api/limits?dataset=1&field=thetao_0")
        ).json()
        self.assertEqual(truth["min"], prediction["min"])
        self.assertEqual(truth["max"], prediction["max"])
        response = await self.client.get("/api/frame?dataset=1&field=thetao_0&time=0")
        frame = np.frombuffer(await response.read(), dtype="<f4").reshape(3, 4)
        self.assertEqual(frame[0, 0], 24)
        self.assertTrue(np.isneginf(frame[2, 2]))
