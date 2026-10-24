# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run with python -m unittest discover -s apps/om4_viewer/tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native import SourceReadError
from server import Dataset, create_app


def write_store(path):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    values[0, 2, 2] = 0  # Zero is a valid ocean value.
    values[0, 2, 3] = np.nan  # Missing in a wet cell.
    mask = np.ones((3, 4), dtype=bool)
    mask[2, 0] = False
    xr.Dataset(
        {
            "thetao_0": (("time", "y", "x"), values, {"units": "degC"}),
            "thetao_18": (("time", "y", "x"), values + 1),
            "zos": (("x", "time", "y"), values.transpose(2, 0, 1)),
            "mask_0": (("y", "x"), mask),
            "mask_18": (("y", "x"), mask),
        },
        coords={
            "time": np.array(["2000-01-01", "2000-01-06"], dtype="datetime64[ns]"),
            "y": [-60.0, 0.0, 60.0],
            "x": [0.5, 90.5, 180.5, 270.5],
        },
    ).to_zarr(path, consolidated=True)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "OM4.zarr"
        write_store(self.path)
        self.dataset = Dataset(str(self.path.parent))

    def tearDown(self):
        self.dataset.close()
        self.temp.cleanup()

    def test_geography_and_wet_zero_missing_land_are_preserved(self):
        d = self.dataset
        np.testing.assert_array_equal(d.lon, [-179.5, -89.5, 0.5, 90.5])
        np.testing.assert_array_equal(d.lat, [60, 0, -60])
        frame = d.frame("thetao_0", 0)
        self.assertEqual(frame[0, 0], 0)
        self.assertTrue(np.isnan(frame[0, 1]))
        self.assertTrue(np.isneginf(frame[0, 2]))
        self.assertEqual(frame[0, 3], 9)
        self.assertFalse(frame.flags.writeable)

    def test_dimension_order_does_not_change_geography(self):
        np.testing.assert_array_equal(
            self.dataset.frame("thetao_0", 0), self.dataset.frame("zos", 0)
        )

    def test_depths_metadata_and_range_use_valid_values(self):
        metadata = self.dataset.metadata()
        temperature = next(f for f in metadata["fields"] if f["id"] == "thetao")
        self.assertEqual(
            [level["label"] for level in temperature["levels"]], ["2.5 m", "6000 m"]
        )
        self.assertEqual(
            metadata["times"], ["2000-01-01 00:00:00", "2000-01-06 00:00:00"]
        )
        limits = self.dataset.limits("thetao_0")
        self.assertTrue(np.isfinite(limits["min"]))
        self.assertLess(limits["min"], limits["max"])
        self.assertGreaterEqual(limits["min"], 0)

    def test_bounds_and_field_validation(self):
        for key, index in [("absent", 0), ("thetao_0", -1), ("thetao_0", 2)]:
            with self.assertRaises(ValueError):
                self.dataset.frame(key, index)


class ApiTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "OM4.zarr"
        write_store(self.path)
        return create_app([str(self.path), str(self.path)])

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temp.cleanup()

    async def test_source_read_failure_returns_actionable_json(self):
        with patch.object(
            Dataset,
            "raster",
            side_effect=SourceReadError("OSN read failed (403): source object"),
        ):
            with self.assertLogs("server", level="ERROR"):
                response = await self.client.get(
                    "/api/map?dataset=0&field=thetao_0&time=0"
                )
            self.assertEqual(response.status, 502)
            self.assertEqual(
                (await response.json())["error"], "OSN read failed (403): source object"
            )

    async def test_frame_and_second_dataset(self):
        response = await self.client.get("/api/metadata")
        data = await response.json()
        self.assertEqual(len(data["datasets"]), 2)
        response = await self.client.get("/api/frame?dataset=1&field=thetao_0&time=0")
        self.assertEqual(response.status, 200)
        values = np.frombuffer(await response.read(), dtype="<f4").reshape(3, 4)
        self.assertEqual(values[0, 0], 0)
        self.assertTrue(np.isnan(values[0, 1]))
        self.assertTrue(np.isneginf(values[0, 2]))

    async def test_map_samples_the_same_cells_at_any_zoom(self):
        response = await self.client.get(
            "/api/map?field=thetao_0&time=0&bounds=-180,45,-140,75&width=16&height=16"
        )
        self.assertEqual(response.status, 200)
        values = np.frombuffer(await response.read(), dtype="<f4")
        np.testing.assert_array_equal(values, 0)
        response = await self.client.get("/api/map?field=zos&time=0&bounds=0,0,9999,90")
        self.assertEqual(response.status, 400)

    async def test_bad_requests_and_static_path_are_not_served(self):
        for query in [
            "dataset=-1&field=zos&time=0",
            "field=bad&time=0",
            "field=zos&time=-1",
            "field=zos&time=bad",
            "field=zos",
            "dataset=2&field=zos&time=0",
        ]:
            response = await self.client.get("/api/frame?" + query)
            self.assertEqual(response.status, 400, query)
            self.assertIn("error", await response.json())
        response = await self.client.get("/server.py")
        self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
