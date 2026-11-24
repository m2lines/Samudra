# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CM4 physical samples, masks, native depths, and noleap time bounds."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cftime
import numpy as np
import xarray as xr
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import CM4_LOCATION, create_app


class CM4ApiTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        dates = [cftime.DatetimeNoLeap(152, 2, 24), cftime.DatetimeNoLeap(152, 3, 1)]
        starts = [cftime.DatetimeNoLeap(152, 2, 19), dates[0]]
        self.values = np.arange(48, dtype="f4").reshape(2, 2, 3, 4)
        self.values[1, 1, 2, 2] = 0
        self.values[1, 1, 2, 3] = np.nan
        wet = np.ones((2, 3, 4), dtype=bool)
        wet[1, 2, 0] = False
        source = xr.Dataset(
            {
                "thetao": (("time", "lev", "y", "x"), self.values, {"units": "degC"}),
                "wfo": (("time", "y", "x"), self.values[:, 0], {"units": "kg m-2 s-1"}),
                "wetmask": (("lev", "y", "x"), wet),
                "time_bnds": (("time", "nv"), np.array(list(zip(starts, dates)))),
            },
            coords={
                "time": dates,
                "lev": [2.5, 17.5],
                "y": [-60, 0, 60],
                "x": [0.5, 90.5, 180.5, 270.5],
            },
            attrs={"title": "CM4 piControl fixture"},
        )
        source.to_zarr(root / "CM4.zarr", consolidated=True)
        with patch("rollouts.ArchiveStore", return_value=root / "CM4.zarr"):
            return create_app([], cm4=True, cache_dir=root / "cache")

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temp.cleanup()

    async def test_native_calendar_bounds_and_depths(self):
        response = await self.client.get("/api/metadata")
        (m,) = (await response.json())["datasets"]
        self.assertEqual(m["path"], CM4_LOCATION)
        self.assertEqual(m["calendar"], "noleap")
        self.assertEqual(m["times"], ["0152-02-24 00:00:00", "0152-03-01 00:00:00"])
        self.assertEqual(m["interval_end_times"], m["times"])
        temperature = next(f for f in m["fields"] if f["id"] == "thetao")
        self.assertEqual(
            [level["label"] for level in temperature["levels"]], ["2.5 m", "17.5 m"]
        )
        self.assertEqual(temperature["levels"][1]["key"], "thetao_1")
        self.assertIn("wfo", [f["id"] for f in m["fields"]])

    async def test_unmodified_samples_depth_masks_zero_and_missing(self):
        response = await self.client.get("/api/frame?field=thetao_1&time=1")
        self.assertEqual(response.status, 200)
        frame = np.frombuffer(await response.read(), dtype="<f4").reshape(3, 4)
        expected = self.values[1, 1][np.ix_([2, 1, 0], [2, 3, 0, 1])].copy()
        expected[0, 2] = -np.inf
        np.testing.assert_array_equal(frame, expected)
        self.assertEqual(frame[0, 0], 0)
        response = await self.client.get("/api/limits?field=wfo")
        self.assertEqual((await response.json())["reference"], "selected CM4 source")
