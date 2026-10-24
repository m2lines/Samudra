# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from aiohttp.test_utils import AioHTTPTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import LocalRolloutDataset, create_app


class LocalRolloutTests(AioHTTPTestCase):
    async def get_application(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "gated.zarr"
        self.data = xr.Dataset(
            {
                "thetao_0": (
                    ("time", "y", "x"),
                    np.arange(8, dtype="f4").reshape(1, 2, 4),
                ),
                "mask_0": (
                    ("y", "x"),
                    [[False, True, True, True], [True, True, True, True]],
                ),
            },
            coords={
                "time": np.array(["2014-10-20T12:00"], dtype="datetime64[ns]"),
                "y": [-45, 45],
                "x": [0.5, 90.5, 180.5, 270.5],
            },
            attrs={
                "viewer_name": "Native gated EMA",
                "title": "Final epoch 70",
                "source_link": "https://example.test/run",
            },
        )
        self.data.to_zarr(self.path, consolidated=True)
        return create_app([], local_rollouts=[str(self.path)])

    async def asyncTearDown(self):
        await super().asyncTearDown()
        self.temp.cleanup()

    async def test_metadata_and_exact_masked_values(self):
        response = await self.client.get("/api/metadata")
        d = (await response.json())["datasets"][0]
        self.assertTrue(d["rollout"])
        self.assertEqual(d["name"], "Native gated EMA")
        self.assertEqual(d["source_link"], "https://example.test/run")
        self.assertEqual(d["interval_end_times"], ["2014-10-23 00:00:00"])
        response = await self.client.get("/api/frame?dataset=0&field=thetao_0&time=0")
        values = np.frombuffer(await response.read(), dtype="<f4").reshape(2, 4)
        np.testing.assert_array_equal(values, [[6, 7, 4, 5], [2, 3, -np.inf, 1]])

    async def test_missing_mask_rejected(self):
        path = Path(self.temp.name) / "invalid.zarr"
        self.data.drop_vars("mask_0").to_zarr(path, consolidated=True)
        with self.assertRaisesRegex(ValueError, "no ocean mask"):
            LocalRolloutDataset(path)

    async def test_preparation_preserves_provenance_after_appends(self):
        root = Path(self.temp.name) / "prepared"
        root.mkdir()
        truth = self.data.copy()
        source = xr.Dataset(
            coords={"time": truth.time, "y": truth.y, "x": truth.x, "lev": [2.5]}
        )
        for field in ("thetao", "so", "uo", "vo"):
            truth[f"{field}_0"] = truth.thetao_0.copy()
            truth[f"{field}_0"].attrs["units"] = "test units"
            source[field] = truth.thetao_0.expand_dims(lev=[2.5]).transpose(
                "time", "lev", "y", "x"
            )
        truth["zos"] = truth.thetao_0.copy()
        truth.zos.attrs["units"] = "m"
        source["zos"] = truth.zos
        lon, lat = np.meshgrid(truth.x.values, truth.y.values)
        source["lon"] = (("y", "x"), lon)
        source["lat"] = (("y", "x"), lat)
        source.to_zarr(root / "predictions.zarr", consolidated=True)
        truth.to_zarr(root / "truth.zarr", consolidated=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "prepare_gated_rollout.py"),
                "--root",
                str(root),
                "--truth",
                str(root / "truth.zarr"),
            ],
            check=True,
            capture_output=True,
        )
        prepared = LocalRolloutDataset(str(root / "viewer.zarr"))
        self.assertEqual(len(prepared.keys), 5)
        self.assertIn("native gated", prepared.name)
        self.assertIn("q5mz68j4", prepared.metadata()["source_link"])
        np.testing.assert_array_equal(
            prepared.data.thetao_0.values, truth.thetao_0.values
        )
        prepared.close()
