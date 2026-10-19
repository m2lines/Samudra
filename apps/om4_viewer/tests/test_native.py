# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native grid, vector rotation, and cache regression checks without OSN."""

import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import cftime
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native import DiskCache, NativeDataset, SourceReadError, time_labels


class CacheTests(unittest.TestCase):
    def test_published_403_uses_same_object_authenticated_and_caches_it(self):
        url = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily.zarr/uo/4147.0.0.0"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "native.urlopen",
                side_effect=HTTPError(url, 403, "Forbidden", Message(), None),
            ) as http,
            patch(
                "native.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout=b"identical object bytes"
                ),
            ) as command,
        ):
            cache = DiskCache(temp)
            self.assertEqual(cache.get(url), b"identical object bytes")
            self.assertEqual(cache.get(url), b"identical object bytes")
            self.assertEqual(http.call_count, 1)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(
                command.call_args.args[0][2],
                "nyu-osn-m2lines-pubs:m2lines-pubs/Samudra/raw/om4_5daily.zarr/uo/4147.0.0.0",
            )

    def test_authenticated_failure_is_sanitized_and_other_hosts_never_use_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = DiskCache(temp)
            for url in [
                "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily.zarr/uo/1.0.0.0",
                "https://example.test/m2lines-pubs/Samudra/raw/om4_5daily.zarr/uo/1.0.0.0",
            ]:
                with (
                    patch(
                        "native.urlopen",
                        side_effect=HTTPError(url, 403, "Forbidden", Message(), None),
                    ),
                    patch(
                        "native.subprocess.run",
                        return_value=SimpleNamespace(
                            returncode=1, stdout=b"", stderr=b"secret diagnostic"
                        ),
                    ) as command,
                ):
                    with self.assertRaises(SourceReadError) as error:
                        cache.get(url)
                    self.assertNotIn("secret diagnostic", str(error.exception))
                    self.assertEqual(
                        command.call_count,
                        1 if url.startswith("https://nyu1.osn.mghpcc.org/") else 0,
                    )

    def test_cache_hit_and_size_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = root / "one", root / "two"
            one.write_bytes(b"1234")
            two.write_bytes(b"5678")
            cache = DiskCache(root / "cache", max_bytes=5)
            self.assertEqual(cache.get(one.as_uri()), b"1234")
            one.unlink()
            self.assertEqual(cache.get(one.as_uri()), b"1234")
            self.assertEqual(cache.get(two.as_uri()), b"5678")
            self.assertLessEqual(sum(p.stat().st_size for p in cache.root.iterdir()), 5)

    def test_missing_data_chunk_is_an_error_not_a_fill_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = DiskCache(temp)
            with patch(
                "native.urlopen",
                side_effect=HTTPError(
                    "http://example/vo/0.0.0.0", 404, "Missing", Message(), None
                ),
            ):
                with self.assertRaises(OSError):
                    cache.get("http://example/vo/0.0.0.0")
                with self.assertRaises(KeyError):
                    cache.get("http://example/.zattrs")


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root = root
        tracer = np.ones((2, 2, 3, 4), dtype="f4")
        tracer[:, 1, 1, 1] = np.nan
        xr.Dataset(
            {
                "thetao": (("time", "lev", "yh", "xh"), tracer),
                "uo": (
                    ("time", "lev", "yh", "xq"),
                    np.full((2, 2, 3, 5), 2.0, dtype="f4"),
                ),
                "vo": (
                    ("time", "lev", "yq", "xh"),
                    np.full((2, 2, 4, 4), 3.0, dtype="f4"),
                ),
            },
            coords={
                "time": np.array(["2000-01-01", "2000-01-06"], dtype="datetime64[ns]"),
                "lev": [2.5, 250.0],
                "yh": [0, 1, 2],
                "xh": [0, 1, 2, 3],
                "xq": [0, 1, 2, 3, 4],
                "yq": [0, 1, 2, 3],
            },
        ).to_zarr(root / "raw", consolidated=True)
        lon, lat = np.meshgrid([-135.0, -45.0, 45.0, 135.0], [30.0, 0.0, -30.0])
        wet = np.ones((3, 4))
        wet[0, 0] = 0
        xr.Dataset(
            {
                "geolon": (("yh", "xh"), lon),
                "geolat": (("yh", "xh"), lat),
                "wet": (("yh", "xh"), wet),
            }
        ).to_zarr(root / "grid", consolidated=True)
        xr.Dataset({"angle_dx": (("nyp", "nxp"), np.full((7, 9), 90.0))}).to_zarr(
            root / "hgrid", consolidated=True
        )
        self.d = NativeDataset(
            DiskCache(root / "cache"),
            {
                "thetao": "Temperature",
                "uo": "Zonal velocity",
                "vo": "Meridional velocity",
            },
            url=(root / "raw").as_uri(),
            grid_url=(root / "grid").as_uri(),
            hgrid_url=(root / "hgrid").as_uri(),
        )

    def tearDown(self):
        self.d.close()
        self.temp.cleanup()

    def test_rotates_vectors_and_uses_depth_mask(self):
        u = self.d.frame("uo_0", 0)
        v = self.d.frame("vo_0", 0)
        self.assertAlmostEqual(float(u[1, 1]), -3.0)
        self.assertAlmostEqual(float(v[1, 1]), 2.0)
        self.assertTrue(np.isneginf(u[0, 0]))
        deep = self.d.frame("uo_1", 0)
        self.assertTrue(np.isneginf(deep[1, 1]))
        self.assertEqual(float(deep[1, 2]), -3.0)

    def test_geographic_lookup_uses_actual_grid_not_nominal_axes(self):
        image = self.d.raster("uo_0", 0, (-45.1, -0.1, -44.9, 0.1), 16, 16)
        np.testing.assert_allclose(image, -3.0)
        # Longitude wrapping addresses the very same native cell.
        wrapped = self.d.raster("uo_0", 0, (314.9, -0.1, 315.1, 0.1), 16, 16)
        np.testing.assert_array_equal(image, wrapped)

    def test_metadata_has_physical_depths_and_exact_dates(self):
        m = self.d.metadata()
        self.assertEqual(m["fields"][0]["levels"][1]["label"], "250 m")
        self.assertEqual(m["times"][1], "2000-01-06 00:00:00")
        self.assertEqual(m["grid"], "native")
        self.assertIn("source_run", m)

    def test_snapshots_use_z_l_and_reuse_time_chunks(self):
        with xr.open_zarr(self.root / "raw", chunks=None) as source:
            snapshots = source.load().rename({"lev": "z_l"})
        snapshots = snapshots.assign_coords(
            time=snapshots.time + np.timedelta64(60, "h")
        )
        for key in ["thetao", "uo", "vo"]:
            snapshots[key].attrs["cell_methods"] = "time: point"
        snapshots["hfds"] = (
            ("time", "yh", "xh"),
            np.ones((2, 3, 4)),
            {"cell_methods": "time: mean"},
        )
        snapshots["wfo"] = (
            ("time", "yh", "xh"),
            np.arange(24, dtype="f4").reshape(2, 3, 4) * 1e-5 - 1e-4,
            {"cell_methods": "time: mean", "units": "kg m-2 s-1"},
        )
        for variable in snapshots.variables.values():
            variable.encoding = {}
        encoding = {
            key: {
                "chunks": tuple(
                    1 if dim == "z_l" else snapshots.sizes[dim]
                    for dim in snapshots[key].dims
                )
            }
            for key in ["thetao", "uo", "vo", "hfds", "wfo"]
        }
        snapshots.to_zarr(self.root / "snapshots", consolidated=True, encoding=encoding)
        native = NativeDataset(
            DiskCache(self.root / "cache"),
            {"uo": "U", "hfds": "Heat flux", "wfo": "Freshwater flux into ocean"},
            url=(self.root / "snapshots").as_uri(),
            snapshot=True,
            geometry_source=self.d,
        )
        try:
            metadata = native.metadata()
            self.assertEqual(metadata["times"], metadata["interval_end_times"])
            self.assertEqual(metadata["times"][0], "2000-01-03 12:00:00")
            self.assertEqual(
                metadata["fields"][0]["levels"][0]["sampling"], "instantaneous"
            )
            self.assertEqual(
                metadata["fields"][1]["levels"][0]["sampling"], "five-day mean"
            )
            self.assertEqual(metadata["fields"][0]["levels"][1]["label"], "250 m")
            native.frame("uo_0", 0)
            native.frame("uo_0", 1)
            self.assertEqual(len(native.raw_cache), 2)
            self.assertTrue(
                all(chunk.shape[0:2] == (2, 1) for chunk in native.raw_cache.values())
            )
            self.assertIs(native._geometry(), self.d._geometry())
            flux = metadata["fields"][2]
            self.assertEqual(flux["levels"][0]["label"], "Surface")
            self.assertEqual(flux["levels"][0]["sampling"], "five-day mean")
            self.assertEqual(flux["levels"][0]["units"], "kg m-2 s-1")
            expected = snapshots.wfo.isel(time=0).values.copy()
            expected[0, 0] = -np.inf
            np.testing.assert_array_equal(native.frame("wfo", 0), expected)
            limits = native.limits("wfo")
            self.assertEqual(limits["palette"], "balance")
            self.assertEqual(limits["min"], -limits["max"])
            self.assertGreater(limits["max"], 0)
            self.assertLess(limits["max"], 1e-3)
        finally:
            native.close()

    def test_interval_offset_preserves_julian_leap_day(self):
        values = np.array([cftime.DatetimeJulian(1900, 2, 28, 12)])
        self.assertEqual(time_labels(values, 60), ["1900-03-02 00:00:00"])
        self.assertEqual(
            self.d.metadata()["interval_end_times"][0], "2000-01-03 12:00:00"
        )


if __name__ == "__main__":
    unittest.main()
