"""Tests for the opt-in Rust LLC loader.

Everything that does not need the compiled extension runs unconditionally --
notably the channel-order derivation, which is the contract that makes a Rust
batch equal to a CPU batch. The reader tests build small synthetic stores and
skip when `ocean_llc_loader` is not built.
"""

import json

import numpy as np
import pytest

from ocean_emulators.rust_data import (
    NativeStoreSpec,
    channel_selectors,
    is_available,
)

requires_extension = pytest.mark.skipif(
    not is_available(), reason="ocean_llc_loader is not built"
)

FACES = 3
SIZE = 8
LEVELS = 4
TIMES = 5
CHUNK = 4


def _write_v2_array(root, name, shape, chunks, dims, values, dtype="<f4"):
    """Write an uncompressed Zarr V2 array the way xarray would."""
    path = root / name
    path.mkdir(parents=True)
    (path / ".zarray").write_text(
        json.dumps(
            {
                "chunks": list(chunks),
                "compressor": None,
                "dimension_separator": "/",
                "dtype": dtype,
                "fill_value": "NaN" if dtype.endswith("f2") or dtype.endswith("f4") else None,
                "filters": None,
                "order": "C",
                "shape": list(shape),
                "zarr_format": 2,
            }
        )
    )
    (path / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": list(dims)}))
    values = np.ascontiguousarray(values, dtype=dtype)
    grid = [-(-s // c) for s, c in zip(shape, chunks)]
    for index in np.ndindex(*grid):
        window = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape)
        )
        block = np.zeros(chunks, dtype=dtype)
        chunk_values = values[window]
        block[tuple(slice(0, n) for n in chunk_values.shape)] = chunk_values
        key = path.joinpath(*(str(i) for i in index))
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_bytes(block.tobytes())


@pytest.fixture
def llc_store(tmp_path):
    """A tiny store shaped like the real one: 3D `(t,k,face,j,i)`, 2D `(t,face,j,i)`."""
    root = tmp_path / "llc.zarr"
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text("{}")

    rng = np.random.default_rng(0)
    arrays = {}
    for name, col_dim in (("Theta", "i"), ("U", "i_g"), ("V", "i")):
        values = rng.standard_normal((TIMES, LEVELS, FACES, SIZE, SIZE))
        dims = ["time", "k", "face", "j" if name != "V" else "j_g", col_dim]
        _write_v2_array(
            root, name, values.shape, (1, LEVELS, 1, CHUNK, CHUNK), dims, values
        )
        arrays[name] = values.astype("<f4")
    for name in ("Eta", "oceQnet"):
        values = rng.standard_normal((TIMES, FACES, SIZE, SIZE))
        _write_v2_array(
            root, name, values.shape, (1, FACES, SIZE, SIZE),
            ["time", "face", "j", "i"], values,
        )
        arrays[name] = values.astype("<f4")
    static = rng.standard_normal((FACES, SIZE, SIZE))
    _write_v2_array(root, "rA", static.shape, (1, CHUNK, CHUNK),
                    ["face", "j", "i"], static)
    arrays["rA"] = static.astype("<f4")
    return root, arrays


@pytest.fixture
def boundary_cache(tmp_path):
    """A tiny `llc-train-ready-v1-boundaryonly` cache: packed, float16, no face."""
    root = tmp_path / "boundary.zarr"
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(
        json.dumps(
            {
                "cache_format": "llc-train-ready-v1-boundaryonly",
                "boundary_channel_names_json": json.dumps(
                    ["oceTAUX", "oceTAUY", "oceQnet", "Eta"]
                ),
            }
        )
    )
    values = np.random.default_rng(7).standard_normal((TIMES, 4, SIZE, SIZE)).astype("<f2")
    _write_v2_array(root, "boundary", values.shape, (1, 4, SIZE, SIZE),
                    ["time", "boundary_channel", "y", "x"], values, dtype="<f2")
    return root, values


def _spec(root, **overrides):
    kwargs = dict(
        path=str(root), face=1, j_start=2, j_stop=6, i_start=4, i_stop=8, read_threads=2
    )
    kwargs.update(overrides)
    return NativeStoreSpec(**kwargs)


class TestChannelSelectors:
    def test_groups_by_variable_then_level(self, llc_store):
        root, _ = llc_store
        names = [f"{var}_{level}" for var in ("U", "Theta") for level in range(LEVELS)]
        names.append("Eta")
        assert channel_selectors(str(root), names) == [
            *[("U", level) for level in range(LEVELS)],
            *[("Theta", level) for level in range(LEVELS)],
            ("Eta", None),
        ]

    def test_surface_only_names_have_no_level(self, llc_store):
        root, _ = llc_store
        assert channel_selectors(str(root), ["oceQnet", "Eta"]) == [
            ("oceQnet", None),
            ("Eta", None),
        ]

    def test_level_order_follows_first_appearance(self, llc_store):
        """The xarray path applies one shared level list to every levelled var."""
        root, _ = llc_store
        assert channel_selectors(str(root), ["Theta_2", "Theta_0", "U_2", "U_0"]) == [
            ("Theta", 2), ("Theta", 0), ("U", 2), ("U", 0),
        ]

    def test_rejects_a_count_mismatch(self, llc_store):
        """A name whose base has no vertical axis cannot carry a level suffix."""
        root, _ = llc_store
        with pytest.raises(ValueError, match="derived"):
            channel_selectors(str(root), ["Eta_0", "Eta_1"])

    def test_packed_cache_indexes_the_channel_axis(self, boundary_cache):
        root, _ = boundary_cache
        assert channel_selectors(
            str(root), ["oceQnet", "Eta"], packed_prefix="boundary"
        ) == [("boundary", 2), ("boundary", 3)]

    def test_packed_cache_missing_channel_is_an_error(self, boundary_cache):
        root, _ = boundary_cache
        with pytest.raises(KeyError, match="missing requested channels"):
            channel_selectors(str(root), ["nope"], packed_prefix="boundary")


class TestChunkAlignment:
    """Alignment to the store's 720 grid is one chunk per read vs four."""

    def test_aligned_window(self):
        spec = NativeStoreSpec(path="/x", face=1, j_start=720, j_stop=1440,
                               i_start=2880, i_stop=3600)
        assert spec.is_chunk_aligned()

    def test_offset_window(self):
        spec = NativeStoreSpec(path="/x", face=1, j_start=736, j_stop=1456,
                               i_start=2896, i_stop=3616)
        assert not spec.is_chunk_aligned()

    def test_packed_caches_are_always_aligned(self):
        spec = NativeStoreSpec(path="/x", face=None, j_start=0, j_stop=1104,
                               i_start=0, i_stop=1104, packed_prefix="boundary")
        assert spec.is_chunk_aligned()


@requires_extension
class TestReader:
    def test_reads_the_configured_tile(self, llc_store):
        from ocean_emulators.rust_data import NativeLlcReader

        root, arrays = llc_store
        reader = NativeLlcReader(
            _spec(root),
            {
                "prognostic": [f"Theta_{level}" for level in range(LEVELS)] + ["Eta"],
                "boundary": ["oceQnet", "Eta"],
            },
        )
        times = [0, 3]
        prognostic = reader.read("prognostic", times).numpy()
        assert prognostic.shape == (2, LEVELS + 1, 4, 4)
        for position, time in enumerate(times):
            for level in range(LEVELS):
                np.testing.assert_array_equal(
                    prognostic[position, level], arrays["Theta"][time, level, 1, 2:6, 4:8]
                )
            np.testing.assert_array_equal(
                prognostic[position, LEVELS], arrays["Eta"][time, 1, 2:6, 4:8]
            )
        boundary = reader.read("boundary", times).numpy()
        assert boundary.shape == (2, 2, 4, 4)
        for position, time in enumerate(times):
            np.testing.assert_array_equal(
                boundary[position, 0], arrays["oceQnet"][time, 1, 2:6, 4:8]
            )
            np.testing.assert_array_equal(
                boundary[position, 1], arrays["Eta"][time, 1, 2:6, 4:8]
            )

    def test_staggered_dims_are_sliced_positionally(self, llc_store):
        """`U`/`V` live on `i_g`/`j_g`; the CPU path renames those to `i`/`j`."""
        from ocean_emulators.rust_data import NativeLlcReader

        root, arrays = llc_store
        reader = NativeLlcReader(_spec(root), {"prognostic": ["U_1", "V_1"]})
        values = reader.read("prognostic", [2]).numpy()
        np.testing.assert_array_equal(values[0, 0], arrays["U"][2, 1, 1, 2:6, 4:8])
        np.testing.assert_array_equal(values[0, 1], arrays["V"][2, 1, 1, 2:6, 4:8])

    @pytest.mark.parametrize("full_rows", ["0", "1"])
    def test_whole_row_reads_do_not_change_the_result(
        self, llc_store, monkeypatch, full_rows
    ):
        """The `i`-widening optimisation must be invisible in the output."""
        from ocean_emulators.rust_data import NativeLlcReader

        root, arrays = llc_store
        monkeypatch.setenv("OCEAN_RUST_LOADER_FULL_ROWS", full_rows)
        reader = NativeLlcReader(_spec(root), {"prognostic": ["Theta_0"]})
        np.testing.assert_array_equal(
            reader.read("prognostic", [1]).numpy()[0, 0],
            arrays["Theta"][1, 0, 1, 2:6, 4:8],
        )

    def test_plane_cache_returns_the_same_values(self, llc_store, monkeypatch):
        from ocean_emulators.rust_data import NativeLlcReader

        root, arrays = llc_store
        monkeypatch.setenv("OCEAN_RUST_LOADER_CACHE_MB", "8")
        reader = NativeLlcReader(_spec(root), {"boundary": ["Eta"]})
        first = reader.read("boundary", [4]).numpy()
        second = reader.read("boundary", [4]).numpy()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0, 0], arrays["Eta"][4, 1, 2:6, 4:8])

    def test_rejects_out_of_range_times(self, llc_store):
        from ocean_emulators.rust_data import NativeLlcReader

        root, _ = llc_store
        reader = NativeLlcReader(_spec(root), {"prognostic": ["Theta_0"]})
        with pytest.raises(RuntimeError, match="out of bounds"):
            reader.read("prognostic", [TIMES])

    def test_rejects_a_window_outside_the_store(self, llc_store):
        from ocean_emulators.rust_data import NativeLlcReader

        root, _ = llc_store
        reader = NativeLlcReader(
            _spec(root, i_stop=SIZE + 4), {"prognostic": ["Theta_0"]}
        )
        with pytest.raises(RuntimeError, match="does not fit"):
            reader.read("prognostic", [0])

    def test_read_static_returns_the_tile(self, llc_store):
        root, arrays = llc_store
        np.testing.assert_array_equal(
            _spec(root).read_static("rA"), arrays["rA"][1, 2:6, 4:8]
        )

    def test_spatial_features_are_absent_without_xc_yc(self, llc_store):
        """A store missing XC/YC degrades to no spatial channels, not a crash."""
        root, _ = llc_store
        assert _spec(root).spatial_features() is None

    def test_boundary_cache_float16_widens_to_float32(self, boundary_cache):
        from ocean_emulators.rust_data import NativeLlcReader

        root, values = boundary_cache
        spec = NativeStoreSpec(path=str(root), face=None, j_start=0, j_stop=SIZE,
                               i_start=0, i_stop=SIZE, read_threads=2,
                               packed_prefix="boundary")
        out = NativeLlcReader(spec, {"boundary": ["oceQnet", "Eta"]}).read(
            "boundary", [1, 4]
        ).numpy()
        assert out.dtype == np.float32 and out.shape == (2, 2, SIZE, SIZE)
        for position, time in enumerate([1, 4]):
            np.testing.assert_array_equal(out[position, 0], values[time, 2].astype(np.float32))
            np.testing.assert_array_equal(out[position, 1], values[time, 3].astype(np.float32))


class TestTimeAlignment:
    """Two stores of the same simulation start at different timestamps."""

    def test_positions_are_matched_by_timestamp(self):
        from ocean_emulators.datasets import _align_times

        prognostic = np.arange("2011-09-13", "2011-09-17", dtype="datetime64[D]")
        boundary = np.arange("2011-09-10", "2011-09-20", dtype="datetime64[D]")
        np.testing.assert_array_equal(
            _align_times(prognostic, boundary, name="b"), [3, 4, 5, 6]
        )

    def test_a_gap_is_a_hard_error_not_a_nearest_match(self):
        from ocean_emulators.datasets import _align_times

        prognostic = np.arange("2011-09-13", "2011-09-17", dtype="datetime64[D]")
        boundary = np.arange("2011-09-13", "2011-09-15", dtype="datetime64[D]")
        with pytest.raises(ValueError, match="does not cover timestamp"):
            _align_times(prognostic, boundary, name="short-cache")

    def test_contiguous_positions_stay_a_slice(self):
        from ocean_emulators.datasets import _contiguous_indexer

        assert _contiguous_indexer(np.array([5, 6, 7])) == slice(5, 8)
        assert isinstance(_contiguous_indexer(np.array([5, 9])), np.ndarray)


class TestTileCatalogFromWindows:
    def test_windows_become_a_group_with_the_expected_overlaps(self):
        from ocean_emulators.tiling import build_group_layout, tile_catalog_from_windows

        # A 2x2 block of chunk-aligned 720 tiles, no overlap.
        windows = [
            (1, 2880, 3600, 720, 1440),
            (1, 3600, 4320, 720, 1440),
            (1, 2880, 3600, 1440, 2160),
            (1, 3600, 4320, 1440, 2160),
        ]
        catalog = tile_catalog_from_windows(windows)
        assert [tile.shape for tile in catalog] == [(720, 720)] * 4
        layout = build_group_layout(catalog)
        assert layout.num_tiles == 4
        assert layout.canonical_origin == (720, 2880)
        assert layout.canonical_shape == (1440, 1440)

    def test_rejects_a_malformed_window(self):
        from ocean_emulators.tiling import tile_catalog_from_windows

        with pytest.raises(ValueError, match="face, i_start"):
            tile_catalog_from_windows([(1, 0, 10, 0)])

    def test_rejects_an_empty_catalog(self):
        from ocean_emulators.tiling import tile_catalog_from_windows

        with pytest.raises(ValueError, match="at least one window"):
            tile_catalog_from_windows([])
