"""Test appending a later time window to an existing training-ready cache.

Append mutates a multi-hundred-gigabyte store in place, and a bad append is
worse than a failed build: the store still loads, still trains, and is quietly
wrong. These tests exist to make every way of getting that wrong a hard error.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_llc_patch_cache_compressed_train_val.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("_cache_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cache_builder"] = module
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def days(start: str, end: str) -> np.ndarray:
    return np.arange(start, end, dtype="datetime64[D]").astype("datetime64[ns]")


def make_store_dataset(
    times: np.ndarray,
    *,
    i_start: int = 0,
    mean: float = 0.0,
    dtype: str = "float16",
) -> xr.Dataset:
    """A minimal store in the packed training-ready layout."""
    n = len(times)
    return xr.Dataset(
        {
            "prognostic": (
                ("time", "prognostic_channel", "y", "x"),
                np.arange(n * 2 * 4 * 4, dtype=dtype).reshape(n, 2, 4, 4),
            ),
            "boundary": (
                ("time", "boundary_channel", "y", "x"),
                np.zeros((n, 1, 4, 4), dtype=dtype),
            ),
            "prognostic_mean": (("prognostic_channel",), np.array([mean, 1.0], dtype=dtype)),
            "prognostic_std": (("prognostic_channel",), np.ones(2, dtype=dtype)),
            "prognostic_mask": (
                ("prognostic_channel", "y", "x"),
                np.ones((2, 4, 4), dtype="int8"),
            ),
            "XC": (("y", "x"), np.zeros((4, 4), dtype="float32")),
        },
        coords={
            "time": times,
            "x": np.arange(i_start, i_start + 4, dtype="int16"),
            "y": np.arange(4, dtype="int16"),
            "prognostic_channel": ["a", "b"],
            "boundary_channel": ["f"],
        },
        attrs={
            "prognostic_channel_names_json": '["a", "b"]',
            "boundary_channel_names_json": '["f"]',
        },
    )


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "cache.zarr"
    make_store_dataset(days("2012-10-01", "2012-10-05")).to_zarr(
        path, mode="w", consolidated=True
    )
    return path


LATER = days("2012-10-05", "2012-10-09")


def test_append_extends_the_time_axis_and_leaves_static_arrays_alone(builder, store):
    builder.append_time_window(
        store, make_store_dataset(LATER), time_batch=2, label="test"
    )
    result = xr.open_zarr(store, consolidated=True)
    times = result["time"].to_numpy()

    assert times.size == 8
    assert np.all(np.diff(times) > np.timedelta64(0)), "time axis must stay sorted"
    assert result.attrs["test_time_count"] == 4
    assert result.attrs["test_start"] == "2012-10-05"
    assert result.attrs["test_end"] == "2012-10-08"
    # Masks, stats and grid arrays carry no time dim and must be untouched.
    assert result["prognostic_mask"].shape == (2, 4, 4)
    assert bool((result["prognostic_mask"].to_numpy() == 1).all())
    assert result["prognostic_mean"].shape == (2,)


def test_append_preserves_chunking_so_the_store_stays_readable(builder, store):
    before = xr.open_zarr(store, consolidated=True)["prognostic"].encoding["chunks"]
    builder.append_time_window(
        store, make_store_dataset(LATER), time_batch=2, label="test"
    )
    after = xr.open_zarr(store, consolidated=True)["prognostic"].encoding["chunks"]
    assert before == after


def test_append_is_idempotent(builder, store):
    """Re-running the same command must not duplicate timestamps."""
    for _ in range(2):
        builder.append_time_window(
            store, make_store_dataset(LATER), time_batch=2, label="test"
        )
    times = xr.open_zarr(store, consolidated=True)["time"].to_numpy()
    assert times.size == 8
    assert np.unique(times).size == 8


def test_partial_overlap_appends_only_the_new_tail(builder, store):
    """Not an error: dropping the already-present head is what makes a failed
    append safe to simply re-run with the same window."""
    overlapping = days("2012-10-03", "2012-10-07")  # 10-03/04 already stored
    builder.append_time_window(
        store, make_store_dataset(overlapping), time_batch=2, label="test"
    )
    result = xr.open_zarr(store, consolidated=True)
    assert result["time"].to_numpy().size == 6
    assert result.attrs["test_time_count"] == 2


def test_append_refuses_a_window_that_would_unsort_the_time_axis(builder, store):
    """Zarr appends by concatenation, not by merge, so an earlier window would
    silently leave the axis out of order rather than interleaving."""
    with pytest.raises(ValueError, match="concatenation"):
        builder.append_time_window(
            store,
            make_store_dataset(days("2012-09-01", "2012-09-03")),
            time_batch=2,
            label="test",
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"i_start": 100}, "x index coordinate differs"),
        ({"mean": 9.0}, "prognostic_mean differs"),
        ({"dtype": "float32"}, "dtype"),
    ],
)
def test_append_refuses_an_incompatible_window(builder, store, kwargs, match):
    """The appended window inherits the store's geometry, masks and
    normalization. If those would not have matched, the result trains wrong
    while looking fine, so it has to be refused up front."""
    with pytest.raises(ValueError, match=match):
        builder.append_time_window(
            store, make_store_dataset(LATER, **kwargs), time_batch=2, label="test"
        )


def test_append_refuses_a_mismatched_channel_layout(builder, store):
    incompatible = make_store_dataset(LATER)
    incompatible.attrs["prognostic_channel_names_json"] = '["a", "c"]'
    with pytest.raises(ValueError, match="channel_names_json differs"):
        builder.append_time_window(
            store, incompatible, time_batch=2, label="test"
        )


def test_appending_nothing_is_a_no_op(builder, store):
    already = make_store_dataset(days("2012-10-01", "2012-10-05"))
    builder.append_time_window(store, already, time_batch=2, label="test")
    result = xr.open_zarr(store, consolidated=True)
    assert result["time"].to_numpy().size == 4
    assert "test_time_count" not in result.attrs
