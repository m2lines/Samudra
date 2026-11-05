# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Independent payload oracles and adversarial publication schedules for PR 959.

Each example owns its directory: shrinking never reuses a previous output.
All source data are tiny and local; no archive credentials or ocean model needed.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import fsspec
import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from ocean_preprocessing import virtualize_om4 as subject


@st.composite
def archives(draw, aligned=False):
    """Vary payload, file partitioning, native chunking, packing and backend."""
    engine = draw(st.sampled_from(["scipy", "h5netcdf"]))
    chunk = draw(st.integers(1, 4))
    if aligned:
        count = draw(st.integers(1, 3))
        lengths = (
            [chunk] * count
            if engine == "scipy"
            else draw(
                st.lists(
                    st.integers(1, 3).map(lambda n: n * chunk),
                    min_size=count,
                    max_size=count,
                )
            )
        )
    else:
        lengths = draw(st.lists(st.integers(1, 4), min_size=1, max_size=3))
    width = draw(st.integers(1, 4))
    values = draw(
        st.lists(
            st.integers(-100, 100),
            min_size=sum(lengths) * width,
            max_size=sum(lengths) * width,
        )
    )
    return (
        lengths,
        np.array(values, dtype="float32").reshape(-1, width),
        engine,
        chunk,
        draw(st.booleans()),
        draw(st.integers(1, 8)),
    )


def write_source(path, values, times, *, engine="h5netcdf", chunk=1, packed=False):
    ds = xr.Dataset(
        {"temperature": (("time", "x"), values)},
        coords={
            "time": (
                "time",
                times,
                {"units": "days since 1958-01-01", "calendar": "julian"},
            ),
            "x": np.arange(values.shape[1], dtype="float64"),
        },
    )
    encoding = {"temperature": {"_FillValue": -9999.0}}
    if engine == "h5netcdf":
        encoding["temperature"].update(
            chunksizes=(min(chunk, len(times)), values.shape[1]), zlib=True
        )
    if packed:
        encoding["temperature"].update(dtype="int16", scale_factor=0.5, add_offset=10.0)
    ds.to_netcdf(path, engine=engine, encoding=encoding)


def read_reference(path):
    mapper = fsspec.filesystem(
        "reference", fo=str(path), remote_protocol="file", asynchronous=True
    ).get_mapper("")
    with xr.open_zarr(
        mapper, consolidated=False, chunks=None, decode_times=False
    ) as ds:
        return ds.load()


def check_round_trip(case):
    lengths, values, engine, chunk, packed, record_size = case
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sources = root / "sources"
        sources.mkdir()
        expected = []
        start = 0
        for index, length in enumerate(lengths):
            path = sources / f"{index:04d}.nc"
            write_source(
                path,
                values[start : start + length],
                np.arange(start, start + length),
                engine=engine,
                chunk=chunk,
                packed=packed,
            )
            with xr.open_dataset(path, decode_times=False) as ds:
                expected.append(ds.load())
            start += length
        subject.build(
            str(sources), root / "output", pattern="*.nc", record_size=record_size
        )
        actual = read_reference(root / "output")
        expected = xr.concat(expected, dim="time")
        xr.testing.assert_identical(actual, expected)
        assert {name: var.dtype for name, var in actual.variables.items()} == {
            name: var.dtype for name, var in expected.variables.items()
        }


@given(archives(aligned=True))
def test_aligned_archive_preserves_every_value(case):
    """Native NetCDF reads are the oracle, including CF decoding and zero values."""
    check_round_trip(case)


@pytest.mark.contract_probe
@given(archives())
def test_arbitrary_native_file_partitions_are_supported(case):
    """Probe the advertised native-layout support, including unequal annual lengths.

    This is a capability probe: failures may require a documented precondition
    rather than payload rewriting, which would defeat zero-copy virtualization.
    """
    check_round_trip(case)


@given(st.lists(st.one_of(st.integers(-10, 10), st.none()), min_size=1, max_size=20))
def test_packing_and_missing_values_do_not_change_decoded_payload(values):
    """CF packing is a storage transformation, including at missing ocean cells."""
    expected = np.array(
        [np.nan if value is None else value for value in values], dtype="float32"
    ).reshape(-1, 1)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        decoded = []
        for packed in [False, True]:
            source = root / str(packed)
            source.mkdir()
            write_source(
                source / "0.nc", expected, np.arange(len(values)), packed=packed
            )
            output = root / f"output-{packed}"
            subject.build(str(source), output, pattern="*.nc")
            decoded.append(read_reference(output))
            np.testing.assert_array_equal(decoded[-1].temperature.values, expected)
        xr.testing.assert_equal(*decoded)


@given(
    st.lists(st.integers(-100, 100), min_size=1, max_size=10),
    st.sampled_from(["scipy", "h5netcdf"]),
)
def test_static_archive_preserves_scalars_and_zero_coordinates(values, engine):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ds = xr.Dataset(
            {
                "depth": ("x", np.array(values, dtype="float32")),
                "scalar": np.float64(42),
            },
            coords={"x": np.arange(len(values))},
        )
        ds.to_netcdf(root / "grid.nc", engine=engine)
        subject.build(str(root), root / "output", pattern="*.nc", static=True)
        with xr.open_dataset(root / "grid.nc", decode_times=False) as expected:
            xr.testing.assert_identical(
                read_reference(root / "output"), expected.load()
            )


@given(st.text(alphabet="abc 09%?#é", min_size=1, max_size=12))
def test_local_source_names_do_not_change_payload(suffix):
    """Renaming a local archive must preserve values, including URI escaping."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / f"archive-{suffix}"
        source.mkdir()
        path = source / f"data-{suffix}.nc"
        write_source(path, np.array([[0.0], [1.0]], dtype="float32"), [0, 1])
        subject.build(str(source), root / "output", pattern="*.nc")
        with xr.open_dataset(path, decode_times=False) as expected:
            xr.testing.assert_identical(
                read_reference(root / "output"), expected.load()
            )


@given(st.sampled_from(["grid", "attributes", "inventory"]), st.integers(1, 100))
def test_incompatible_sources_are_rejected(change, delta):
    """Changing a static grid, physical units or inventory must not silently align."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for index in range(2):
            ds = xr.Dataset(
                {"temperature": (("time", "x"), np.ones((1, 2), dtype="float32"))},
                coords={"time": [index], "x": [0, 1]},
            )
            if index == 1:
                if change == "grid":
                    ds = ds.assign_coords(x=ds.x + delta)
                elif change == "attributes":
                    ds.temperature.attrs["units"] = "changed"
                else:
                    ds["extra"] = ds.temperature * delta
            ds.to_netcdf(root / f"{index}.nc", engine="h5netcdf")
        with pytest.raises(ValueError):
            subject.build(str(root), root / "output", pattern="*.nc")
        assert not (root / "output").exists()


@pytest.mark.contract_probe
@given(
    st.lists(st.integers(0, 50), min_size=2, max_size=8).filter(
        lambda times: any(a >= b for a, b in zip(times, times[1:]))
    )
)
def test_ambiguous_time_axis_is_rejected(times):
    """A published chronological archive must not silently contain overlaps/reversals."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        write_source(root / "0.nc", np.ones((len(times), 1), dtype="float32"), times)
        with pytest.raises(ValueError):
            subject.build(str(root), root / "output", pattern="*.nc")
        assert not (root / "output").exists()


@pytest.mark.parametrize(
    "mutation",
    ["modify", "remove", pytest.param("add", marks=pytest.mark.contract_probe)],
)
@given(length=st.integers(1, 6))
def test_source_mutations_abort_publication(length, mutation):
    """Inject a source change after serialization, before the final source check.

    Modification/removal check the current contract; addition proposes the
    stronger, separately marked contract of stable glob membership.
    The real serializer, validator, and object-store metadata calls still execute.
    """
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        path = root / "0.nc"
        write_source(path, np.ones((length, 1), dtype="float32"), np.arange(length))
        validate = subject.validate_parquet_references

        def mutate(*args):
            validate(*args)
            if mutation == "add":
                write_source(root / "1.nc", np.ones((1, 1), dtype="float32"), [length])
            elif mutation == "modify":
                path.write_bytes(path.read_bytes() + b"changed")
            else:
                path.unlink()

        patch.setattr(subject, "validate_parquet_references", mutate)
        with pytest.raises((ValueError, FileNotFoundError)):
            subject.build(str(root), root / "output", pattern="*.nc")
        assert not (root / "output").exists()
        assert not list(root.glob(".output-*"))


class PublicationMachine(RuleBasedStateMachine):
    """Compare repeated publish/fail/retry/delete commands with a one-slot model."""

    def __init__(self):
        super().__init__()
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.committed = None

    @rule(payload=st.binary(max_size=40), abort=st.booleans())
    def publish(self, payload, abort):
        def attempt():
            with subject.staged_reference_directory(self.output) as staged:
                staged.mkdir()
                (staged / "payload").write_bytes(payload)
                if abort:
                    raise RuntimeError("injected failure")

        if self.committed is not None:
            with pytest.raises(FileExistsError):
                attempt()
        elif abort:
            with pytest.raises(RuntimeError):
                attempt()
        else:
            attempt()
            self.committed = payload

    @rule()
    def remove(self):
        if self.committed is not None:
            (self.output / "payload").unlink()
            self.output.rmdir()
            self.committed = None

    @invariant()
    def visible_output_matches_last_successful_commit(self):
        assert self.output.exists() == (self.committed is not None)
        if self.committed is not None:
            assert (self.output / "payload").read_bytes() == self.committed
        assert not list(self.root.glob(".output-*"))

    def teardown(self):
        self.temporary.cleanup()


TestPublicationMachine = PublicationMachine.TestCase


@given(st.sampled_from(["absent", "empty", "occupied"]), st.booleans())
def test_publication_schedule_never_clobbers_another_writer(competitor, abort):
    """Schedule a competing directory creation between existence check and rename."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "output"

        def publish():
            with subject.staged_reference_directory(output) as staged:
                staged.mkdir()
                (staged / "ours").write_text("complete")
                if competitor != "absent":
                    output.mkdir()
                if competitor == "occupied":
                    (output / "theirs").write_text("preserve")
                if abort:
                    raise RuntimeError("injected serializer failure")

        if abort:
            with pytest.raises(RuntimeError):
                publish()
        elif competitor != "absent":
            with pytest.raises(OSError):
                publish()
        else:
            publish()
            assert (output / "ours").read_text() == "complete"
        if competitor != "absent":
            assert output.is_dir()
            assert not (output / "ours").exists()
            if competitor == "occupied":
                assert (output / "theirs").read_text() == "preserve"
            else:
                assert list(output.iterdir()) == []
        assert not list(root.glob(".output-*"))
