# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Independent payload oracles and adversarial publication schedules for PR 959.

Each example owns its directory: shrinking never reuses a previous output.
All source data are tiny and local; no archive credentials or ocean model needed.
"""

import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import fsspec
import h5py
import numpy as np
import pytest
import xarray as xr
from hypothesis import example, given
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


@st.composite
def om4_archives(draw):
    """Small native ocean grids with independent variable chunk grids and bounds."""
    nz, ny, nx = [draw(st.integers(1, 3)) for _ in range(3)]
    chunks = [draw(st.integers(1, 2)) for _ in range(2)]
    nt = math.lcm(*chunks) * draw(st.integers(1, 2))
    files = draw(st.integers(1, 3))
    count = files * nt * nz * ny * nx
    values = draw(st.lists(st.integers(-100, 100), min_size=count, max_size=count))
    return (
        np.array(values, dtype="float32").reshape(files * nt, nz, ny, nx),
        files,
        nt,
        chunks,
        draw(st.integers(1, nx)),
    )


def check_om4_round_trip(case, client=None):
    values, files, nt, chunks, xchunk = case
    _, nz, ny, nx = values.shape
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sources = root / "sources"
        sources.mkdir()
        expected, paths = [], []
        for index in range(files):
            block = values[index * nt : (index + 1) * nt]
            times = np.arange(index * nt, (index + 1) * nt)
            ds = xr.Dataset(
                {
                    "thetao": (("time", "z", "y", "x"), block),
                    "so": (("time", "z", "y", "x"), block + 200),
                    "hfds": (("time", "y", "x"), block[:, 0]),
                    "wet": (("y", "x"), np.ones((ny, nx), dtype="int8")),
                    "time_bnds": (
                        ("time", "nv"),
                        np.stack([times, times + 1], axis=-1),
                    ),
                    "z_bnds": (
                        ("z", "nv"),
                        np.stack([np.arange(nz), np.arange(nz) + 1], axis=-1),
                    ),
                },
                coords={
                    "time": (
                        "time",
                        times,
                        {
                            "units": "days since 1958-01-01",
                            "calendar": "julian",
                            "bounds": "time_bnds",
                        },
                    ),
                    "z": ("z", np.arange(nz), {"bounds": "z_bnds"}),
                    "y": np.arange(ny),
                    "x": np.arange(nx),
                    "nv": [0, 1],
                },
            )
            path = sources / f"{index:04d}.nc"
            ds.to_netcdf(
                path,
                engine="h5netcdf",
                encoding={
                    "thetao": {"chunksizes": (chunks[0], nz, 1, xchunk), "zlib": True},
                    "so": {"chunksizes": (chunks[1], 1, ny, nx), "zlib": True},
                    "hfds": {"chunksizes": (chunks[1], ny, 1), "zlib": True},
                },
            )
            paths.append(path)
            with xr.open_dataset(path, decode_times=False) as native:
                expected.append(native.load())
        output = root / "output"
        subject.build(str(sources), output, pattern="*.nc", record_size=3)
        if client:
            script = Path(__file__).parents[1] / "integration" / "zarr2_reader.py"
            subprocess.run(
                [client, str(script), str(output), *map(str, paths)],
                check=True,
                timeout=30,
            )
        else:
            actual = read_reference(output)
            xr.testing.assert_identical(
                actual,
                xr.concat(expected, dim="time", data_vars="minimal", coords="minimal"),
            )


@given(om4_archives())
def test_om4_depth_bounds_and_variable_chunks_round_trip(case):
    check_om4_round_trip(case)


@pytest.mark.skipif(
    not os.environ.get("SAMUDRA_ZARR2_PYTHON"),
    reason="Requires an isolated real Zarr 2 client",
)
@given(om4_archives())
def test_om4_payload_reads_with_zarr2_client(case):
    check_om4_round_trip(case, client=os.environ["SAMUDRA_ZARR2_PYTHON"])


@given(archives())
def test_native_file_partitions_round_trip_or_report_layout_error(case):
    """Unsupported native chunk boundaries fail clearly, without rewriting data."""
    lengths, _, engine, chunk, _, _ = case
    native_chunks = lengths if engine == "scipy" else [min(chunk, n) for n in lengths]
    compatible = len(set(native_chunks)) == 1 and all(
        length % native_chunks[0] == 0 for length in lengths[:-1]
    )
    try:
        check_round_trip(case)
    except ValueError as error:
        assert not compatible, str(error)
        assert "native chunk" in str(error).lower()
        assert "temperature" in str(error)
        assert ".nc" in str(error)
        assert "Rechunk" in str(error)


@st.composite
def partial_boundary_archives(draw):
    """Generate only layouts with matching chunks and an unaligned first boundary."""
    chunk = draw(st.integers(2, 4))
    lengths = [
        chunk * draw(st.integers(1, 3)) + draw(st.integers(1, chunk - 1)),
        chunk * draw(st.integers(1, 3)),
    ]
    width = draw(st.integers(1, 3))
    size = sum(lengths) * width
    values = draw(st.lists(st.integers(-100, 100), min_size=size, max_size=size))
    return (
        lengths,
        np.array(values, dtype="float32").reshape(-1, width),
        "h5netcdf",
        chunk,
        draw(st.booleans()),
        draw(st.integers(1, 8)),
    )


@given(partial_boundary_archives())
def test_nonfinal_partial_chunks_have_actionable_errors(case):
    """Every draw exercises the boundary, rather than hoping a broad draw hits it."""
    with pytest.raises(ValueError) as caught:
        check_round_trip(case)
    message = str(caught.value)
    assert "Partial native chunk" in message
    assert "temperature" in message and "0000.nc" in message
    assert "non-final" in message and "Rechunk" in message


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
    st.integers(-100, 100),
)
def test_static_archive_preserves_scalars_and_zero_coordinates(values, engine, scalar):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ds = xr.Dataset(
            {
                "depth": ("x", np.array(values, dtype="float32")),
                "scalar": np.float64(42),
                "integer_scalar": np.int32(scalar),
                "integer_field": ("x", np.array(values, dtype="int32")),
            },
            coords={"x": np.arange(len(values))},
        )
        ds.to_netcdf(root / "grid.nc", engine=engine)
        subject.build(str(root), root / "output", pattern="*.nc", static=True)
        with xr.open_dataset(root / "grid.nc", decode_times=False) as expected:
            xr.testing.assert_identical(
                read_reference(root / "output"), expected.load()
            )


@example("%3F%252F%23%25é")
@given(
    st.one_of(
        st.text(alphabet="abc 09%?#é", min_size=1, max_size=12),
        st.lists(
            st.sampled_from(["%3F", "%252F", "%23", "%25", "é", " ", "?", "#", "abc"]),
            min_size=1,
            max_size=5,
        ).map("".join),
    )
)
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


@st.composite
def sparse_fields(draw):
    chunk_count = draw(st.integers(2, 5))
    chunk_size = draw(st.integers(1, 3))
    written = draw(
        st.sets(st.integers(0, chunk_count - 1), min_size=1, max_size=chunk_count - 1)
    )
    return (
        chunk_count,
        chunk_size,
        written,
        draw(st.integers(-100, 100)),
        draw(st.booleans()),
    )


@given(sparse_fields())
def test_sparse_native_chunks_require_declared_cf_fill(case):
    """Storage fill alone cannot safely represent absent chunks for Zarr 2 readers."""
    count, chunk, written, fill, declared = case
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "grid.nc"
        with h5py.File(source, "w") as handle:
            axis = handle.create_dataset("x", data=np.arange(count * chunk))
            axis.make_scale("x")
            field = handle.create_dataset(
                "field",
                shape=(count * chunk,),
                chunks=(chunk,),
                dtype="i4",
                fillvalue=fill,
            )
            field.dims[0].attach_scale(axis)
            if declared:
                field.attrs["_FillValue"] = np.int32(fill)
            for index in written:
                field[index * chunk : (index + 1) * chunk] = fill + 1
        if declared:
            subject.build(str(root), root / "output", pattern="grid.nc", static=True)
            with xr.open_dataset(
                source, engine="h5netcdf", decode_times=False
            ) as expected:
                xr.testing.assert_identical(
                    read_reference(root / "output"), expected.load()
                )
        else:
            with pytest.raises(ValueError, match="unwritten native chunks"):
                subject.build(
                    str(root), root / "output", pattern="grid.nc", static=True
                )
            assert not (root / "output").exists()
            assert not list(root.glob(".output-*"))


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


@given(
    times=st.lists(st.integers(0, 50), min_size=2, max_size=8), split=st.integers(1, 7)
)
def test_source_time_order_is_preserved(times, split):
    """Virtualization preserves duplicates and reversals; consumers own chronology."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        boundary = min(split, len(times) - 1)
        values = np.arange(len(times), dtype="float32").reshape(-1, 1)
        expected = []
        for index, part in enumerate([slice(0, boundary), slice(boundary, None)]):
            path = root / f"{index}.nc"
            write_source(path, values[part], times[part])
            with xr.open_dataset(path, decode_times=False) as native:
                expected.append(native.load())
        subject.build(str(root), root / "output", pattern="*.nc")
        xr.testing.assert_identical(
            read_reference(root / "output"), xr.concat(expected, dim="time")
        )


@pytest.mark.parametrize(
    "mutation",
    ["modify", "remove"],
)
@given(length=st.integers(1, 6))
def test_source_mutations_abort_publication(length, mutation):
    """Inject a source change after serialization, before the final source check.

    Modification/removal of the frozen selection must abort publication.
    The real serializer, validator, and object-store metadata calls still execute.
    """
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        path = root / "0.nc"
        write_source(path, np.ones((length, 1), dtype="float32"), np.arange(length))
        validate = subject.validate_parquet_references

        def mutate(*args):
            validate(*args)
            if mutation == "modify":
                path.write_bytes(path.read_bytes() + b"changed")
            else:
                path.unlink()

        patch.setattr(subject, "validate_parquet_references", mutate)
        with pytest.raises((ValueError, FileNotFoundError)):
            subject.build(str(root), root / "output", pattern="*.nc")
        assert not (root / "output").exists()
        assert not list(root.glob(".output-*"))


@given(length=st.integers(1, 6))
def test_late_sources_are_excluded_from_payload_and_provenance(length):
    """The archive is the frozen selection, not all objects present at commit time."""
    import json

    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        root = Path(directory)
        original = root / "0.nc"
        write_source(
            original,
            np.arange(length, dtype="float32").reshape(-1, 1),
            np.arange(length),
        )
        validate = subject.validate_parquet_references

        def add_source(*args):
            validate(*args)
            write_source(
                root / "1.nc", np.full((1, 1), 1000, dtype="float32"), [length]
            )

        patch.setattr(subject, "validate_parquet_references", add_source)
        subject.build(str(root), root / "output", pattern="*.nc")
        with xr.open_dataset(original, decode_times=False) as expected:
            xr.testing.assert_identical(
                read_reference(root / "output"), expected.load()
            )
        provenance = json.loads((root / "output" / "provenance.json").read_text())
        assert [item["url"] for item in provenance["source_files"]] == [
            original.resolve().as_uri()
        ]
        assert "selection frozen at discovery" in provenance["validation"]


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
