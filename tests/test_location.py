import tempfile
from pathlib import Path

import pytest
import xarray as xr
from pydantic import ValidationError

from ocean_emulators.utils.location import (
    LocalLocation,
    Location,
    S3Location,
    UnresolvedLocation,
)


class TestUnresolvedLocation:
    """Test cases for UnresolvedLocation class."""

    def test_valid_relative_path(self):
        """Test that relative paths are accepted."""
        loc = UnresolvedLocation(path="data/test.zarr")
        assert loc.path == "data/test.zarr"

    def test_valid_absolute_path(self):
        """Test that absolute paths are accepted."""
        loc = UnresolvedLocation(path="/absolute/path/data.zarr")
        assert loc.path == "/absolute/path/data.zarr"

    def test_invalid_absolute_url(self):
        """Test that absolute URLs are rejected."""
        with pytest.raises(ValidationError, match="Absolute urls are not supported"):
            UnresolvedLocation(path="s3://bucket/path")

    def test_invalid_http_url(self):
        """Test that HTTP URLs are rejected."""
        with pytest.raises(ValidationError, match="Absolute urls are not supported"):
            UnresolvedLocation(path="http://example.com/data")

    def test_serialization(self):
        """Test that UnresolvedLocation serializes to just the path string."""
        loc = UnresolvedLocation(path="data/test.zarr")
        assert loc.model_dump() == "data/test.zarr"


class TestLocalLocation:
    """Test cases for LocalLocation class."""

    def test_creation_with_path_string(self):
        """Test LocalLocation creation with string path."""
        loc = LocalLocation(path=Path("/tmp/data"))
        assert loc.path == Path("/tmp/data")

    def test_creation_with_path_object(self):
        """Test LocalLocation creation with Path object."""
        path = Path("/tmp/data")
        loc = LocalLocation(path=path)
        assert loc.path == path

    def test_str_representation(self):
        """Test string representation of LocalLocation."""
        loc = LocalLocation(path=Path("/tmp/data"))
        assert str(loc) == "/tmp/data"

    def test_resolve_unresolved_location(self):
        """Test resolving an UnresolvedLocation against a LocalLocation."""
        base = LocalLocation(path=Path("/base/path"))
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        resolved = base.resolve(unresolved)

        assert isinstance(resolved, LocalLocation)
        assert resolved.path == Path("/base/path/subdir/file.zarr")

    def test_resolve_resolved_location(self):
        """Test resolving a ResolvedLocation returns it unchanged."""
        base = LocalLocation(path=Path("/base/path"))
        other = LocalLocation(path=Path("/other/path"))

        resolved = base.resolve(other)

        assert resolved is other

    def test_truediv_operator(self):
        """Test the / operator for path joining."""
        base = LocalLocation(path=Path("/base/path"))
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        resolved = base / unresolved

        assert isinstance(resolved, LocalLocation)
        assert resolved.path == Path("/base/path/subdir/file.zarr")

    def test_open_zarr_file(self):
        """Test opening a zarr file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple zarr store
            zarr_path = Path(tmp_dir) / "test.zarr"
            ds = xr.Dataset({"temperature": (["x", "y"], [[1, 2], [3, 4]])})
            ds.to_zarr(zarr_path)

            # Test opening
            loc = LocalLocation(path=zarr_path)
            opened_ds = loc.open()

            assert isinstance(opened_ds, xr.Dataset)
            assert "temperature" in opened_ds.data_vars

    def test_open_netcdf_file(self):
        """Test opening a netcdf file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple netcdf file
            nc_path = Path(tmp_dir) / "test.nc"
            ds = xr.Dataset({"temperature": (["x", "y"], [[1, 2], [3, 4]])})
            ds.to_netcdf(nc_path)

            # Test opening
            loc = LocalLocation(path=nc_path)
            opened_ds = loc.open()

            assert isinstance(opened_ds, xr.Dataset)
            assert "temperature" in opened_ds.data_vars

    def test_open_with_chunks(self):
        """Test opening a file with chunks parameter."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a zarr store
            zarr_path = Path(tmp_dir) / "test.zarr"
            ds = xr.Dataset({"temperature": (["x", "y"], [[1, 2], [3, 4]])})
            ds.to_zarr(zarr_path)

            # Test opening with chunks
            loc = LocalLocation(path=zarr_path)
            opened_ds = loc.open(chunks={})

            assert isinstance(opened_ds, xr.Dataset)
            assert "temperature" in opened_ds.data_vars


class TestS3Location:
    """Test cases for S3Location class."""

    def test_creation_with_endpoint(self):
        """Test S3Location creation with custom endpoint."""
        loc = S3Location(
            bucket="test-bucket",
            path="data/test.zarr",
            endpoint_url="https://s3.example.com",
        )
        assert loc.endpoint_url == "https://s3.example.com"

    def test_url_generation(self):
        """Test URL generation for S3Location."""
        loc = S3Location(bucket="test-bucket", path="data/test.zarr")
        assert loc.url() == "s3://test-bucket/data/test.zarr"

    def test_url_generation_with_special_chars(self):
        """Test URL generation with special characters."""
        loc = S3Location(bucket="test-bucket", path="data/test file.zarr")
        assert loc.url() == "s3://test-bucket/data/test%20file.zarr"

    def test_str_representation(self):
        """Test string representation of S3Location."""
        loc = S3Location(bucket="test-bucket", path="data/test.zarr")
        assert str(loc) == "s3://test-bucket/data/test.zarr"

    def test_resolve_unresolved_location(self):
        """Test resolving an UnresolvedLocation against an S3Location."""
        base = S3Location(bucket="test-bucket", path="base/path")
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        resolved = base.resolve(unresolved)

        assert isinstance(resolved, S3Location)
        assert resolved.bucket == "test-bucket"
        assert resolved.path == "base/path/subdir/file.zarr"
        assert resolved.endpoint_url == base.endpoint_url

    def test_resolve_unresolved_location_with_endpoint(self):
        """Test resolving preserves endpoint_url."""
        base = S3Location(
            bucket="test-bucket",
            path="base/path",
            endpoint_url="https://s3.example.com",
        )
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        resolved = base.resolve(unresolved)
        assert isinstance(resolved, S3Location)

        assert resolved.endpoint_url == "https://s3.example.com"

    def test_resolve_resolved_location(self):
        """Test resolving a ResolvedLocation returns it unchanged."""
        base = S3Location(bucket="test-bucket", path="base/path")
        other = LocalLocation(path=Path("/other/path"))

        resolved = base.resolve(other)

        assert resolved is other

    def test_truediv_operator(self):
        """Test the / operator for path joining."""
        base = S3Location(bucket="test-bucket", path="base/path")
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        resolved = base / unresolved

        assert isinstance(resolved, S3Location)
        assert resolved.path == "base/path/subdir/file.zarr"

class TestLocationValidation:
    """Test cases for the Location validation annotation and string conversion."""

    def test_string_to_unresolved_conversion(self):
        """Test that strings are automatically converted to UnresolvedLocation."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            location: Location

        # Test string conversion
        model = TestModel.model_validate({"location": "data/test.zarr"})
        assert isinstance(model.location, UnresolvedLocation)
        assert model.location.path == "data/test.zarr"

    def test_unresolved_location_passthrough(self):
        """Test that UnresolvedLocation objects pass through unchanged."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            location: Location

        unresolved = UnresolvedLocation(path="data/test.zarr")
        model = TestModel(location=unresolved)
        assert model.location is unresolved


class TestLocationIntegration:
    """Integration tests for the location system."""

    def test_mixed_location_types(self):
        """Test mixing different location types."""
        base_s3 = S3Location(bucket="test-bucket", path="base")
        base_local = LocalLocation(path=Path("/local/base"))
        unresolved = UnresolvedLocation(path="subdir/file.zarr")

        # Test S3 resolution
        s3_resolved = base_s3.resolve(unresolved)
        assert isinstance(s3_resolved, S3Location)
        assert s3_resolved.path == "base/subdir/file.zarr"

        # Test local resolution
        local_resolved = base_local.resolve(unresolved)
        assert isinstance(local_resolved, LocalLocation)
        assert local_resolved.path == Path("/local/base/subdir/file.zarr")

    def test_path_joining_edge_cases(self):
        """Test edge cases in path joining."""
        # Test with trailing slash on base
        base = LocalLocation(path=Path("/base/"))
        unresolved = UnresolvedLocation(path="subdir/file.zarr")
        resolved = base.resolve(unresolved)
        assert isinstance(resolved, LocalLocation)
        assert resolved.path == Path("/base/subdir/file.zarr")

        # Test with leading slash on unresolved path (== absolute path)
        unresolved_abs = UnresolvedLocation(path="/subdir/file.zarr")
        resolved_abs = base.resolve(unresolved_abs)
        assert isinstance(resolved_abs, LocalLocation)
        assert resolved_abs.path == Path("/subdir/file.zarr")

        # Test S3 path joining
        s3_base = S3Location(bucket="test", path="")
        s3_resolved = s3_base.resolve(unresolved)
        assert isinstance(s3_resolved, S3Location)
        assert s3_resolved.path == "/subdir/file.zarr"
