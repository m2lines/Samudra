# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth resolution for viz: `--data` vs. explicit `groundtruth_location`."""

import pytest

from samudra.config import LlcDataSourceConfig, Om4DataSourceConfig
from samudra.utils.location import S3Location, UnresolvedLocation
from samudra.viz.config import VizConfig


def _cfg(*extra_args: str) -> VizConfig:
    return VizConfig.from_yaml_and_cli(["samudra_om4/viz.yaml", *extra_args])


def test_groundtruth_defaults_to_primary_data_source():
    """With no groundtruth_location, ground truth is the primary source's data."""
    cfg = _cfg("--data", "@data/om4_demo.yaml")

    loc = cfg._groundtruth_location()

    assert isinstance(loc, S3Location)
    assert cfg.data is not None
    assert loc == cfg.data.sources[0].data_location


def test_explicit_groundtruth_location_overrides_data():
    """An explicit groundtruth_location wins even when a data source is given."""
    cfg = _cfg("--data", "@data/om4_demo.yaml", "--groundtruth_location", "custom.zarr")

    loc = cfg._groundtruth_location()

    assert loc == UnresolvedLocation(path="custom.zarr")


def test_missing_groundtruth_and_data_is_an_error():
    """Neither groundtruth_location nor data is a loud, actionable failure."""
    cfg = _cfg()
    cfg.data = None
    cfg.groundtruth_location = None

    with pytest.raises(ValueError, match="groundtruth_location"):
        cfg._groundtruth_location()


def test_grid_type_comes_from_the_data_source():
    """Viz reuses the source's grid_type instead of configuring its own.

    Configuring it separately would let viz disagree with train and eval about
    the geometry of the same dataset.
    """
    cfg = _cfg("--data", "@data/om4_demo.yaml")
    assert cfg.data is not None
    source = cfg.data.sources[0]
    assert isinstance(source, Om4DataSourceConfig)
    source.grid_type = "tripolar"

    assert cfg._grid_type() == "tripolar"


def test_grid_type_defaults_to_gaussian_without_a_data_source():
    """An explicit groundtruth_location carries no grid metadata."""
    cfg = _cfg()
    cfg.data = None

    assert cfg._grid_type() == "gaussian"


def test_llc_sources_are_always_curvilinear():
    """LLC carries no `grid_type` field, but its layout is never rectilinear.

    Defaulting a missing field to "gaussian" would quietly tell viz that
    lat-lon-cap data can be plotted against its index axes.
    """
    cfg = _cfg("--data", "@data/om4_demo.yaml")
    assert cfg.data is not None
    cfg.data.sources[0] = LlcDataSourceConfig.model_construct()

    assert cfg._grid_type() == "tripolar"
