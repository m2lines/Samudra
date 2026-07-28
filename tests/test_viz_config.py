# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth resolution for viz: `--data` vs. explicit `groundtruth_location`."""

import pytest

from samudra.utils.location import S3Location, UnresolvedLocation
from samudra.viz.config import VizConfig


def _cfg(*extra_args: str) -> VizConfig:
    return VizConfig.from_yaml_and_cli(["samudra_om4/viz.yaml", *extra_args])


def test_groundtruth_defaults_to_primary_data_source():
    """With no groundtruth_location, ground truth is the primary source's data."""
    cfg = _cfg("--data", "@data/om4_demo.yaml")

    loc = cfg._groundtruth_location()

    assert isinstance(loc, S3Location)
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
