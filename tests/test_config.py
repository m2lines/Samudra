# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cftime
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from samudra.config import (
    CpuDataLoadingConfig,
    DataConfig,
    GpuDataLoadingConfig,
    JulianDate,
    LlcDataSourceConfig,
    LlcTimeConfig,
    Om4DataSourceConfig,
    Om4TimeConfig,
    TrainConfig,
)
from samudra.config_schema import get_pydantic_models
from samudra.utils.location import LocalLocation, UnresolvedLocation
from tests.llc_fixtures import write_raw_llc_datasets


def om4_source_config(**kwargs) -> Om4DataSourceConfig:
    return Om4DataSourceConfig(
        train_time=Om4TimeConfig(
            start=JulianDate("1975-01-03"), end=JulianDate("2013-10-05")
        ),
        val_time=Om4TimeConfig(
            start=JulianDate("2013-10-05"), end=JulianDate("2014-10-05")
        ),
        data_location=UnresolvedLocation(path="data.zarr"),
        data_means_location=UnresolvedLocation(path="means.zarr"),
        data_stds_location=UnresolvedLocation(path="stds.zarr"),
        **kwargs,
    )


def test_data_config_rejects_legacy_num_workers_field():
    with pytest.raises(ValidationError, match="num_workers"):
        DataConfig.model_validate(
            {
                "sources": [
                    {
                        "type": "om4",
                        "train_time": {
                            "start": "1975-01-03",
                            "end": "2013-10-05",
                        },
                        "val_time": {
                            "start": "2013-10-05",
                            "end": "2014-10-05",
                        },
                        "data_location": "data.zarr",
                        "data_means_location": "means.zarr",
                        "data_stds_location": "stds.zarr",
                    }
                ],
                "num_workers": 4,
            }
        )


def test_data_config_defaults_to_cpu_loading():
    cfg = DataConfig(sources=[om4_source_config()])

    assert isinstance(cfg.loading, CpuDataLoadingConfig)
    assert cfg.loading.num_workers == 4
    assert cfg.loading.num_pytorch_workers() == 4
    assert isinstance(cfg.sources[0], Om4DataSourceConfig)


def test_om4_dataset_config_builds_selected_spec():
    cfg = om4_source_config(
        prognostic_vars_key="thetao_1",
        boundary_vars_key="hfds",
    )

    spec = cfg.dataset_spec

    assert spec.prognostic_var_names == ["thetao_0"]
    assert spec.boundary_var_names == ["hfds"]


def test_data_source_time_configs_use_native_types():
    om4_time = Om4TimeConfig.model_validate(
        {"start": "2011-09-10", "end": "2011-09-20"}
    )
    llc_time = LlcTimeConfig.model_validate(
        {"start": "2011-09-10T14:00:00+02:00", "end": "2011-09-20T18:00:00Z"}
    )

    assert isinstance(om4_time.start.datetime, cftime.datetime)
    assert om4_time.start.datetime.calendar == "julian"
    assert str(om4_time.start) == "2011-09-10"
    assert isinstance(llc_time.start, np.datetime64)
    assert llc_time.start == np.datetime64("2011-09-10T12:00:00", "ns")
    assert llc_time.model_dump(mode="json")["start"].endswith("Z")


def test_llc_time_config_serializes_as_safe_yaml():
    time = LlcTimeConfig.model_validate(
        {"start": "2011-09-10T12:00:00Z", "end": "2011-09-20T12:00:00Z"}
    )

    serialized = yaml.dump(time.model_dump())

    assert yaml.safe_load(serialized) == {
        "start": "2011-09-10T12:00:00.000000000Z",
        "end": "2011-09-20T12:00:00.000000000Z",
    }


def test_data_source_time_fields_are_immutable():
    source = om4_source_config(
        inference_times=[
            Om4TimeConfig(start=JulianDate("2014-10-10"), end=JulianDate("2014-10-20"))
        ]
    )
    replacement = Om4TimeConfig(
        start=JulianDate("1980-01-01"), end=JulianDate("1981-01-01")
    )

    assert isinstance(source.inference_times, tuple)
    with pytest.raises(ValidationError, match="Field is frozen"):
        source.train_time = replacement
    with pytest.raises(ValidationError, match="Instance is frozen"):
        source.train_time.start = JulianDate("1980-01-01")


def test_om4_time_config_accepts_julian_leap_day():
    time = Om4TimeConfig.model_validate({"start": "1900-02-29", "end": "1900-03-01"})

    assert str(time.start) == "1900-02-29"


def test_llc_time_config_rejects_date_only():
    with pytest.raises(ValidationError, match="should have timezone info"):
        LlcTimeConfig.model_validate({"start": "2011-09-10", "end": "2011-09-20"})


def test_llc_time_config_requires_utc_offset():
    with pytest.raises(ValidationError, match="should have timezone info"):
        LlcTimeConfig.model_validate(
            {"start": "2011-09-10T12:00:00", "end": "2011-09-20T18:00:00Z"}
        )


def test_data_config_accepts_llc_dataset_type():
    cfg = DataConfig.model_validate(
        {
            "sources": [
                {
                    "type": "llc",
                    "face": 2,
                    "i_start": 10,
                    "i_end": 20,
                    "j_start": 30,
                    "j_end": 40,
                    "train_time": {
                        "start": "2011-09-10T12:00:00Z",
                        "end": "2012-09-01T12:00:00Z",
                    },
                    "val_time": {
                        "start": "2012-09-01T12:00:00Z",
                        "end": "2012-11-15T12:00:00Z",
                    },
                    "inference_times": [
                        {
                            "start": "2012-11-15T12:00:00Z",
                            "end": "2012-12-15T12:00:00Z",
                        }
                    ],
                    "data_location": "data.zarr",
                    "data_means_location": "means.zarr",
                    "data_stds_location": "stds.zarr",
                }
            ],
        }
    )

    source = cfg.sources[0]
    assert isinstance(source, LlcDataSourceConfig)
    assert source.face == 2
    assert isinstance(source.inference_times[0], LlcTimeConfig)
    assert source.dataset_spec.prognostic_var_names == ["Theta_0"]


def test_data_config_rejects_invalid_llc_crop():
    with pytest.raises(ValidationError, match="i_start < i_end"):
        DataConfig.model_validate(
            {
                "sources": [
                    {
                        "type": "llc",
                        "i_start": 20,
                        "i_end": 20,
                        "train_time": {
                            "start": "2011-09-10T12:00:00Z",
                            "end": "2012-09-01T12:00:00Z",
                        },
                        "val_time": {
                            "start": "2012-09-01T12:00:00Z",
                            "end": "2012-11-15T12:00:00Z",
                        },
                        "data_location": "data.zarr",
                        "data_means_location": "means.zarr",
                        "data_stds_location": "stds.zarr",
                    }
                ],
            }
        )


def test_data_source_rejects_overlapping_time_splits():
    with pytest.raises(ValidationError, match="Training time range.*overlaps"):
        Om4DataSourceConfig(
            train_time=Om4TimeConfig(
                start=JulianDate("1975-01-03"), end=JulianDate("2013-10-05")
            ),
            val_time=Om4TimeConfig(
                start=JulianDate("2013-01-01"), end=JulianDate("2014-10-05")
            ),
            data_location=UnresolvedLocation(path="data.zarr"),
            data_means_location=UnresolvedLocation(path="means.zarr"),
            data_stds_location=UnresolvedLocation(path="stds.zarr"),
        )


def test_data_config_builds_llc_source_from_local_files(tmp_path):
    write_raw_llc_datasets(tmp_path)
    cfg = DataConfig.model_validate(
        {
            "sources": [
                {
                    "type": "llc",
                    "face": 1,
                    "i_start": 1,
                    "i_end": 4,
                    "j_start": 1,
                    "j_end": 3,
                    "train_time": {
                        "start": "2011-09-10T12:00:00Z",
                        "end": "2011-09-11T12:00:00Z",
                    },
                    "val_time": {
                        "start": "2011-09-11T12:00:00Z",
                        "end": "2011-09-12T12:00:00Z",
                    },
                    "inference_times": [
                        {
                            "start": "2011-09-10T12:00:00Z",
                            "end": "2011-09-11T12:00:00Z",
                        }
                    ],
                    "data_location": "data.zarr",
                    "data_means_location": "means.nc",
                    "data_stds_location": "stds.nc",
                }
            ],
        }
    )

    container = cfg.build(LocalLocation(path=tmp_path))
    source = container.primary_source

    assert source.dataset_spec.type == "llc"
    assert "Theta_0" in source.data.variables
    assert "wetmask_0" in source.data.variables
    assert "face" not in source.data.dims
    assert source.data["Theta_0"].dims == ("time", "lat", "lon")
    assert source.data["wetmask_0"].dims == ("lat", "lon")
    assert source.data["Theta_0"].shape == (2, 2, 3)
    assert np.issubdtype(source.data.time.dtype, np.datetime64)
    assert container.train_sources[0].data.sizes["time"] == 2
    assert container.val_sources[0].data.sizes["time"] == 2
    assert container.inference_source is not None
    assert container.inference_source.data.sizes["time"] == 2

    sliced = source.slice(
        LlcTimeConfig(
            start=np.datetime64("2011-09-10T12:00:00", "ns"),
            end=np.datetime64("2011-09-11T12:00:00", "ns"),
        )
    )
    assert sliced.data.sizes["time"] == 2


def test_data_config_rejects_multiple_dataset_specs(tmp_path):
    cfg = DataConfig(
        sources=[
            om4_source_config(prognostic_vars_key="thetao_1"),
            om4_source_config(prognostic_vars_key="thermo_dynamic_all"),
        ]
    )

    with pytest.raises(AssertionError, match="same dataset spec"):
        cfg.build(LocalLocation(path=tmp_path))


def test_data_config_accepts_gpu_loading():
    cfg = DataConfig.model_validate(
        {
            "sources": [
                {
                    "type": "om4",
                    "train_time": {
                        "start": "1975-01-03",
                        "end": "2013-10-05",
                    },
                    "val_time": {
                        "start": "2013-10-05",
                        "end": "2014-10-05",
                    },
                    "data_location": "data.zarr",
                    "data_means_location": "means.zarr",
                    "data_stds_location": "stds.zarr",
                }
            ],
            "loading": {
                "type": "gpu",
                "kvikio_task_size": 32 * 1024 * 1024,
                "kvikio_num_threads": 4,
            },
        }
    )

    assert isinstance(cfg.loading, GpuDataLoadingConfig)
    assert cfg.loading.kvikio_task_size == 32 * 1024 * 1024
    assert cfg.loading.kvikio_num_threads == 4
    assert cfg.loading.num_pytorch_workers() == 0


def test_train_config_allows_cli_override_for_cpu_num_workers(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "test" / "train_default.yaml"
    )

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
            "--data.loading.num_workers",
            "2",
        ]
    )

    assert isinstance(cfg.data.loading, CpuDataLoadingConfig)
    assert cfg.data.loading.num_workers == 2


def test_get_pydantic_models_collects_loading_variants():
    models = get_pydantic_models(TrainConfig)

    assert models["CpuDataLoadingConfig"] is CpuDataLoadingConfig
    assert models["GpuDataLoadingConfig"] is GpuDataLoadingConfig
