from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from ocean_emulators.config import (
    CpuDataLoadingConfig,
    DataConfig,
    DataSourceConfig,
    GpuDataLoadingConfig,
    LlcDatasetConfig,
    Om4DatasetConfig,
    SamudraConfig,
    TimeConfig,
    TrainConfig,
)
from ocean_emulators.config_schema import get_pydantic_models
from ocean_emulators.utils.location import LocalLocation, UnresolvedLocation


def _write_llc_fixture(tmp_path: Path) -> None:
    n_time = 3
    n_face = 2
    n_lev = 51
    n_j = 4
    n_i = 5
    times = np.array(
        [
            "2011-09-10T12:00:00",
            "2011-09-11T12:00:00",
            "2011-09-12T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    data = xr.Dataset(
        {
            "Theta": (
                ["time", "face", "k", "j", "i"],
                np.arange(
                    n_time * n_face * n_lev * n_j * n_i,
                    dtype=np.float32,
                ).reshape(n_time, n_face, n_lev, n_j, n_i),
            ),
            "oceQnet": (
                ["time", "face", "j", "i"],
                np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
                    n_time, n_face, n_j, n_i
                ),
            ),
            "mask_c": (
                ["face", "k", "j", "i"],
                np.ones((n_face, n_lev, n_j, n_i), dtype=bool),
            ),
            "U": (
                ["time", "face", "k", "j", "i_g"],
                np.arange(
                    n_time * n_face * n_lev * n_j * n_i,
                    dtype=np.float32,
                ).reshape(n_time, n_face, n_lev, n_j, n_i),
            ),
            "V": (
                ["time", "face", "k", "j_g", "i"],
                np.arange(
                    n_time * n_face * n_lev * n_j * n_i,
                    dtype=np.float32,
                ).reshape(n_time, n_face, n_lev, n_j, n_i),
            ),
            "oceTAUX": (
                ["time", "face", "j", "i_g"],
                np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
                    n_time, n_face, n_j, n_i
                ),
            ),
            "oceTAUY": (
                ["time", "face", "j_g", "i"],
                np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
                    n_time, n_face, n_j, n_i
                ),
            ),
        },
        coords={
            "time": times,
            "face": np.arange(n_face),
            "k": np.arange(n_lev),
            "j": np.arange(n_j),
            "i": np.arange(n_i),
            "j_g": np.arange(n_j),
            "i_g": np.arange(n_i),
        },
    )
    means = xr.Dataset(
        {
            **{f"Theta_lev_{i}": float(i) for i in range(n_lev)},
            "oceQnet": 0.0,
        }
    )
    stds = xr.Dataset(
        {
            **{f"Theta_lev_{i}": 1.0 for i in range(n_lev)},
            "oceQnet": 1.0,
        }
    )

    data.to_zarr(tmp_path / "data.zarr")
    means.to_netcdf(tmp_path / "means.nc")
    stds.to_netcdf(tmp_path / "stds.nc")


def test_data_config_rejects_legacy_num_workers_field():
    with pytest.raises(ValidationError, match="num_workers"):
        DataConfig.model_validate(
            {
                "sources": [
                    {
                        "data_location": "data.zarr",
                        "data_means_location": "means.zarr",
                        "data_stds_location": "stds.zarr",
                    }
                ],
                "num_workers": 4,
            }
        )


def test_data_config_defaults_to_cpu_loading():
    cfg = DataConfig(
        sources=[
            DataSourceConfig(
                data_location=UnresolvedLocation(path="data.zarr"),
                data_means_location=UnresolvedLocation(path="means.zarr"),
                data_stds_location=UnresolvedLocation(path="stds.zarr"),
            )
        ]
    )

    assert isinstance(cfg.loading, CpuDataLoadingConfig)
    assert cfg.loading.num_workers == 4
    assert cfg.loading.num_pytorch_workers() == 4
    assert isinstance(cfg.dataset, Om4DatasetConfig)


def test_om4_dataset_config_builds_selected_spec():
    cfg = Om4DatasetConfig(
        prognostic_vars_key="thetao_1",
        boundary_vars_key="hfds",
    )

    spec = cfg.build()

    assert spec.prognostic_var_names == ["thetao_0"]
    assert spec.boundary_var_names == ["hfds"]


def test_data_config_accepts_llc_dataset_type():
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "type": "llc",
                "face": 2,
                "i_start": 10,
                "i_end": 20,
                "j_start": 30,
                "j_end": 40,
            },
            "sources": [
                {
                    "data_location": "data.zarr",
                    "data_means_location": "means.zarr",
                    "data_stds_location": "stds.zarr",
                }
            ],
        }
    )

    assert isinstance(cfg.dataset, LlcDatasetConfig)
    assert cfg.dataset.face == 2
    assert cfg.dataset.build().prognostic_var_names == ["Theta_0"]


def test_data_config_rejects_invalid_llc_crop():
    with pytest.raises(ValidationError, match="i_start < i_end"):
        DataConfig.model_validate(
            {
                "dataset": {
                    "type": "llc",
                    "i_start": 20,
                    "i_end": 20,
                },
                "sources": [
                    {
                        "data_location": "data.zarr",
                        "data_means_location": "means.zarr",
                        "data_stds_location": "stds.zarr",
                    }
                ],
            }
        )


def test_data_config_builds_llc_source_from_local_files(tmp_path):
    _write_llc_fixture(tmp_path)
    cfg = DataConfig.model_validate(
        {
            "dataset": {
                "type": "llc",
                "face": 1,
                "i_start": 1,
                "i_end": 4,
                "j_start": 1,
                "j_end": 3,
            },
            "sources": [
                {
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
    assert source.data["Theta_0"].shape == (3, 2, 3)
    assert isinstance(source.data.time.values[0], cftime.DatetimeJulian)

    sliced = source.slice(
        TimeConfig.model_validate({"start": "2011-09-10", "end": "2011-09-12"})
    )
    assert sliced.data.sizes["time"] == 2


def test_data_config_accepts_gpu_loading():
    cfg = DataConfig.model_validate(
        {
            "sources": [
                {
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


def test_llc_train_config_uses_group_norm_and_temporal_stride(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "samudra_llc" / "train.yaml"
    )

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert cfg.data.dataset.type == "llc"
    assert isinstance(cfg.model, SamudraConfig)
    assert cfg.temporal_stride == 24
    assert cfg.data.dataset.prognostic_vars_key == "single_1"
    assert cfg.data.dataset.boundary_vars_key == "single_1"
    assert cfg.model.unet.core_block.norm == "group"
    assert cfg.model.unet.core_block.group_norm_groups == 32


def test_llc_train_config_allows_cli_override_for_temporal_stride(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "samudra_llc" / "train.yaml"
    )

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
            "--temporal_stride",
            "12",
        ]
    )

    assert cfg.temporal_stride == 12
