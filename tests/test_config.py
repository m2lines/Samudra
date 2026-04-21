from pathlib import Path

import pytest
from pydantic import ValidationError

from ocean_emulators.config import (
    CpuDataLoadingConfig,
    DataConfig,
    DataSourceConfig,
    GpuDataLoadingConfig,
    LlcDatasetConfig,
    Om4DatasetConfig,
    TrainConfig,
)
from ocean_emulators.config_schema import get_pydantic_models
from ocean_emulators.utils.location import UnresolvedLocation


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

    spec = cfg.build_spec()

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
    spec = cfg.dataset.build_spec()
    assert spec.spatial_subset is not None
    assert spec.spatial_subset.face == 2


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
