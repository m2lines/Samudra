from pathlib import Path

import pytest
from pydantic import ValidationError

from ocean_emulators.config import (
    CpuDataLoadingConfig,
    DataConfig,
    DataSourceConfig,
    GpuDataLoadingConfig,
    TrainConfig,
)
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


def test_data_config_rejects_legacy_zarr_gpu_decode_field():
    with pytest.raises(ValidationError, match="zarr_gpu_decode"):
        DataConfig.model_validate(
            {
                "sources": [
                    {
                        "data_location": "data.zarr",
                        "data_means_location": "means.zarr",
                        "data_stds_location": "stds.zarr",
                    }
                ],
                "zarr_gpu_decode": True,
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


def test_llc_train_config_uses_sources_and_gpu_loading(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "samudra_llc" / "train.yaml"
    )

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert len(cfg.data.sources) == 1
    assert cfg.data.sources[0].data_location == UnresolvedLocation(
        path="/orcd/pool/008/jrusak/data/LLC4320v3/"
    )
    assert cfg.data.sources[0].data_means_location == UnresolvedLocation(
        path="002/cody/LLC_means_stds/var_96_LLC_means.zarr"
    )
    assert cfg.data.sources[0].data_stds_location == UnresolvedLocation(
        path="002/cody/LLC_means_stds/var_96_LLC_stds.zarr"
    )
    assert isinstance(cfg.data.loading, GpuDataLoadingConfig)
    assert cfg.data.loading.kvikio_task_size == 64 * 1024 * 1024
    assert cfg.data.loading.kvikio_num_threads == 8


def test_llc_train_config_allows_cli_override_for_kvikio_settings(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "samudra_llc" / "train.yaml"
    )

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
            "--data.loading.kvikio_task_size",
            str(32 * 1024 * 1024),
            "--data.loading.kvikio_num_threads",
            "4",
        ]
    )

    assert isinstance(cfg.data.loading, GpuDataLoadingConfig)
    assert cfg.data.loading.kvikio_task_size == 32 * 1024 * 1024
    assert cfg.data.loading.kvikio_num_threads == 4


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
