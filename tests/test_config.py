# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from samudra.config import (
    CpuDataLoadingConfig,
    DataConfig,
    DataSourceConfig,
    GpuDataLoadingConfig,
    LlcDatasetConfig,
    Om4DatasetConfig,
    OtterConfig,
    TrainConfig,
)
from samudra.config_schema import get_pydantic_models
from samudra.utils.location import UnresolvedLocation
from samudra.utils.schedule import CosineSchedulerConfig


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


@pytest.mark.parametrize(
    "config_name",
    [
        "train_1deg_torch_smoke.yaml",
        "train_1deg_mse_updates.yaml",
    ],
)
def test_otter_configs_preserve_existing_history_contract(
    config_name: str, tmp_path: Path
):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "otter_om4" / config_name
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

    assert isinstance(cfg.model, OtterConfig)
    assert cfg.data.hist == 1
    assert cfg.steps == [1]
    assert cfg.model.pred_residuals
    assert cfg.model.backbone.patch_size == 3
    assert cfg.model.backbone.stage_depths == (2, 8, 4)


def test_otter_full_config_uses_global_batch_32_on_eight_gpus(tmp_path: Path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "otter_om4"
        / "train_1deg_mse_updates.yaml"
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

    assert cfg.batch_size * 8 * cfg.gradient_accumulation_steps == 32
    assert isinstance(cfg.scheduler, CosineSchedulerConfig)
    assert cfg.scheduler.target_epochs == 70


def test_otter_config_rejects_incompatible_head_dimension():
    with pytest.raises(ValidationError, match="head dimension"):
        OtterConfig.model_validate(
            {
                "backbone": {
                    "token_dim": 40,
                    "num_heads": 4,
                }
            }
        )
