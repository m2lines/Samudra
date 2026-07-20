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
    RustDataLoadingConfig,
    SamudraMiniConfig,
    SamudraMultiConfig,
    TrainConfig,
)
from samudra.config_schema import get_pydantic_models
from samudra.utils.location import LocalLocation, UnresolvedLocation
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


def test_data_config_accepts_rust_loading():
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
                "type": "rust",
                "prefetch_batches": 3,
                "max_concurrent_reads": 12,
                "prefetch_to_device": False,
            },
        }
    )

    assert isinstance(cfg.loading, RustDataLoadingConfig)
    assert cfg.loading.prefetch_batches == 3
    assert cfg.loading.max_concurrent_reads == 12
    assert cfg.loading.prefetch_to_device is False
    assert cfg.loading.num_pytorch_workers() == 0
    assert cfg.loading.persistent_pytorch_workers() is False


@pytest.mark.parametrize("field", ["prefetch_batches", "max_concurrent_reads"])
def test_rust_loading_requires_positive_bounds(field):
    with pytest.raises(ValidationError, match=field):
        RustDataLoadingConfig.model_validate({field: 0})


def test_rust_loading_rejects_non_local_locations(tmp_path):
    cfg = DataConfig.model_validate(
        {
            "sources": [
                {
                    "data_location": {
                        "type": "s3",
                        "bucket": "example",
                        "path": "data.zarr",
                    },
                    "data_means_location": "means.zarr",
                    "data_stds_location": "stds.zarr",
                }
            ],
            "loading": {"type": "rust"},
        }
    )

    with pytest.raises(ValueError, match="requires local data"):
        cfg.build(LocalLocation(path=tmp_path))


def test_rust_loading_rejects_derived_boundary_variables_before_open(tmp_path):
    cfg = DataConfig(
        dataset=Om4DatasetConfig(boundary_vars_key="tau_hfds_hfds_anom"),
        sources=[
            DataSourceConfig(
                data_location=UnresolvedLocation(path="missing-data.zarr"),
                data_means_location=UnresolvedLocation(path="missing-means.zarr"),
                data_stds_location=UnresolvedLocation(path="missing-stds.zarr"),
            )
        ],
        loading=RustDataLoadingConfig(),
    )

    with pytest.raises(ValueError, match="derived boundary variables.*hfds_anomalies"):
        cfg.build(LocalLocation(path=tmp_path))


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


def test_full_data_1deg_promotion_config_preserves_baseline_contract(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
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

    assert cfg.loss == "mse"
    assert cfg.steps == [1]
    assert cfg.train_sample_selection is None
    assert cfg.batch_size * cfg.gradient_accumulation_steps * 4 == 32
    assert isinstance(cfg.scheduler, CosineSchedulerConfig)
    assert cfg.scheduler.interval == "optimizer_update"
    assert cfg.scheduler.target_updates == 6160
    assert len(cfg.data.sources) == 1
    assert "onedeg" in str(cfg.data.sources[0].data_location.path)
    assert isinstance(cfg.data.loading, RustDataLoadingConfig)
    assert isinstance(cfg.model, SamudraMultiConfig)
    assert cfg.model.patch_extent == [3.0, 5.0]
    assert cfg.model.pred_residuals is False


def test_get_pydantic_models_collects_loading_variants():
    models = get_pydantic_models(TrainConfig)

    assert models["CpuDataLoadingConfig"] is CpuDataLoadingConfig
    assert models["GpuDataLoadingConfig"] is GpuDataLoadingConfig
    assert models["RustDataLoadingConfig"] is RustDataLoadingConfig


def test_selective_checkpointing_is_scoped_to_samudra_multi():
    cfg = SamudraMultiConfig(checkpointing="selective")
    assert cfg.checkpointing == "selective"
    assert cfg.processor_checkpointing() == "all"

    with pytest.raises(ValidationError, match="checkpointing"):
        SamudraMiniConfig.model_validate({"checkpointing": "selective"})
