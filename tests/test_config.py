# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from samudra.config import (
    CpuDataLoadingConfig,
    DataConfig,
    DataSourceConfig,
    DecoderConfig,
    EncoderConfig,
    GpuDataLoadingConfig,
    LlcDatasetConfig,
    Om4DatasetConfig,
    PerceiverConfig,
    RustDataLoadingConfig,
    SamudraMiniConfig,
    SamudraMultiConfig,
    TrainConfig,
)
from samudra.config_schema import get_pydantic_models
from samudra.models.modules import (
    CanonicalResampleEncoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    PerceiverEncoder,
    ResampleAttentionResidualDecoder,
    ResampleProjectionDecoder,
)
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


def test_iterable_inverse_proxy_cycles_supported_processor_depths(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "samudra_multi_om4"
        / "train_1deg_iterable_inverse_masked_mse_stratified_updates_proxy.yaml"
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

    assert cfg.train_processor_depths == [1, 2, 4]
    assert cfg.steps == [4]
    assert not hasattr(cfg.model, "latent_boundary_encoder")


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


def test_samudra_multi_accepts_processor_bypass_control():
    cfg = SamudraMultiConfig(bypass_processor=True)

    assert cfg.bypass_processor


def test_naive_perceiver_normalization_controls_replace_lossy_norms():
    cfg = PerceiverConfig(
        depth=2,
        latent_dim=8,
        num_latents=1,
        normalize_input_context=False,
        normalize_encoder_output=False,
    )

    encoder = cfg.build(
        in_channels=10,
        out_channels=12,
        max_patch_size=(1, 1),
        implementation="naive",
    )
    naive_encoder = cast(Any, cast(nn.Sequential, encoder)[1])
    assert all(
        isinstance(cross_attention.norm_context, nn.Identity)
        for cross_attention, _, _ in naive_encoder.layers
    )
    assert isinstance(naive_encoder.to_logits[1], nn.Identity)

    decoder = cfg.build_io(
        in_channels=12,
        queries_dim=8,
        out_channels=10,
        implementation="naive",
    )
    decoder_internal = cast(Any, decoder)
    assert isinstance(decoder_internal.cross_attend_blocks[0].norm_context, nn.Identity)


def test_naive_perceiver_decoder_exposes_cross_attention_width():
    cfg = PerceiverConfig(
        depth=1,
        latent_dim=128,
        num_latents=4,
        cross_heads=2,
        cross_dim_head=64,
    )

    decoder = cfg.build_io(
        in_channels=128,
        queries_dim=128,
        out_channels=77,
        implementation="naive",
    )
    decoder_internal = cast(Any, decoder)
    input_attention = decoder_internal.cross_attend_blocks[0].fn
    output_attention = decoder_internal.decoder_cross_attn.fn

    assert input_attention.heads == 2
    assert input_attention.to_q.out_features == 128
    assert output_attention.heads == 2
    assert output_attention.to_q.out_features == 128


def test_flash_perceiver_rejects_nondefault_normalization_controls():
    cfg = PerceiverConfig(normalize_input_context=False)

    with pytest.raises(ValueError, match="require the naive implementation"):
        cfg.build(
            in_channels=10,
            out_channels=12,
            max_patch_size=(1, 1),
            implementation="flash",
        )


def test_spatial_query_encoder_expands_processor_channels():
    cfg = EncoderConfig(
        perceiver=PerceiverConfig(depth=1, latent_dim=8, num_latents=4),
        spatial_query_shape=(3, 5),
        spatial_query_channels=16,
        queries_dim=8,
    )

    encoder = cfg.build(
        in_channels=10,
        out_channels=128,
        patch_extent=(3.0, 5.0),
        max_lat_size=180,
        max_lon_size=360,
        implementation="naive",
    )

    assert encoder.out_channels == 240


def test_direct_representation_configs_build_projection_heads():
    encoder = EncoderConfig(direct_projection=True).build(
        in_channels=10,
        out_channels=128,
        patch_extent=(1.0, 1.0),
        max_lat_size=180,
        max_lon_size=360,
        implementation="naive",
    )
    decoder = DecoderConfig(direct_projection=True).build(
        in_channels=380,
        out_channels=154,
        patch_extent=(1.0, 1.0),
        implementation="naive",
    )

    assert isinstance(encoder, DirectPatchEncoder)
    assert encoder.out_channels == 128
    assert isinstance(decoder, DirectPatchDecoder)


def test_canonical_resample_encoder_config_uses_configured_finest_grid():
    canonical_resolution = (torch.linspace(-89.75, 89.75, 360), torch.arange(720) / 2)
    encoder = EncoderConfig(canonical_resampling=True, geometry_mode="none").build(
        in_channels=10,
        out_channels=128,
        patch_extent=(1.0, 1.0),
        max_lat_size=360,
        max_lon_size=720,
        implementation="naive",
        canonical_resolution=canonical_resolution,
    )

    assert isinstance(encoder, CanonicalResampleEncoder)
    assert encoder.out_channels == 128
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            encoder.output_resolution((torch.empty(0), torch.empty(0))),
            canonical_resolution,
        )
    )


def test_native_projection_encoder_config_allows_multi_cell_physical_patch():
    encoder = EncoderConfig(native_projection=True, geometry_mode="none").build(
        in_channels=10,
        out_channels=160,
        patch_extent=(90.0, 90.0),
        max_lat_size=360,
        max_lon_size=720,
        implementation="naive",
    )

    assert isinstance(encoder, DirectPatchEncoder)
    assert not encoder.enforce_one_pixel_patch
    x = torch.randn(1, 10, 4, 8)
    resolution = (torch.linspace(-67.5, 67.5, 4), torch.arange(8) * 45.0)
    assert encoder(x, resolution).shape == (1, 160, 4, 8)


def test_encoder_config_can_disable_post_encoder_geometry():
    encoder = EncoderConfig(geometry_mode="none").build(
        in_channels=10,
        out_channels=128,
        patch_extent=(1.0, 1.0),
        max_lat_size=180,
        max_lon_size=360,
        implementation="naive",
    )

    assert isinstance(encoder, PerceiverEncoder)
    assert encoder.geometry_mode == "none"
    assert encoder.pos_embed is None
    assert encoder.scale_embed is None


def test_encoder_sidecar_mode_keeps_geometry_out_of_encoder_content():
    encoder = EncoderConfig(geometry_mode="sidecar").build(
        in_channels=10,
        out_channels=128,
        patch_extent=(1.0, 1.0),
        max_lat_size=180,
        max_lon_size=360,
        implementation="naive",
    )

    assert isinstance(encoder, PerceiverEncoder)
    assert encoder.geometry_mode == "sidecar"
    assert encoder.pos_embed is None
    assert encoder.scale_embed is None


def test_resample_projection_decoder_config_supports_different_grids():
    decoder = DecoderConfig(resample_projection=True, coordinate_resampling=True).build(
        in_channels=128,
        out_channels=154,
        patch_extent=(3.0, 5.0),
        implementation="naive",
    )

    assert isinstance(decoder, ResampleProjectionDecoder)
    assert decoder.coordinate_resampling


def test_resample_projection_decoder_config_supports_masked_output_projection():
    decoder = DecoderConfig(
        resample_projection=True,
        coordinate_resampling=True,
        project_before_resample=True,
    ).build(
        in_channels=128,
        out_channels=154,
        patch_extent=(1.0, 1.0),
        implementation="naive",
    )

    assert isinstance(decoder, ResampleProjectionDecoder)
    assert decoder.project_before_resample


def test_project_before_resample_rejects_other_decoder_types():
    config = DecoderConfig(
        direct_projection=True,
        project_before_resample=True,
    )

    with pytest.raises(ValueError, match="only supported"):
        config.build(
            in_channels=128,
            out_channels=154,
            patch_extent=(1.0, 1.0),
            implementation="naive",
        )


def test_decoder_config_rejects_multiple_projection_controls():
    config = DecoderConfig(direct_projection=True, resample_projection=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        config.build(
            in_channels=128,
            out_channels=154,
            patch_extent=(1.0, 1.0),
            implementation="naive",
        )


def test_hybrid_decoder_config_builds_zero_initialized_local_correction():
    decoder = DecoderConfig(
        resample_attention_residual=True,
        residual_hidden_dim=32,
        residual_heads=2,
        residual_dim_head=16,
        residual_neighborhood_radius=2,
        residual_query_chunk_size=64,
    ).build(
        in_channels=128,
        out_channels=77,
        patch_extent=(1.0, 1.0),
        implementation="naive",
    )

    assert isinstance(decoder, ResampleAttentionResidualDecoder)
    assert decoder.base.coordinate_resampling
    assert decoder.correction.neighborhood_radius == 2
    assert decoder.correction.query_chunk_size == 64
    assert torch.count_nonzero(decoder.correction.output_projection.weight) == 0


def test_direct_encoder_config_rejects_spatial_compression():
    config = EncoderConfig(direct_projection=True)

    with pytest.raises(ValueError, match="requires a one-cell patch"):
        config.build(
            in_channels=10,
            out_channels=128,
            patch_extent=(3.0, 5.0),
            max_lat_size=180,
            max_lon_size=360,
            implementation="naive",
        )
