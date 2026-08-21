# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from samudra.config import EvalConfig, SamudraMultiConfig, TrainConfig
from samudra.viz.config import VizConfig

CONFIG_DIR = (
    Path(__file__).parent.parent
    / "src"
    / "samudra"
    / "configs"
    / "perceiver_spatial_grid_1deg"
)


def test_full_one_degree_presets_validate():
    train = TrainConfig.from_yaml_and_cli([str(CONFIG_DIR / "train.yaml")])
    evaluation = EvalConfig.from_yaml_and_cli([str(CONFIG_DIR / "eval.yaml")])
    visualization = VizConfig.from_yaml_and_cli([str(CONFIG_DIR / "viz.yaml")])

    assert train.epochs == 70
    assert train.gradient_accumulation_steps == 8
    assert len(train.data.sources) == 1
    assert train.data.sources[0].data_location.path == "OM4.zarr"
    assert train.data.sources[0].boundary_vars_key == "tau_hfds_hfds_anom"
    assert isinstance(train.model, SamudraMultiConfig)
    assert train.model.patch_extent == [6.0, 10.0]
    assert train.model.encoder.architecture == "spatial_grid"
    assert train.model.encoder.spatial_query_shape == (2, 2)
    assert train.model.decoder.architecture == "direct_cross_attention"
    assert train.model.decoder.context_patches == 0
    assert train.model.decoder.output_overlap_patches == 1
    assert train.model.decoder.processor_conditioning is True
    assert train.model.processor.ch_width == [380, 480, 520]
    assert evaluation.num_model_steps_forward == 25
    assert visualization.runs[0].name == "perceiver-spatial-grid-1deg"
