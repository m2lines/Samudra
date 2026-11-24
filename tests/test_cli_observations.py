# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""The observation entrypoint uses the standard console command and argv contract."""

import sys
from types import ModuleType

from samudra.cli import main


def test_observation_train_command_preserves_config_and_overrides(monkeypatch):
    seen = []
    module = ModuleType("samudra.observations.train")
    module.main = lambda: seen.append(sys.argv.copy())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["samudra", "train-obs", "observations/train.yaml", "--main_updates", "20"],
    )
    main()
    assert seen == [
        ["samudra train-obs", "observations/train.yaml", "--main_updates", "20"]
    ]


def test_schema_command_registers_observation_and_shared_components(monkeypatch):
    from samudra import config_schema
    from samudra.config import ExperimentConfig, UNetBackboneConfig, WandBConfig
    from samudra.observations.config import ObservationTrainConfig

    seen = {}
    monkeypatch.setattr(
        config_schema, "generate_schemas", lambda output, models: seen.update(models)
    )
    monkeypatch.setattr(config_schema, "validate_config_files", lambda *args: True)
    monkeypatch.setattr(sys, "argv", ["samudra.config_schema"])
    config_schema.main()
    for model in (
        ObservationTrainConfig,
        ExperimentConfig,
        WandBConfig,
        UNetBackboneConfig,
    ):
        assert seen[model.__name__] is model
