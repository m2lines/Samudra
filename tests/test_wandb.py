# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from samudra.utils.multiton import MultitonScope
from samudra.utils.wandb import WandBLogger


class DummyWandbConfig:
    def model_dump(self):
        return {"mode": "online"}


class DummyConfig:
    def __init__(self, output_dir: Path, config: dict | None = None):
        self.experiment = SimpleNamespace(
            output_dir=output_dir,
            wandb=DummyWandbConfig(),
        )
        self.config = config or {}

    def model_dump(self):
        return self.config


class DummyDataContainer:
    train_sources: list[Any] = []


def test_wandb_resume_setup_skips_checkpoint_load_when_disabled(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("not a torch checkpoint")

    def fail_load(*args, **kwargs):
        raise AssertionError("disabled W&B ranks should not load resume checkpoints")

    monkeypatch.setattr(torch, "load", fail_load)

    with MultitonScope():
        logger = WandBLogger.init_instance()
        logger.configure(enabled=True, is_main_process=False)

        assert logger.setup_run(
            str(checkpoint_path),
            cast(Any, DummyConfig(tmp_path)),
            cast(Any, DummyDataContainer()),
        ) == (None, None)


def test_wandb_resume_setup_loads_metadata_on_cpu(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_text("not a torch checkpoint")
    init_kwargs = {}

    def fake_load(path, *, map_location=None):
        assert path == str(checkpoint_path)
        assert map_location == "cpu"
        return {"wandb_id": "run-123", "wandb_name": "resume-me"}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(torch, "load", fake_load)

    with MultitonScope():
        logger = WandBLogger.init_instance()
        logger.configure(enabled=True, is_main_process=True)
        monkeypatch.setattr(logger, "init", fake_init)

        assert logger.setup_run(
            str(checkpoint_path),
            cast(Any, DummyConfig(tmp_path)),
            cast(Any, DummyDataContainer()),
        ) == ("run-123", "resume-me")

    assert init_kwargs["resume"] == "must"
    assert init_kwargs["id"] == "run-123"
    assert init_kwargs["name"] == "resume-me"
    assert init_kwargs["dir"] == tmp_path


def test_wandb_config_preserves_namespaced_search_config(tmp_path):
    expected = {
        "experiment": {
            "search": {
                "name": "perceiver-search",
                "candidate": "direct-query",
                "rung": 2,
            }
        }
    }
    with MultitonScope():
        logger = WandBLogger.init_instance()
        config = logger._make_config(
            cast(Any, DummyConfig(tmp_path, expected)),
            cast(Any, DummyDataContainer()),
        )

    assert config["config"] == expected
