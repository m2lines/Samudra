# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.wandb import WandBLogger


class DummyWandbConfig:
    def model_dump(self):
        return {"mode": "online"}


class DummyConfig:
    def __init__(self, output_dir: Path):
        self.experiment = SimpleNamespace(
            output_dir=output_dir,
            wandb=DummyWandbConfig(),
        )

    def model_dump(self):
        return {}


class DummyDataContainer:
    sources: list[Any] = []


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
