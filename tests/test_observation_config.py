# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Config composition and the standard observation launch/resume lifecycle."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from samudra.config import ExperimentConfig, UNetBackboneConfig, WandBConfig
from samudra.observations.config import ObservationTrainConfig
from samudra.utils.location import UnresolvedLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.train import CheckpointPaths


def config(tmp_path: Path, **changes) -> ObservationTrainConfig:
    return ObservationTrainConfig(
        experiment=ExperimentConfig(
            name="test-observations",
            base_output_dir=str(tmp_path),
            data_root=UnresolvedLocation(path=str(tmp_path / "data")),
            rand_seed=1729,
            wandb=WandBConfig(mode="disabled"),
        ),
        backend="cpu",
        **changes,
    )


def test_bundled_config_includes_and_cli_overrides(tmp_path):
    cfg = ObservationTrainConfig.from_yaml_and_cli(
        [
            "observations/train.yaml",
            "--main_updates",
            "12",
            "--cooldown_updates",
            "3",
            "--gradient_accumulation_steps",
            "2",
            "--fresh_examples",
            "1",
            "--learning_rate",
            "0.0002",
            "--replay.max_age",
            "40",
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.wandb.mode",
            "disabled",
            "--model.backbone.ch_width",
            "[4,8]",
            "--model.backbone.dilation",
            "[1,2]",
            "--model.backbone.n_layers",
            "[1,1]",
        ]
    )
    assert isinstance(cfg.experiment, ExperimentConfig)
    assert isinstance(cfg.experiment.wandb, WandBConfig)
    assert isinstance(cfg.model.backbone, UNetBackboneConfig)
    assert cfg.total_updates == 15
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.learning_rate == 0.0002
    assert cfg.replay.max_age == 40
    assert cfg.model.backbone.ch_width == [4, 8]
    assert (
        cfg.data.local_path("observations", cfg.experiment) == tmp_path / "obs/samples"
    )
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    assert cfg.model.build(names).initializer.channels == 77


def test_scientific_identity_ignores_stop_controls_but_tracks_recipe(tmp_path):
    cfg = config(tmp_path)
    altered = ObservationTrainConfig.model_validate(
        cfg.model_dump()
        | {"stop_after": 20, "debug": True, "resume_ckpt_path": "/run/ckpt.pt"}
    )
    assert altered.scientific_config() == cfg.scientific_config()
    for field, value in (("main_updates", 20), ("learning_rate", 0.0002)):
        changed = ObservationTrainConfig.model_validate(
            cfg.model_dump() | {field: value}
        )
        assert changed.scientific_config() != cfg.scientific_config()
    model_changed = cfg.model_copy(deep=True)
    model_changed.model.backbone.ch_width[0] += 1
    assert model_changed.scientific_config() != cfg.scientific_config()


@pytest.mark.parametrize(
    "override",
    [
        {"fresh_examples": 9},
        {"gradient_accumulation_steps": 0},
        {"mix": "om4-cm4"},
        {"final_learning_rate": 0.1},
        {"batch_size": 2},
    ],
)
def test_invalid_recipes_fail_before_loading_data(tmp_path, override):
    with pytest.raises(ValidationError):
        config(tmp_path, **override)


def test_structured_local_paths_and_saved_yaml_roundtrip(tmp_path):
    cfg = config(tmp_path)
    cfg.data.observations = cfg.data.model_validate(
        {"observations": {"type": "local", "path": str(tmp_path / "samples")}}
    ).observations
    path = tmp_path / "config.yaml"
    cfg.save_yaml(path)
    restored = ObservationTrainConfig.from_yaml(path)
    assert (
        restored.data.local_path("observations", restored.experiment)
        == tmp_path / "samples"
    )


def test_multiprocess_launch_rejected_before_files_or_backend(tmp_path, monkeypatch):
    from samudra.observations import train as entry

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        entry, "init_train_backend", lambda _: pytest.fail("backend initialized")
    )
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="one process"):
        entry.run_training(cfg)
    assert not cfg.experiment.output_dir.exists()


class TinySources:
    names = ("obs",)

    def __init__(self, data, models):
        self.obs = data

    def fresh_origin(self, source, rng):
        return int(rng.integers(10))

    def layout(self, source, origin, month):
        if month >= 12:
            return None
        return {"start_lead": month * 6, "next_start_lead": (month + 1) * 6}

    def load(self, source, origin, month=0):
        return self.layout(source, origin, month)

    def auxiliary(self, source, origin, month, sample):
        return sample


def test_entrypoint_reuses_backend_logger_and_checkpoint_layout(tmp_path, monkeypatch):
    from samudra.backend import init_train_backend
    from samudra.models.surface_initialized import SurfaceInitializedConfig
    from samudra.observations import train as entry
    from samudra.observations import training

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls: dict[str, list[Any]] = {"backend": [], "init": [], "log": [], "finish": []}

    def backend(name):
        calls["backend"].append(name)
        return init_train_backend(name)

    def samples(root, device, **kwargs):
        return SimpleNamespace(
            device=device,
            grid={"names": np.array(["test"])},
            trajectory=lambda *args, **kwargs: {},
        )

    def objective(model, data, source, sample, auxiliary, seed, **kwargs):
        loss = model.weight.square().mean()
        return loss, {"forecast": loss}, torch.zeros(1, 2, 1, 1, 1)

    def wandb_init(**kwargs):
        calls["init"].append(kwargs)
        return SimpleNamespace(id=kwargs["id"])

    monkeypatch.setattr(entry, "init_train_backend", backend)
    monkeypatch.setattr(entry, "Samples", samples)
    monkeypatch.setattr(entry, "Sources", TinySources)
    monkeypatch.setattr(
        entry, "run_manifest", lambda cfg, *_: {"config": cfg.scientific_config()}
    )
    monkeypatch.setattr(
        SurfaceInitializedConfig, "build", lambda *_: torch.nn.Linear(1, 1, bias=False)
    )
    monkeypatch.setattr(training, "objective", objective)
    monkeypatch.setattr(
        entry, "annual_metrics", lambda *_: {training.SELECTION_METRIC: 0.5}
    )
    monkeypatch.setattr(entry.wandb, "init", wandb_init)
    monkeypatch.setattr(
        entry.wandb, "log", lambda values, **kwargs: calls["log"].append(values)
    )
    monkeypatch.setattr(entry.wandb, "finish", lambda: calls["finish"].append(True))
    cfg = config(
        tmp_path,
        main_updates=2,
        cooldown_updates=1,
        stop_after=1,
        gradient_accumulation_steps=2,
        fresh_examples=1,
        save_freq=1,
    )
    cfg.experiment.wandb.mode = "online"
    with MultitonScope():
        entry.run_training(cfg)
    checkpoints = CheckpointPaths(cfg.experiment.nets_dir)
    first = torch.load(checkpoints.latest_checkpoint_path, weights_only=False)
    assert first["updates"] == 1
    assert (cfg.experiment.output_dir / "config.yaml").exists()
    assert (cfg.experiment.output_dir / "metrics.jsonl").exists()
    cfg.stop_after = None
    with MultitonScope():
        entry.run_training(cfg)
    final = torch.load(checkpoints.latest_checkpoint_path, weights_only=False)
    assert final["updates"] == 3
    assert final["cooled"]
    assert checkpoints.best_validation_checkpoint_path.exists()
    assert calls["backend"] == ["cpu", "cpu"]
    assert len(calls["finish"]) == 2
    assert calls["init"][0]["id"] == calls["init"][1]["id"]
    assert calls["init"][1]["resume"] == "allow"
    assert len(calls["log"]) == 4  # Three updates plus final validation.
    saved_config = ObservationTrainConfig.from_yaml(
        cfg.experiment.output_dir / "config.yaml"
    )
    assert saved_config.stop_after is None
    cfg.main_updates += 1
    with MultitonScope(), pytest.raises(ValueError, match="another code/data/recipe"):
        entry.run_training(cfg)
    receipt = json.loads((cfg.experiment.nets_dir / "COMPLETE.json").read_text())
    assert receipt["updates"] == 3
