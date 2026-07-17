# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch

from samudra.config import (
    CpuDataLoadingConfig,
    DynamicLossConfig,
    SearchRunConfig,
    TrainConfig,
)
from samudra.models.base import BaseModel
from samudra.train import (
    Trainer,
    should_log_validation_images,
    should_run_on_epoch_freq,
)
from samudra.utils.ctx import BatchGrid
from samudra.utils.logging import handle_logging
from samudra.utils.loss import DynamicLoss
from samudra.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, SAMUDRA_MULTI_CONFIG, TrainPair


def test_rollout_validation_passes_source_to_inference_dataset(monkeypatch):
    validation_source = object()
    captured_source = None

    class EmptyInferenceDataset:
        def __init__(self, *, source, **kwargs):
            nonlocal captured_source
            captured_source = source

        def __len__(self):
            return 0

    trainer = cast(Any, object.__new__(Trainer))
    trainer.rollout_validation = object()
    trainer.data_bundle = SimpleNamespace(
        train_sources=[object()], val_sources=[validation_source]
    )
    trainer.distributed = None
    trainer.model = SimpleNamespace(eval=lambda: None)
    trainer.prognostic_var_names = []
    trainer.boundary_var_names = []
    trainer.hist = 0
    trainer.normalize_before_mask = True
    trainer.masked_fill_value = 0.0

    monkeypatch.setattr("samudra.train.InferenceDataset", EmptyInferenceDataset)
    monkeypatch.setattr("samudra.train.is_main_process", lambda: True)

    assert trainer.validate_rollout_one_epoch(epoch=1) == {}
    assert captured_source is validation_source


def test_handle_logging_replaces_handlers_between_local_jobs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    handle_logging(False, first)
    logging.getLogger().info("first candidate")
    handle_logging(False, second)
    logging.getLogger().info("second candidate")

    assert "first candidate" in (first / "experiment.log").read_text()
    assert "second candidate" not in (first / "experiment.log").read_text()
    assert (second / "experiment.log").read_text().count("second candidate") == 1


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair
    monkeypatch.setattr(
        "samudra.train.write_training_summary",
        lambda *args, **kwargs: pytest.fail(
            "ordinary training must not emit search summaries"
        ),
    )

    trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_search_training_persists_full_epoch_history(trainer_pair: TrainPair):
    _, trainer = trainer_pair
    trainer.epochs = 2
    history_path = trainer.output_dir / "search_metrics.parquet"
    history_path.unlink(missing_ok=True)
    (trainer.output_dir / "search_worker_status.json").unlink(missing_ok=True)
    trainer.search_run = SearchRunConfig(
        name="agent-observable-search",
        candidate="perceiver",
        rung=0,
        target_epochs=2,
        objective="validation_loss",
        executor="local",
        code_commit="f" * 40,
    )

    trainer.run()

    history = pd.read_parquet(history_path)
    assert history["epoch"].tolist() == [1, 2]
    assert history["candidate"].unique().tolist() == ["perceiver"]
    assert history["train_loss"].notna().all()
    assert history["validation_loss"].notna().all()
    assert "val/mean/loss" in history
    summary = json.loads(
        (trainer.output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["val/mean/loss"] == pytest.approx(history.iloc[-1]["val/mean/loss"])
    worker_status = json.loads(
        (trainer.output_dir / "search_worker_status.json").read_text(encoding="utf-8")
    )
    stages = [event["stage"] for event in worker_status["history"]]
    assert stages.count("first_batch") == 1
    assert stages.count("optimizer_step") == 1


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_sequential_trainers_construct_in_isolated_multiton_scopes(train_config):
    """Local searches can initialize more than one real Trainer per process."""
    for _ in range(2):
        with MultitonScope():
            trainer = Trainer(train_config.model_copy(deep=True))
            trainer.finish()


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda", marks=pytest.mark.cuda)],
    indirect=True,
)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_samudra_mini.yaml")],
    indirect=True,
)
def test_trainer__samudra_mini_smoke_cuda(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    # The torchinfo summary path can OOM on the shared CI GPU despite this tiny config.
    trainer.num_batches_seen = 1
    trainer.run()


def _resume_parity_config(
    train_config: TrainConfig, tmp_path: Path, run_name: str
) -> TrainConfig:
    cfg_data = json.loads(train_config.model_dump_json())
    cfg_data["experiment"]["name"] = run_name
    cfg_data["experiment"]["base_output_dir"] = str(tmp_path / "runs")
    cfg_data["resume_ckpt_path"] = None
    return TrainConfig.model_validate_json(json.dumps(cfg_data))


def _run_to_latest_checkpoint(cfg: TrainConfig) -> Path:
    with MultitonScope():
        trainer = Trainer(cfg)
        if cfg.resume_ckpt_path is None:
            # Match the existing CUDA smoke test: skip the torchinfo summary path,
            # which can OOM on shared CI GPUs even with this small config.
            trainer.num_batches_seen = 1
        trainer.run()
        checkpoint_path = trainer.ckpt_paths.latest_checkpoint_path
        del trainer
    return checkpoint_path


def _assert_nested_close(
    actual, expected, path: str, ignored_paths: set[str] | None = None
) -> None:
    if ignored_paths is not None and path in ignored_paths:
        return

    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected), path
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict), path
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_nested_close(
                actual[key],
                expected[key],
                f"{path}[{key!r}]",
                ignored_paths=ignored_paths,
            )
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(
                actual_item,
                expected_item,
                f"{path}[{index}]",
                ignored_paths=ignored_paths,
            )
    elif isinstance(actual, float) or isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), path
    else:
        assert actual == expected, path


def _assert_checkpoints_close(continuous_path: Path, resumed_path: Path) -> None:
    continuous = torch.load(continuous_path, map_location="cpu")
    resumed = torch.load(resumed_path, map_location="cpu")

    ignored_keys = {"wandb_name"}
    ignored_paths = {
        # Wall-clock timing is checkpointed and resumed, but not deterministic
        # across separate continuous and interrupted training runs.
        "checkpoint['train_progress']['gpu_seconds']",
    }
    assert set(continuous) - ignored_keys == set(resumed) - ignored_keys
    for key in continuous:
        if key in ignored_keys:
            continue
        _assert_nested_close(
            resumed[key],
            continuous[key],
            f"checkpoint[{key!r}]",
            ignored_paths=ignored_paths,
        )


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda", marks=pytest.mark.cuda)],
    indirect=True,
)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_samudra_om4_v2_resume.yaml")],
    indirect=True,
)
def test_checkpoint_resume_matches_continuous_cuda(
    train_config, tmp_path, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        pytest.fail("CUDA test requested but torch.cuda.is_available() is False")

    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    torch.backends.cudnn.benchmark = False
    # Pytorch docs say that bilinear interpolation (which we use) is not usable
    # under torch.use_deterministic_algorithms(True) but see
    # https://github.com/m2lines/Samudra/pull/778#discussion_r3623773768:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    try:
        continuous_cfg = _resume_parity_config(train_config, tmp_path, "continuous")
        continuous_checkpoint = _run_to_latest_checkpoint(continuous_cfg)

        interrupted_cfg = _resume_parity_config(train_config, tmp_path, "resumed")
        interrupted_cfg.epochs = 1
        interrupted_checkpoint = _run_to_latest_checkpoint(interrupted_cfg)

        resume_cfg = _resume_parity_config(train_config, tmp_path, "resumed")
        resume_cfg.resume_ckpt_path = str(interrupted_checkpoint)
        resumed_checkpoint = _run_to_latest_checkpoint(resume_cfg)
    finally:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(deterministic_algorithms)
        torch.cuda.empty_cache()

    _assert_checkpoints_close(continuous_checkpoint, resumed_checkpoint)


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_ema(train_config, caplog):
    caplog.set_level(logging.INFO)
    train_config.epochs = 1
    train_config.save_freq = 1

    with MultitonScope():
        e2e_trainer = Trainer(train_config)
        e2e_trainer.run()

    with MultitonScope():
        train_config.resume_ckpt_path = e2e_trainer.ckpt_paths.latest_checkpoint_path
        resume_trainer = Trainer(train_config)

    # TODO(jder): would be nice to generalize to testing the whole trainer state,
    # or even running it forward and checking the output is identical
    assert resume_trainer._ema == e2e_trainer._ema


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_dynamic_loss_state(train_config, caplog):
    """DynamicLoss has internal rolling state; ensure it round-trips via checkpoints."""
    caplog.set_level(logging.INFO)
    train_config.epochs = 1
    train_config.save_freq = 1
    train_config.loss = DynamicLossConfig(metric="mse", limit=100.0)

    with MultitonScope():
        e2e_trainer = Trainer(train_config)
        assert isinstance(e2e_trainer.loss_fn, DynamicLoss)
        e2e_trainer.run()
        scale_before = e2e_trainer.loss_fn.loss_scale_per_channel().detach().cpu()

        # Make the test meaningful: ensure at least one update away from the init value.
        assert torch.isfinite(scale_before).all()
        assert not torch.allclose(scale_before, torch.ones_like(scale_before))


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_inference(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    hist = trainer.hist
    assert trainer.inference_source is not None
    resolution = trainer.inference_source.resolution
    wet = trainer.inference_source.masks.prognostic_with_hist(hist)
    ctx = BatchGrid(wet, resolution, resolution).to(trainer.device)
    data = trainer.inference_loader.dataset[0]
    inference_dataset, _num_steps = data
    prog, boundary, _label = inference_dataset[0]
    prog = prog.to(trainer.device)
    boundary = boundary.to(trainer.device)
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10
    trainer.train_progress.sample_windows_seen = 2
    trainer.train_progress.model_examples_seen = 4
    trainer.train_progress.output_grid_cells_seen = 24
    trainer.train_progress.target_values_seen = 48
    trainer.train_progress.optimizer_steps = 3
    trainer.train_progress.gpu_seconds = 12.5

    model = trainer.model
    assert isinstance(model, BaseModel)
    out = model.forward_once(prog, boundary, ctx)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, Path(tmpdir) / "test.pt")
        trainer.load_checkpoint(Path(tmpdir) / "test.pt")

    out2 = model.forward_once(prog, boundary, ctx)

    assert torch.allclose(out, out2)
    assert trainer.train_progress.sample_windows_seen == 2
    assert trainer.train_progress.model_examples_seen == 4
    assert trainer.train_progress.output_grid_cells_seen == 24
    assert trainer.train_progress.target_values_seen == 48
    assert trainer.train_progress.optimizer_steps == 3
    assert trainer.train_progress.gpu_seconds == 12.5


def test_should_log_validation_images_every_n_epochs():
    assert [
        epoch for epoch in range(1, 26) if should_log_validation_images(epoch, 10)
    ] == [1, 11, 21]


def test_should_log_validation_images_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Epoch must be >= 1"):
        should_log_validation_images(0, 10)

    with pytest.raises(ValueError, match="Frequency must be >= 1"):
        should_run_on_epoch_freq(1, 0)

    with pytest.raises(ValueError, match="Validation image log frequency"):
        should_log_validation_images(1, 0)


def test_should_run_on_epoch_freq_every_n_epochs():
    assert [epoch for epoch in range(1, 26) if should_run_on_epoch_freq(epoch, 10)] == [
        1,
        11,
        21,
    ]


def test_should_run_on_epoch_freq_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Epoch must be >= 1"):
        should_run_on_epoch_freq(0, 10)

    with pytest.raises(ValueError, match="Frequency must be >= 1"):
        should_run_on_epoch_freq(1, 0)


def test_run_closes_training_and_inference_loaders(monkeypatch):
    class CloseSpy:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.train_loader = CloseSpy()
    trainer.val_loader = CloseSpy()
    trainer.inference_loader = object()
    trainer._run = lambda: None
    closed_inference_loaders: list[object] = []
    monkeypatch.setattr(
        "samudra.train.close_pytorch_dataloader",
        closed_inference_loaders.append,
    )

    trainer.run()

    assert trainer.train_loader.closed
    assert trainer.val_loader.closed
    assert closed_inference_loaders == [trainer.inference_loader]


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", SAMUDRA_MULTI_CONFIG)],
    indirect=True,
)
def test_multiscale_training_validates_primary_source_and_logs_reduced_metrics(
    train_config,
):
    train_config.data.sources.append(train_config.data.sources[0].model_copy(deep=True))
    train_config.data.loading.num_workers = 0
    train_config.model.perceiver_implementation = "naive"
    train_config.debug = True

    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.init_data_loaders(cur_step=train_config.steps[0])

        assert len(trainer.train_loader._datasets) == 2
        assert len(trainer.val_loader._datasets) == 1
        val_dataset = next(iter(trainer.val_loader._datasets.values()))
        assert val_dataset.sources[0].grid_size == trainer.primary_source.grid_size

        class PerfectModel(BaseModel):
            def __init__(self):
                super().__init__(0, 0, 0, False, 1, "constant", 0)

            def forward(self, batch, loss_fn=None):
                return [batch.get_label(0)]

        trainer.model = PerfectModel()
        trainer.test_using_ema = False
        val_logs = trainer.validate_one_epoch(epoch=1)

    assert any(key.startswith("val/reduced/weighted_rmse/") for key in val_logs)


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default.yaml")],
    indirect=True,
)
def test_data_loaders_enable_persistent_workers_on_positive_num_workers(
    trainer_pair: TrainPair,
):
    _, trainer = trainer_pair

    assert trainer.mp_context is not None
    assert trainer.mp_context.get_start_method() == "spawn"
    assert trainer.train_loader._host_loader.persistent_workers is True
    assert trainer.val_loader._host_loader.persistent_workers is True
    assert trainer.inference_source is not None


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default.yaml")],
    indirect=True,
)
def test_data_loaders_disable_persistent_workers_when_num_workers_is_zero(
    train_config,
):
    assert isinstance(train_config.data.loading, CpuDataLoadingConfig)
    train_config.data.loading.num_workers = 0
    train_config.data.loading.persistent_workers = True

    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.init_data_loaders(cur_step=train_config.steps[0])

    assert trainer.mp_context is None
    assert trainer.train_loader._host_loader.persistent_workers is False
    assert trainer.val_loader._host_loader.persistent_workers is False
