# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import tempfile
from pathlib import Path

import pytest
import torch

from samudra.config import CpuDataLoadingConfig, DynamicLossConfig, TrainConfig
from samudra.datasets import TrainData
from samudra.models.base import BaseModel
from samudra.train import (
    Trainer,
    get_train_batch_progress,
    get_train_batch_throughput_metrics,
    should_log_validation_images,
)
from samudra.utils.ctx import GridContext
from samudra.utils.loss import DynamicLoss
from samudra.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, SAMUDRA_MULTI_CONFIG, TrainPair


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
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda", marks=pytest.mark.cuda)],
    indirect=True,
)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_samudra_mini.yaml")],
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


def _assert_nested_close(actual, expected, path: str) -> None:
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected), path
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict), path
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_nested_close(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(actual, float) or isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), path
    else:
        assert actual == expected, path


def _assert_checkpoints_close(continuous_path: Path, resumed_path: Path) -> None:
    continuous = torch.load(continuous_path, map_location="cpu")
    resumed = torch.load(resumed_path, map_location="cpu")

    ignored_keys = {"wandb_name"}
    assert set(continuous) - ignored_keys == set(resumed) - ignored_keys
    for key in continuous:
        if key in ignored_keys:
            continue
        _assert_nested_close(resumed[key], continuous[key], f"checkpoint[{key!r}]")


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda", marks=pytest.mark.cuda)],
    indirect=True,
)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_samudra_om4_v2_resume.yaml")],
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
    [("mock-om4", "test/train_default_2step.yaml")],
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
    [("mock-om4", "test/train_default_2step.yaml")],
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
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_inference(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    hist = trainer.hist
    assert trainer.inference_src is not None
    resolution = trainer.inference_src.resolution
    wet = trainer.inference_src.masks.prognostic_with_hist(hist)
    ctx = GridContext(wet, resolution, resolution).to(trainer.device)
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
    trainer.train_progress.tensor_bytes_seen = 256
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
    assert trainer.train_progress.tensor_bytes_seen == 256
    assert trainer.train_progress.optimizer_steps == 3
    assert trainer.train_progress.gpu_seconds == 12.5


def test_should_log_validation_images_every_n_epochs():
    assert [
        epoch for epoch in range(1, 26) if should_log_validation_images(epoch, 10)
    ] == [
        1,
        11,
        21,
    ]


def test_should_log_validation_images_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Epoch must be >= 1"):
        should_log_validation_images(0, 10)

    with pytest.raises(ValueError, match="Validation image log frequency must be >= 1"):
        should_log_validation_images(1, 0)


def test_get_train_batch_progress_counts_global_training_units():
    batch_size = 2
    world_size = 4
    input_channels = 3
    boundary_channels = 1
    output_channels = 2
    input_grid = (3, 4)
    output_grid = (5, 6)
    num_model_steps = 2

    ctx = GridContext(
        label_mask=torch.ones(output_channels, *output_grid, dtype=torch.bool),
        input_resolution_cpu=(torch.arange(input_grid[0]), torch.arange(input_grid[1])),
        output_resolution_cpu=(
            torch.arange(output_grid[0]),
            torch.arange(output_grid[1]),
        ),
    )
    train_data = TrainData(input_channels, boundary_channels, ctx)
    for _ in range(num_model_steps):
        train_data.append(
            torch.zeros(batch_size, input_channels, *input_grid),
            torch.zeros(batch_size, boundary_channels, *input_grid),
            torch.zeros(batch_size, output_channels, *output_grid),
        )

    progress = get_train_batch_progress(train_data, world_size)

    assert progress.sample_windows == batch_size * world_size
    assert progress.model_examples == batch_size * world_size * num_model_steps
    assert (
        progress.output_grid_cells
        == batch_size * world_size * num_model_steps * output_grid[0] * output_grid[1]
    )
    assert (
        progress.target_values
        == batch_size
        * world_size
        * num_model_steps
        * output_channels
        * output_grid[0]
        * output_grid[1]
    )
    assert progress.tensor_bytes == world_size * sum(
        tensor.numel() * tensor.element_size()
        for step in range(len(train_data))
        for tensor in train_data[step]
    )
    assert progress.input_grid_lat == input_grid[0]
    assert progress.input_grid_lon == input_grid[1]
    assert progress.output_grid_lat == output_grid[0]
    assert progress.output_grid_lon == output_grid[1]


def test_get_train_batch_throughput_metrics_uses_batch_seconds():
    batch_size = 1
    ctx = GridContext(
        label_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        input_resolution_cpu=(torch.arange(2), torch.arange(3)),
        output_resolution_cpu=(torch.arange(2), torch.arange(3)),
    )
    train_data = TrainData(1, 1, ctx)
    train_data.append(
        torch.zeros(batch_size, 1, 2, 3),
        torch.zeros(batch_size, 1, 2, 3),
        torch.zeros(batch_size, 1, 2, 3),
    )
    progress = get_train_batch_progress(train_data, world_size=2)

    metrics = get_train_batch_throughput_metrics(progress, batch_seconds=0.5)

    assert metrics["throughput/model_examples_per_second"] == 4
    assert metrics["throughput/output_grid_cells_per_second"] == 24
    assert metrics["throughput/tensor_bytes_per_second"] == progress.tensor_bytes / 0.5
    assert get_train_batch_throughput_metrics(progress, batch_seconds=0.0) == {}


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
        assert val_dataset.prognostic_src.grid_size == trainer.primary_src.grid_size

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
    [("mock-om4", "test/train_default.yaml")],
    indirect=True,
)
def test_data_loaders_enable_persistent_workers_on_positive_num_workers(
    trainer_pair: TrainPair,
):
    _, trainer = trainer_pair

    assert trainer.mp_context is not None
    assert trainer.mp_context.get_start_method() == "spawn"
    assert trainer.train_loader._dataloader.persistent_workers is True
    assert trainer.val_loader._dataloader.persistent_workers is True
    assert trainer.inference_src is not None


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default.yaml")],
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
    assert trainer.train_loader._dataloader.persistent_workers is False
    assert trainer.val_loader._dataloader.persistent_workers is False
