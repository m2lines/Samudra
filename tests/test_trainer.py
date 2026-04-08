import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch

from ocean_emulators.config import (
    CpuDataLoadingConfig,
    DistributedConfig,
    DynamicLossConfig,
)
from ocean_emulators.models.base import BaseModel
from ocean_emulators.stepper import TrainBatchOutput
from ocean_emulators.train import Trainer, should_log_validation_images
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.device import set_device
from ocean_emulators.utils.loss import DynamicLoss
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, TrainPair


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
    [("mock-om4", "test/train_fomini.yaml")],
    indirect=True,
)
def test_trainer__fomini_smoke_cuda(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    # The torchinfo summary path can OOM on the shared CI GPU despite this tiny config.
    trainer.num_batches_seen = 1
    trainer.run()


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

    model = trainer.model
    assert isinstance(model, BaseModel)
    out = model.forward_once(prog, boundary, ctx)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, Path(tmpdir) / "test.pt")
        trainer.load_checkpoint(Path(tmpdir) / "test.pt")

    out2 = model.forward_once(prog, boundary, ctx)

    assert torch.allclose(out, out2)


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [
        (
            "mock",
            DEFAULT_CONFIG,
            [
                "--train_time.start",
                "1975-08-01",
                "--train_time.end",
                "1975-09-01",
                "--val_time.start",
                "1975-08-15",
                "--val_time.end",
                "1975-09-01",
            ],
        ),
    ],
    indirect=True,
)
def test_trainer_overlapping_time_ranges_raises_error(train_config, caplog):
    """Creating a trainer with overlapping train + val times should error."""

    with MultitonScope():
        with pytest.raises(ValueError, match="Training time range.*"):
            Trainer(train_config)


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

    assert trainer.train_loader._dataloader.persistent_workers is True
    assert trainer.val_loader._dataloader.persistent_workers is True


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

    assert trainer.train_loader._dataloader.persistent_workers is False
    assert trainer.val_loader._dataloader.persistent_workers is False


class _FakeDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, **kwargs):
        super().__init__()
        self.module = module
        self.kwargs = kwargs
        self.no_sync_calls = 0

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_trainer_scales_cpu_loader_workers_per_rank(train_config, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    train_config.data.loading = CpuDataLoadingConfig(num_workers=8)
    train_config.ddp_max_data_workers_per_rank = 2
    train_config.ddp_timeout_minutes = 17
    train_config.inference_epochs = []
    captured_timeout: dict[str, int] = {}

    def fake_init_train_backend(backend, ddp_timeout_minutes=60):
        captured_timeout["value"] = ddp_timeout_minutes
        set_device(torch.device("cpu"))
        return torch.device("cpu"), DistributedConfig(world_size=4, rank=0, gpu=0)

    monkeypatch.setattr(
        "ocean_emulators.train.init_train_backend", fake_init_train_backend
    )
    monkeypatch.setattr("ocean_emulators.train.get_world_size", lambda: 4)
    monkeypatch.setattr(
        "ocean_emulators.train.nn.SyncBatchNorm.convert_sync_batchnorm",
        lambda model: model,
    )
    monkeypatch.setattr(
        "ocean_emulators.train.nn.parallel.DistributedDataParallel", _FakeDDP
    )
    monkeypatch.setattr(
        "ocean_emulators.train.torch.distributed.broadcast",
        lambda tensor, src=0: None,
    )

    with MultitonScope():
        trainer = Trainer(train_config)

    assert captured_timeout["value"] == 17
    assert trainer.num_workers == 2
    assert isinstance(train_config.data.loading, CpuDataLoadingConfig)
    assert train_config.data.loading.num_workers == 2
    assert "Scaling data.loading.num_workers from 8 to 2 per rank" in caplog.text
    assert "Initializing DDP with options" in caplog.text


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_trainer_disables_no_sync_with_static_graph(train_config, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    train_config.gradient_accumulation_steps = 2
    train_config.ddp_static_graph = True
    train_config.ddp_use_no_sync_for_accumulation = True
    train_config.inference_epochs = []

    def fake_init_train_backend(backend, ddp_timeout_minutes=60):
        set_device(torch.device("cpu"))
        return torch.device("cpu"), DistributedConfig(world_size=2, rank=0, gpu=0)

    monkeypatch.setattr(
        "ocean_emulators.train.init_train_backend", fake_init_train_backend
    )
    monkeypatch.setattr("ocean_emulators.train.get_world_size", lambda: 2)
    monkeypatch.setattr(
        "ocean_emulators.train.nn.SyncBatchNorm.convert_sync_batchnorm",
        lambda model: model,
    )
    monkeypatch.setattr(
        "ocean_emulators.train.nn.parallel.DistributedDataParallel", _FakeDDP
    )
    monkeypatch.setattr(
        "ocean_emulators.train.torch.distributed.broadcast",
        lambda tensor, src=0: None,
    )

    with MultitonScope():
        trainer = Trainer(train_config)

    assert trainer.ddp_use_no_sync_for_accumulation is False
    assert (
        "Disabling DDP no_sync accumulation because ddp_static_graph=true"
        in caplog.text
    )


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_train_one_epoch_uses_no_sync_for_intermediate_microbatches(
    trainer_pair: TrainPair, monkeypatch
):
    _, trainer = trainer_pair
    trainer.num_batches_seen = 1
    trainer.gradient_accumulation_steps = 2
    trainer.ddp_static_graph = False
    trainer.ddp_use_no_sync_for_accumulation = True
    trainer.distributed = DistributedConfig(world_size=2, rank=0, gpu=0)

    monkeypatch.setattr(
        "ocean_emulators.train.torch.nn.parallel.DistributedDataParallel", _FakeDDP
    )
    trainer.model = cast(
        torch.nn.parallel.DistributedDataParallel,
        _FakeDDP(trainer.model),
    )
    monkeypatch.setattr(
        "ocean_emulators.train.Stepper.train_batch",
        lambda model, data, loss_fn: TrainBatchOutput(
            loss=torch.tensor(1.0, requires_grad=True),
            loss_per_channel=torch.ones(trainer.num_out),
        ),
    )
    monkeypatch.setattr("ocean_emulators.train.all_reduce_mean", lambda x: x)
    monkeypatch.setattr(trainer, "_maybe_update_loss", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer.profiler, "after_batch", lambda *args, **kwargs: None)

    total_batches = len(trainer.train_loader)
    trainer.train_one_epoch(1)

    expected_no_sync_calls = sum(
        1
        for batch_idx in range(total_batches)
        if (batch_idx + 1) % trainer.gradient_accumulation_steps != 0
        and batch_idx + 1 != total_batches
    )
    assert trainer.model.no_sync_calls == expected_no_sync_calls


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_train_one_epoch_warns_for_slow_batches(
    trainer_pair: TrainPair, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    _, trainer = trainer_pair
    trainer.num_batches_seen = 1
    trainer.slow_batch_log_threshold_seconds = -1.0

    monkeypatch.setattr(
        "ocean_emulators.train.Stepper.train_batch",
        lambda model, data, loss_fn: TrainBatchOutput(
            loss=torch.tensor(1.0, requires_grad=True),
            loss_per_channel=torch.ones(trainer.num_out),
        ),
    )
    monkeypatch.setattr("ocean_emulators.train.all_reduce_mean", lambda x: x)
    monkeypatch.setattr(trainer, "_maybe_update_loss", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer.profiler, "after_batch", lambda *args, **kwargs: None)

    trainer.train_one_epoch(1)

    assert "Slow batch load time" in caplog.text


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_init_data_loaders_disables_pin_memory_for_gpu_decode(
    trainer_pair: TrainPair, caplog
):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.pin_mem = True
    trainer.data_container.sources[0].use_zarr_gpu_decode = True
    trainer.data_container.inference_source.use_zarr_gpu_decode = True

    trainer.init_data_loaders(cur_step=trainer.steps[0])

    assert trainer.train_loader._dataloader.pin_memory is False
    assert trainer.val_loader._dataloader.pin_memory is False
    assert "Disabling DataLoader pin_memory because GPU zarr decode" in caplog.text
