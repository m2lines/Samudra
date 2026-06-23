# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

import samudra.constants as c
from samudra.config import CpuDataLoadingConfig, DynamicLossConfig, TrainConfig
from samudra.models.base import BaseModel
from samudra.train import Trainer, should_log_validation_images
from samudra.utils.ctx import GridContext
from samudra.utils.loss import DynamicLoss
from samudra.utils.multiton import MultitonScope
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
    [("mock-om4", "test/train_samudra_mini.yaml")],
    indirect=True,
)
def test_trainer__samudra_mini_smoke_cuda(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    # The torchinfo summary path can OOM on the shared CI GPU despite this tiny config.
    trainer.num_batches_seen = 1
    trainer.run()


def _write_tiny_om4_zarr(data_root: Path) -> None:
    spec = c.build_om4_spec(
        prognostic_vars_key="thermo_dynamic_all",
        boundary_vars_key="tau_hfds_hfds_anom",
    )
    rng = np.random.default_rng(314159)
    time = xr.cftime_range("1975-01-03", periods=20, freq="5D", calendar="julian")
    lat = np.linspace(-82.0, 82.0, 32, dtype=np.float32)
    lon = np.linspace(0.5, 359.5, 64, dtype=np.float32)

    time_signal = np.arange(len(time), dtype=np.float32)[:, None, None] * 0.03
    lat_signal = np.sin(np.deg2rad(lat, dtype=np.float32))[None, :, None]
    lon_signal = np.cos(np.deg2rad(lon, dtype=np.float32))[None, None, :]

    variables = {}
    for var_index, var_name in enumerate(
        list(spec.prognostic_var_names) + list(spec.boundary_var_names)
    ):
        noise = rng.standard_normal((len(time), len(lat), len(lon)), dtype=np.float32)
        variables[var_name] = (
            ("time", "lat", "lon"),
            (
                0.05 * noise
                + time_signal
                + 0.01 * lat_signal
                + 0.01 * lon_signal
                + var_index * 0.001
            ).astype(np.float32),
        )

    for mask_name in spec.mask_vars:
        variables[mask_name] = (("lat", "lon"), np.ones((len(lat), len(lon)), bool))

    data = xr.Dataset(
        variables,
        coords={
            "time": xr.DataArray(time, dims=["time"]),
            "lat": xr.DataArray(lat, dims=["lat"]),
            "lon": xr.DataArray(lon, dims=["lon"]),
        },
    )
    data_vars = list(spec.prognostic_var_names) + list(spec.boundary_var_names)
    means = data[data_vars].mean(dim=["time", "lat", "lon"])
    stds = data[data_vars].std(dim=["time", "lat", "lon"]) + 1e-6

    data.to_zarr(
        data_root / "OM4.zarr",
        encoding={name: {"compressor": None} for name in data.data_vars},
    )
    means.to_zarr(data_root / "OM4_means.zarr")
    stds.to_zarr(data_root / "OM4_stds.zarr")


def _tiny_v2_config(tmp_path: Path, data_root: Path, run_name: str) -> TrainConfig:
    config_path = Path(__file__).parents[1] / "configs/samudra_om4_v2/train.yaml"
    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(data_root),
            "--experiment.base_output_dir",
            str(tmp_path / "runs"),
            "--experiment.name",
            run_name,
            "--backend",
            "cuda",
            "--train_time.start",
            "1975-01-03",
            "--train_time.end",
            "1975-02-12",
            "--val_time.start",
            "1975-02-12",
            "--val_time.end",
            "1975-03-24",
        ]
    )
    cfg.epochs = 2
    cfg.save_freq = 1
    cfg.batch_size = 1
    cfg.steps = [1]
    cfg.step_transition = []
    cfg.data.concurrent_compute = False
    assert isinstance(cfg.data.loading, CpuDataLoadingConfig)
    cfg.data.loading.num_workers = 0
    cfg.data.loading.persistent_workers = False

    assert hasattr(cfg.model, "unet")
    cfg.model.unet.ch_width = [8, 8, 8, 8]
    cfg.model.unet.dilation = [1, 1, 1, 1]
    cfg.model.unet.n_layers = [1, 1, 1, 1]
    return cfg


def _initialize_manual_run(cfg: TrainConfig) -> Trainer:
    trainer = Trainer(cfg)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if cfg.resume_ckpt_path is None:
        trainer.best_val_loss = 1e8
        trainer.best_inf_loss = 1e8
    trainer.init_data_loaders(cur_step=cfg.steps[0])
    return trainer


def _run_train_val_epoch(trainer: Trainer, epoch: int) -> tuple[float, float]:
    for sampler in [trainer.train_sampler, trainer.val_sampler]:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    train_stats = trainer.train_one_epoch(epoch)
    val_stats = trainer.validate_one_epoch(epoch)
    train_loss = float(train_stats["train/mean/loss"])
    val_loss = float(val_stats["val/mean/loss"])
    trainer.save_all_checkpoints(epoch, val_loss, inf_loss=None)
    return train_loss, val_loss


@pytest.mark.cuda
def test_checkpoint_resume_matches_continuous_tiny_v2_cuda(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    if not torch.cuda.is_available():
        pytest.fail("CUDA test requested but torch.cuda.is_available() is False")

    data_root = tmp_path / "tiny_om4"
    data_root.mkdir()
    _write_tiny_om4_zarr(data_root)

    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    try:
        continuous_cfg = _tiny_v2_config(tmp_path, data_root, "continuous")
        with MultitonScope():
            continuous_trainer = _initialize_manual_run(continuous_cfg)
            _run_train_val_epoch(continuous_trainer, epoch=1)
            checkpoint_path = (
                continuous_trainer.ckpt_paths.latest_checkpoint_path_with_epoch(1)
            )
            continuous_epoch_2 = _run_train_val_epoch(continuous_trainer, epoch=2)
            del continuous_trainer

        resume_cfg = _tiny_v2_config(tmp_path, data_root, "resumed")
        resume_cfg.resume_ckpt_path = str(checkpoint_path)
        with MultitonScope():
            resume_trainer = _initialize_manual_run(resume_cfg)
            assert resume_trainer.start_epoch == 2
            resumed_epoch_2 = _run_train_val_epoch(
                resume_trainer, epoch=resume_trainer.start_epoch
            )
            del resume_trainer
    finally:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(deterministic_algorithms)
        torch.cuda.empty_cache()

    torch.testing.assert_close(
        torch.tensor(continuous_epoch_2),
        torch.tensor(resumed_epoch_2),
        rtol=1e-6,
        atol=1e-6,
    )


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

    assert trainer.mp_context is not None
    assert trainer.mp_context.get_start_method() == "spawn"
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

    assert trainer.mp_context is None
    assert trainer.train_loader._dataloader.persistent_workers is False
    assert trainer.val_loader._dataloader.persistent_workers is False
