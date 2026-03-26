from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data import BatchSampler, SequentialSampler

from ocean_emulators.train import _OffsetBatchSampler
from ocean_emulators.utils.logging import MetricLogger, SmoothedValue
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.wandb import WandBLogger


def test_smoothed_value_empty_is_safe():
    meter = SmoothedValue(fmt="{value:.3f} {max:.3f} {global_avg:.3f}")
    text = str(meter)
    assert "nan" in text.lower()


def test_metric_logger_log_every_supports_global_start_index(caplog):
    caplog.set_level("INFO")
    metric_logger = MetricLogger(delimiter="  ")

    class _Batch:
        load_stats = None

    data_loader = [_Batch(), _Batch()]
    for _ in metric_logger.log_every(
        data_loader,
        print_freq=1,
        header="Train",
        start_index=5,
        total_steps=10,
    ):
        pass

    messages = [record.message for record in caplog.records]
    assert any("[ 5/10]" in message for message in messages)
    assert any("[ 6/10]" in message for message in messages)


def test_offset_batch_sampler_skips_prefix_batches():
    base_sampler = BatchSampler(SequentialSampler(range(10)), batch_size=2, drop_last=False)
    sampler = _OffsetBatchSampler(base_sampler, start_batch=3)
    assert len(sampler) == 2
    assert list(sampler) == [[6, 7], [8, 9]]


def test_offset_batch_sampler_empty_when_offset_past_end():
    base_sampler = BatchSampler(SequentialSampler(range(4)), batch_size=2, drop_last=False)
    sampler = _OffsetBatchSampler(base_sampler, start_batch=10)
    assert len(sampler) == 0
    assert list(sampler) == []


def _fake_cfg():
    return SimpleNamespace(
        model_dump=lambda: {"model": "cfg"},
        experiment=SimpleNamespace(
            output_dir=Path("."),
            name="test-run",
            wandb=SimpleNamespace(model_dump=lambda: {}),
        ),
    )


def _fake_data_container():
    return SimpleNamespace(source=SimpleNamespace(data=SimpleNamespace(attrs={})))


def test_wandb_resume_without_id_starts_new_run(monkeypatch):
    calls = []

    class _Run:
        id = "new-run-id"

    def fake_init(**kwargs):
        calls.append(kwargs)
        return _Run()

    monkeypatch.setattr("ocean_emulators.utils.wandb.torch.load", lambda _: {"wandb_id": None, "wandb_name": "resume-name"})
    monkeypatch.setattr("ocean_emulators.utils.wandb.wandb.init", fake_init)

    with MultitonScope():
        logger = WandBLogger.init_instance()
        logger.configure(enabled=True, is_main_process=True)
        wandb_id, wandb_name = logger.setup_run(
            checkpoint_path="dummy.ckpt",
            cfg=_fake_cfg(),
            data_container=_fake_data_container(),
            finetune=False,
        )

    assert len(calls) == 1
    assert "resume" not in calls[0]
    assert wandb_id == "new-run-id"
    assert wandb_name == "resume-name"


def test_wandb_resume_fallback_keeps_logging_enabled(monkeypatch):
    calls = []

    class _Run:
        id = "fallback-run-id"

    def fake_init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("forced resume failure")
        return _Run()

    monkeypatch.setattr(
        "ocean_emulators.utils.wandb.torch.load",
        lambda _: {"wandb_id": "old-id", "wandb_name": "resume-name"},
    )
    monkeypatch.setattr("ocean_emulators.utils.wandb.wandb.init", fake_init)

    with MultitonScope():
        logger = WandBLogger.init_instance()
        logger.configure(enabled=True, is_main_process=True)
        wandb_id, _ = logger.setup_run(
            checkpoint_path="dummy.ckpt",
            cfg=_fake_cfg(),
            data_container=_fake_data_container(),
            finetune=False,
        )
        assert logger.enabled

    assert len(calls) == 2
    assert calls[0]["resume"] == "must"
    assert calls[0]["id"] == "old-id"
    assert "resume" not in calls[1]
    assert wandb_id == "fallback-run-id"
