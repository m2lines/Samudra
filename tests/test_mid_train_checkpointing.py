# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

from torch.utils.data import BatchSampler, SequentialSampler

from samudra.train import Trainer, _OffsetBatchSampler
from samudra.utils.logging import MetricLogger
from samudra.utils.train import CheckpointPaths


@dataclass
class _FakeTrainData:
    load_stats = type("LoadStats", (), {"load_time_seconds": 0.0})()


class _FakeTrainLoader:
    def __init__(self, size: int):
        self._items = [_FakeTrainData() for _ in range(size)]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def test_offset_batch_sampler_skips_prefix_batches():
    sampler = BatchSampler(SequentialSampler(range(10)), batch_size=2, drop_last=False)
    offset_sampler = _OffsetBatchSampler(sampler, start_batch=2)

    assert list(offset_sampler) == [[4, 5], [6, 7], [8, 9]]
    assert len(offset_sampler) == 3


def test_offset_batch_sampler_is_empty_when_offset_past_end():
    sampler = BatchSampler(SequentialSampler(range(4)), batch_size=2, drop_last=False)
    offset_sampler = _OffsetBatchSampler(sampler, start_batch=5)

    assert list(offset_sampler) == []
    assert len(offset_sampler) == 0


def test_metric_logger_log_every_supports_global_start_index(caplog):
    caplog.set_level(logging.INFO)
    metric_logger = MetricLogger(delimiter="  ")

    for _data in metric_logger.log_every(
        _FakeTrainLoader(size=2),
        print_freq=1,
        header="Training Epoch: [3]",
        start_index=5,
        total_steps=7,
    ):
        metric_logger.update(loss=1.0)

    assert any("[5/7]" in record.message for record in caplog.records)
    assert any("[6/7]" in record.message for record in caplog.records)


def test_mid_train_checkpoint_path_and_resumable_selection(tmp_path):
    paths = CheckpointPaths(tmp_path)
    assert paths.latest_mid_train_checkpoint_path == tmp_path / "ckpt_mid_train.pt"
    assert paths.latest_resumable_checkpoint_path() is None

    paths.latest_checkpoint_path.write_text("epoch")
    assert paths.latest_resumable_checkpoint_path() == paths.latest_checkpoint_path

    paths.latest_mid_train_checkpoint_path.write_text("mid")
    assert (
        paths.latest_resumable_checkpoint_path()
        == paths.latest_mid_train_checkpoint_path
    )


def test_mid_train_checkpoint_interval_saves_and_advances_timer(monkeypatch):
    trainer = Trainer.__new__(Trainer)
    trainer._next_mid_train_checkpoint_time = 10.0
    trainer.mid_train_checkpoint_interval_seconds = 5.0
    calls = []

    def save_mid_train_checkpoint(**kwargs):
        calls.append(kwargs)

    trainer.save_mid_train_checkpoint = save_mid_train_checkpoint
    monkeypatch.setattr("samudra.train.time.perf_counter", lambda: 21.0)
    monkeypatch.setattr("samudra.train.is_main_process", lambda: True)

    trainer._maybe_save_mid_train_checkpoint(epoch=3, batch_in_epoch=7)

    assert calls == [
        {
            "epoch": 3,
            "batch_in_epoch": 7,
            "reason": "periodic_interval",
        }
    ]
    assert trainer._next_mid_train_checkpoint_time == 25.0


def test_save_mid_train_checkpoint_uses_mid_train_path_and_metadata(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.ckpt_paths = CheckpointPaths(tmp_path)
    captured = {}

    def save_checkpoint(**kwargs):
        captured.update(kwargs)

    trainer.save_checkpoint = save_checkpoint

    path = trainer.save_mid_train_checkpoint(epoch=2, batch_in_epoch=4)

    assert path == tmp_path / "ckpt_mid_train.pt"
    assert captured["checkpoint_path"] == path
    assert captured["epoch"] == 2
    assert captured["batch_in_epoch"] == 4
    assert captured["epoch_complete"] is False
    assert captured["save_reason"] == "mid_train_periodic_interval"
