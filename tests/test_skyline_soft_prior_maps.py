# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from samudra.skyline.channel_vocabulary import ChannelVocabulary
from samudra.skyline.soft_prior_maps import (
    channel_index,
    point_readout,
    query_column,
    task_grid,
)


def test_field_readout_is_discrete_and_auxiliary_supervision_is_required():
    cfg = {
        "arm": "all_fields",
        "temperature_loss_only": False,
        "model": {"objective": "absolute_ce"},
    }
    vocab = ChannelVocabulary([16.0] * 77, 8)
    channels = torch.tensor([[13, 32, 51, 70, 76], [0, 19, 38, 57, 76]])
    batch = {"query_channel": channels, "target": torch.zeros(2, 5)}
    logits = torch.zeros(2, 5, 8, dtype=torch.bfloat16)
    logits[:, 3, 6] = 3
    pred, extra = point_readout(logits, batch, "so", cfg, vocab)
    assert torch.equal(pred, vocab.centers[channels[:, 3], 6])
    assert extra["predicted_bin"].tolist() == [6, 6]
    torch.testing.assert_close(
        extra["predicted_probability"], logits[:, 3].float().softmax(-1)[:, 6]
    )
    assert channel_index("so", 13) == 70
    with pytest.raises(ValueError, match="unsupervised"):
        point_readout(logits, batch, "so", dict(cfg, temperature_loss_only=True), vocab)
    with pytest.raises(ValueError, match="surface"):
        channel_index("zos", 13)
    with pytest.raises(ValueError, match="supervised"):
        query_column(dict(cfg, arm="temperature"), "uo")


def test_regression_queries_and_native_rectangles():
    cfg = {"arm": "temperature", "model": {"objective": "regression"}}
    pred, extra = point_readout(torch.tensor([[1.0], [2.0]]), {}, "thetao", cfg)
    assert pred.tolist() == [1.0, 2.0] and extra == {}
    source = SimpleNamespace(lat=np.arange(5.0), lon=np.arange(7.0), height=5, width=7)
    rows, cols, yy, xx = task_grid(source, {"lat": [1, 4], "lon": [2, 5]})
    np.testing.assert_array_equal(rows, [1, 2, 3])
    np.testing.assert_array_equal(cols, [2, 3, 4])
    assert yy.shape == xx.shape == (3, 3)
    a = task_grid(source, {"rows": [1, 4], "cols": [2, 5]})
    for x, y in zip(a, (rows, cols, yy, xx), strict=True):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        task_grid(source, {"rows": [-1, 4], "cols": [2, 5]})


def test_latest_loader_uses_ema_not_raw_weights(tmp_path):
    from samudra.skyline.ema import restore_ema
    from samudra.skyline.model import ModelConfig, PointTransformer
    from samudra.skyline.soft_prior_maps import load_selection

    cfg: dict[str, Any] = dict(
        source="test-source",
        reference_root="test-reference",
        arm="all_fields",
        crop=17,
        model=dict(
            width=32,
            layers=1,
            heads=4,
            bins=8,
            objective="regression",
            field_queries=True,
        ),
        ema=dict(decay=0.999, faster_decay_at_start=True),
    )
    model = PointTransformer(ModelConfig(**cfg["model"]))
    expected = {k: v.clone() for k, v in model.named_parameters()}
    ema = restore_ema(model, cfg["ema"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    receipt = {"metadata_sha256": "fixture", "mean": [0.0] * 80, "std": [1.0] * 80}
    source = SimpleNamespace(
        root="test-source", reference_root="test-reference", receipt=lambda: receipt
    )
    path = tmp_path / "latest.pt"
    torch.save(
        dict(
            config=cfg,
            model=model.state_dict(),
            ema=ema.get_state(True),
            step=0,
            source=receipt,
        ),
        path,
    )
    loaded, vocab, actual, metadata = load_selection(
        dict(name="fixture", checkpoint=str(path), step=0, from_latest=True),
        source,
        torch.device("cpu"),
    )
    assert vocab is None and actual == cfg and metadata["weights_kind"] == "ema"
    for name, value in loaded.named_parameters():
        assert torch.equal(value, expected[name])
    with pytest.raises(ValueError, match="Snapshot"):
        load_selection(
            dict(name="fixture", checkpoint=str(path), step=0, from_latest=False),
            source,
            torch.device("cpu"),
        )


def test_shared_task_fields_can_exclude_unsupervised_auxiliaries():
    from samudra.skyline.soft_prior_maps import task_fields

    task = dict(
        models=["temperature", "all-fields"],
        fields=["thetao", "uo", "vo"],
        fields_by_model={"temperature": ["thetao"]},
    )
    assert task_fields(task, "temperature") == ["thetao"]
    assert task_fields(task, "all-fields") == ["thetao", "uo", "vo"]
    invalid: list[dict[str, list[str]]] = [
        {"other": ["thetao"]},
        {"temperature": []},
        {"temperature": ["thetao", "thetao"]},
        {"temperature": ["unknown"]},
    ]
    for overrides in invalid:
        with pytest.raises(ValueError):
            task_fields(dict(task, fields_by_model=overrides), "temperature")
