# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.backend import init_train_backend


def test_search_worker_auto_backend_does_not_initialize_ddp(monkeypatch):
    monkeypatch.setenv("SAMUDRA_DISABLE_DISTRIBUTED", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "samudra.backend.init_distributed_mode",
        lambda: (_ for _ in ()).throw(AssertionError("DDP must remain disabled")),
    )
    monkeypatch.setattr("samudra.backend.set_device", lambda device: None)

    device, distributed = init_train_backend("auto")

    assert device == torch.device("cuda")
    assert distributed is None
