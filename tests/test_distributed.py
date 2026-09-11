# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist

from samudra.utils.distributed import destroy_distributed_mode


def test_destroy_distributed_mode_skips_uninitialized_group(monkeypatch):
    destroyed = False

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    def destroy() -> None:
        nonlocal destroyed
        destroyed = True

    monkeypatch.setattr(dist, "destroy_process_group", destroy)

    destroy_distributed_mode()

    assert not destroyed


def test_destroy_distributed_mode_destroys_initialized_group(monkeypatch):
    destroyed = False

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def destroy() -> None:
        nonlocal destroyed
        destroyed = True

    monkeypatch.setattr(dist, "destroy_process_group", destroy)

    destroy_distributed_mode()

    assert destroyed
