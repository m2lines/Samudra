# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pytest

from samudra.experiments.diffusion_latent_summary import continuation_contract


def test_only_budget_change_from_recorded_parent_is_accepted():
    old = dict(seed=1729, learning_rate=1e-4, max_updates=6000, max_seconds=72000)
    new = dict(old, max_updates=10000, max_seconds=244800)
    parent = dict(training_protocol=old, checkpoint_sha256="parent")
    child: dict[str, Any] = dict(training_protocol=new, checkpoint_sha256="child")
    receipt: dict[str, Any] = dict(
        parent_hashes={"best.pt": "parent"},
        parent_protocol=dict(signature=old),
        continuation_protocol=dict(signature=new),
    )
    continuation_contract(parent, child, receipt)
    changed = copy.deepcopy(child)
    changed["training_protocol"]["learning_rate"] = 2e-4
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["continuation_protocol"]["signature"] = changed["training_protocol"]
    with pytest.raises(ValueError, match="lineage"):
        continuation_contract(parent, changed, changed_receipt)
    with pytest.raises(ValueError, match="lineage"):
        continuation_contract(dict(parent, checkpoint_sha256="other"), child, receipt)
