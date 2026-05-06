# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PCGB primitives.

These tests cover the algorithmic core of PCGB without requiring a full
data + model setup: SampleWeights' R2 update, mask selection logic, and
state-dict round-trip. The full PCGB driver is exercised end-to-end via
the launch chain in scripts/launch_boosted_samudra.sh.
"""

from __future__ import annotations

import torch

from ocean_emulators.config import MaskPoolConfig
from ocean_emulators.pcgb import SampleWeights, build_mask_pool, pick_next_mask


class TestSampleWeights:
    def test_r2_update_concentrates_on_hard_examples_and_preserves_mean(self):
        """R2 reweighting upweights samples with high residual and preserves
        the constraint that mean(D) = 1 (so loss magnitudes stay comparable
        across rounds). Also verifies the clamp limit is respected.

        Tests three orthogonal invariants in one update:
          1. Hard samples get higher weight than easy samples.
          2. mean(D_{t+1}) = 1 exactly (renormalization).
          3. max(D)/min(D) <= clamp_limit (DynamicLoss-borrowed clamp).
        """
        sw = SampleWeights(
            n_samples=100,
            device=torch.device("cpu"),
            ema_lambda=0.3,
            clamp_limit=20.0,
        )
        idx = torch.arange(100)
        residuals = torch.zeros(100)
        residuals[10:15] = 5.0  # 5 "hard" samples
        residuals[20:30] = 1.0  # 10 "medium"
        # Rest are 0 (easy).

        sw.update(idx, residuals)

        hard_mean = sw.weights[10:15].mean().item()
        easy_mean = sw.weights[40:50].mean().item()
        assert hard_mean > easy_mean, (hard_mean, easy_mean)
        assert abs(sw.weights.mean().item() - 1.0) < 1e-5
        assert (sw.weights.max() / sw.weights.min()).item() <= 20.0 + 1e-3

    def test_state_dict_round_trip(self):
        sw = SampleWeights(
            n_samples=50,
            device=torch.device("cpu"),
            ema_lambda=0.5,
            clamp_limit=10.0,
        )
        sw.update(torch.arange(50), torch.linspace(0, 1, 50))

        sw2 = SampleWeights(
            n_samples=50,
            device=torch.device("cpu"),
            ema_lambda=0.5,
            clamp_limit=10.0,
        )
        sw2.load_state_dict(sw.state_dict())
        torch.testing.assert_close(sw.weights, sw2.weights)


class TestPickNextMask:
    def setup_method(self):
        self.pool = build_mask_pool(MaskPoolConfig(mode="enumerate_all", num_skips=3))
        # 8 masks, all distinct
        assert len(self.pool) == 8

    def test_adversarial_picks_argmax(self):
        scores = torch.zeros(8)
        scores[5] = 1.0  # mask index 5 has the highest weighted MSE
        chosen = pick_next_mask(
            self.pool,
            scores,
            schedule="adversarial",
            history=[],
            no_repeat_window=0,
        )
        assert chosen == self.pool[5]

    def test_no_repeat_window_excludes_recent_then_falls_back_under_full_exclusion(
        self,
    ):
        """Two orthogonal properties of the no-repeat window:
        1. When fewer than `pool_size` recent masks are excluded, argmax
           over the remaining pool is returned.
        2. When the window excludes the entire pool, fall back to the
           global argmax (so we never crash from an empty candidate set).
        """
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # (1) recent history excludes mask 7 (the global argmax) — picker
        # should return the second-best (any other mask has score 0; we
        # just check it's not pool[7]).
        history_excl_argmax = [self.pool[7]]
        chosen = pick_next_mask(
            self.pool,
            scores,
            schedule="adversarial",
            history=history_excl_argmax,
            no_repeat_window=1,
        )
        assert chosen != self.pool[7]

        # (2) window large enough to exclude entire pool — fall back to
        # global argmax (pool[7]) instead of erroring.
        full_history = list(self.pool)
        chosen = pick_next_mask(
            self.pool,
            scores,
            schedule="adversarial",
            history=full_history,
            no_repeat_window=len(self.pool),
        )
        assert chosen == self.pool[7]

    def test_round_robin_walks_pool_in_order(self):
        scores = torch.zeros(8)  # adversarial would all tie; round-robin ignores
        for round_idx in range(10):
            history = [self.pool[i % 8] for i in range(round_idx)]
            chosen = pick_next_mask(
                self.pool,
                scores,
                schedule="round_robin",
                history=history,
                no_repeat_window=0,
            )
            assert chosen == self.pool[round_idx % 8]
