# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PCGB primitives.

These tests cover the algorithmic core of PCGB without requiring a full
data + model setup: SampleWeights' R2 update, the two MaskSearcher
implementations, and state-dict round-trip. The full PCGB driver is
exercised end-to-end via the launch chain in
scripts/launch_boosted_samudra.sh.
"""

from __future__ import annotations

import torch

from ocean_emulators.pcgb import (
    EnumerateSearcher,
    MixtureSearcher,
    PathMask,
    SampleWeights,
    all_kept,
    bits_to_mask,
)


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
        residuals[10:15] = 5.0
        residuals[20:30] = 1.0

        sw.update(idx, residuals)

        hard_mean = sw.weights[10:15].mean().item()
        easy_mean = sw.weights[40:50].mean().item()
        assert hard_mean > easy_mean, (hard_mean, easy_mean)
        assert abs(sw.weights.mean().item() - 1.0) < 1e-5
        assert (sw.weights.max() / sw.weights.min()).item() <= 20.0 + 1e-3

    def test_beta_clip_prevents_sign_flip_when_L_bar_above_half(self):
        """When L̄ > 0.5, raw AdaBoost.R2 β = L̄/(1-L̄) > 1 — and
        `D' = D · β^(1-L_i)` then up-weights *easier* samples (lower
        L_i) more than harder ones (L_i ≈ 1). Clipping β to ≤ 1 keeps
        the reweighting direction correct: easier samples can only be
        flat or down-weighted, never up-weighted past harder samples.

        We construct residuals that exercise the pathology: 90 samples
        at residual 0.8 (the "easier" cohort) and 10 samples at 1.0
        (the hardest). After range-normalize: L = 0.8 / 1.0 for the
        easier 90 and L = 1.0 for the hardest 10. With uniform D this
        gives L̄ ≈ 0.82 → raw β ≈ 4.56. Then D'_easier ∝ β^0.2 ≈ 1.36
        and D'_hardest ∝ β^0 = 1 — easier > hardest, the sign flip.

        With beta_max=1.0, β clips to 1.0 and D stays uniform; the sign
        flip is suppressed.
        """
        idx = torch.arange(100)
        residuals = torch.cat([torch.full((90,), 0.8), torch.full((10,), 1.0)])

        sw_unclipped = SampleWeights(
            n_samples=100,
            device=torch.device("cpu"),
            ema_lambda=1.0,  # full replacement so we read D' directly
            clamp_limit=1e6,  # disable max/min clamp for this test
            beta_max=1e6,  # effectively disable β clip
        )
        sw_unclipped.update(idx, residuals)
        easier_unclipped = sw_unclipped.weights[:90].mean().item()
        hardest_unclipped = sw_unclipped.weights[90:].mean().item()
        # Sign-flip pathology: easier samples up-weighted *above* hardest.
        assert easier_unclipped > hardest_unclipped, (
            easier_unclipped,
            hardest_unclipped,
        )

        sw_clipped = SampleWeights(
            n_samples=100,
            device=torch.device("cpu"),
            ema_lambda=1.0,
            clamp_limit=1e6,
            beta_max=1.0,
        )
        sw_clipped.update(idx, residuals)
        easier_clipped = sw_clipped.weights[:90].mean().item()
        hardest_clipped = sw_clipped.weights[90:].mean().item()
        # Clipped: easier samples never exceed hardest. (Both are flat at
        # 1.0 here because β=1 → D' = D · 1^(1-L) = D, no change.)
        assert easier_clipped <= hardest_clipped + 1e-6, (
            easier_clipped,
            hardest_clipped,
        )

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


class TestPathMask:
    def test_hashable_and_str(self):
        m = PathMask(skip_drops=(False, True), block_drops=(True, False, True))
        assert hash(m) == hash(m)
        assert str(m) == "s01/b101"
        assert m.num_drops() == 3
        assert m.to_bits() == (False, True, True, False, True)

    def test_skip_only_str_omits_block_segment(self):
        m = PathMask(skip_drops=(True, False))
        assert str(m) == "s10"

    def test_bits_to_mask_splits_correctly(self):
        m = bits_to_mask([True, False, True, False, True], num_skips=2)
        assert m.skip_drops == (True, False)
        assert m.block_drops == (True, False, True)


class TestEnumerateSearcher:
    def setup_method(self):
        self.searcher = EnumerateSearcher(
            num_skips=3, schedule="adversarial", no_repeat_window=0
        )

    def test_pool_size_is_2_to_num_skips(self):
        candidates = self.searcher.candidates_for_round(0)
        assert len(candidates) == 8
        assert all(m.num_drops() <= 3 for m in candidates)
        assert all_kept(3) in candidates

    def test_adversarial_picks_argmax(self):
        candidates = self.searcher.candidates_for_round(0)
        scores = [0.0] * len(candidates)
        scores[5] = 1.0
        scored = list(zip(candidates, scores))
        chosen = self.searcher.select_next(scored, history=[], round_idx=0)
        assert chosen == candidates[5]

    def test_no_repeat_window_falls_back_under_full_exclusion(self):
        """When the window excludes the entire pool, fall back to the
        global argmax (so we never crash from an empty candidate set).
        """
        searcher = EnumerateSearcher(
            num_skips=3, schedule="adversarial", no_repeat_window=8
        )
        candidates = searcher.candidates_for_round(0)
        scores = [0.0] * len(candidates)
        scores[7] = 1.0
        scored = list(zip(candidates, scores))
        chosen = searcher.select_next(scored, history=list(candidates), round_idx=8)
        assert chosen == candidates[7]

    def test_round_robin_walks_pool_in_order(self):
        searcher = EnumerateSearcher(num_skips=3, schedule="round_robin")
        candidates = searcher.candidates_for_round(0)
        zero_scores = [(m, 0.0) for m in candidates]
        for round_idx in range(10):
            chosen = searcher.select_next(zero_scores, history=[], round_idx=round_idx)
            assert chosen == candidates[(round_idx + 1) % len(candidates)]


class TestMixtureSearcher:
    def test_candidate_budget_matches_num_candidates(self):
        """Sampled candidate set has the requested size (within dedupe slack)."""
        s = MixtureSearcher(num_skips=4, num_blocks=10, num_candidates=64, seed=0)
        candidates = s.candidates_for_round(0)
        # Dedupe may remove a few collisions on small lattices; allow ≥90%.
        assert len(candidates) >= 58
        # All candidates have the right shape.
        assert all(
            len(m.skip_drops) == 4 and len(m.block_drops) == 10 for m in candidates
        )

    def test_surrogate_concentrates_drop_probability_on_high_score_bits(self):
        """After observing that bit b correlates with high score, the
        surrogate should bias future samples toward dropping bit b.

        This is the property that makes the searcher useful — without it,
        the prior branch is just random sampling.
        """
        s = MixtureSearcher(
            num_skips=4,
            num_blocks=10,
            num_candidates=200,
            prior_weight=1.0,  # all-prior so we can read p_b directly
            stratified_weight=0.0,
            uniform_weight=0.0,
            surrogate_decay=0.0,  # no smoothing — fresh fit each round
            prior_temperature=2.0,
            seed=42,
        )

        # Synthetic: bit 7 strongly increases score; others near zero. Train
        # the surrogate over a few rounds with fresh candidates each round so
        # the lstsq problem has rank.
        d = 4 + 10
        torch.manual_seed(0)
        for _ in range(3):
            scored = []
            for _ in range(200):
                bits = torch.bernoulli(torch.full((d,), 0.5)).bool().tolist()
                m = bits_to_mask(bits, num_skips=4)
                base_score = 0.05 * sum(bits)
                lift = 1.0 if bits[7] else 0.0
                scored.append((m, base_score + lift))
            s.update(scored)

        importance = s.importance
        # Bit 7 should dominate — it's the only strong signal in the synthetic data.
        assert importance[7].item() > 0.5, importance.tolist()
        assert importance[7].item() > importance[:7].abs().max().item()
        assert importance[7].item() > importance[8:].abs().max().item()

    def test_select_next_picks_argmax_over_scored(self):
        s = MixtureSearcher(num_skips=2, num_blocks=2, num_candidates=8, seed=1)
        candidates = s.candidates_for_round(0)
        scored = [(m, float(i)) for i, m in enumerate(candidates)]
        chosen = s.select_next(scored, history=[], round_idx=0)
        assert chosen == candidates[-1]
