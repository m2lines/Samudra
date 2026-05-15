# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Path-Cyclic Gradient Boosting (PCGB) for Samudra v2.

Algorithm and motivation: configs/samudra_om4_v2/boosted_samudra.md.

The PCGB driver subclasses :class:`ocean_emulators.train.Trainer` so it
inherits the existing DDP / optimizer / scheduler / wandb / checkpoint
machinery, and replaces the per-epoch loop with a per-round loop that
fixes a deterministic skip-mask, applies per-sample loss reweighting,
and performs an end-of-round adversarial mask scan.

The search over mask space is encapsulated by a :class:`MaskSearcher`,
selected via config:

* :class:`EnumerateSearcher` — v1 default; enumerates all 2^num_skips
  UNet-skip masks (16 for our backbone) with adversarial argmax or
  round-robin scheduling. No conv-block residual drops.
* :class:`MixtureSearcher` — v2; samples masks from a mixture of
  (structured prior, stratified-by-Hamming-weight, uniform) over the
  full path lattice (skips + conv-block residuals = 14 bits, 16 384
  configurations). The structured prior is an online linear surrogate
  fit to past (mask, weighted-MSE) pairs, so non-uniform sampling
  concentrates candidates on masks the model is currently failing on.
"""

from __future__ import annotations

import datetime
import itertools
import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ocean_emulators.aggregator import Aggregator
from ocean_emulators.config import (
    EnumerateSearcherConfig,
    MaskSearcherConfig,
    MixtureSearcherConfig,
    PCGBConfig,
)
from ocean_emulators.datasets import TorchTrainDataset, TrainData, TrainDataLoader
from ocean_emulators.models import Samudra
from ocean_emulators.stepper import validate_batch
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_world_size, is_main_process
from ocean_emulators.utils.logging import MetricLogger
from ocean_emulators.utils.output import TrainBatchOutput
from ocean_emulators.utils.train import collate_raw_train_data
from ocean_emulators.utils.wandb import Metrics

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PathMask + helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PathMask:
    """Composite mask over both UNet skip connections and conv-block residuals.

    ``skip_drops[i] == True``  → zero UNet skip ``i`` (route through bottleneck).
    ``block_drops[j] == True`` → zero conv-block ``j``'s trunk contribution
        (Veit-style "skip block j": ``y = y_{j-1}``).

    Frozen + tuple-typed so PathMask is hashable, which lets us use it in
    sets for the no-repeat-window logic.
    """

    skip_drops: tuple[bool, ...]
    block_drops: tuple[bool, ...] = ()

    def num_drops(self) -> int:
        return sum(self.skip_drops) + sum(self.block_drops)

    def to_bits(self) -> tuple[bool, ...]:
        """Concatenated bit vector: skips first, then blocks."""
        return self.skip_drops + self.block_drops

    def total_bits(self) -> int:
        return len(self.skip_drops) + len(self.block_drops)

    def __str__(self) -> str:
        s = _format_bits(self.skip_drops)
        if self.block_drops:
            return f"s{s}/b{_format_bits(self.block_drops)}"
        return f"s{s}"


def _format_bits(bits: tuple[bool, ...]) -> str:
    return "".join("1" if b else "0" for b in bits)


def all_kept(num_skips: int, num_blocks: int = 0) -> PathMask:
    """Round-1 default: every skip and every block kept (no drops)."""
    return PathMask(
        skip_drops=(False,) * num_skips,
        block_drops=(False,) * num_blocks,
    )


def bits_to_mask(bits: Sequence[bool], num_skips: int) -> PathMask:
    """Split a flat bit vector into (skip_drops, block_drops) by num_skips."""
    bits = tuple(bool(b) for b in bits)
    return PathMask(
        skip_drops=bits[:num_skips],
        block_drops=bits[num_skips:],
    )


# -----------------------------------------------------------------------------
# MaskSearcher — protocol + two implementations
# -----------------------------------------------------------------------------


class MaskSearcher(Protocol):
    """Owns the mask candidate set and the next-mask selection per round.

    Implementations are stateful across rounds (a v2-style searcher fits a
    surrogate over past observations to bias future sampling).
    """

    @property
    def num_skips(self) -> int: ...

    @property
    def num_blocks(self) -> int: ...

    def candidates_for_round(self, round_idx: int) -> list[PathMask]:
        """Masks to evaluate during round ``round_idx``'s scoring pass."""
        ...

    def update(self, scored: list[tuple[PathMask, float]]) -> None:
        """Update internal state from this round's scored candidates."""
        ...

    def select_next(
        self,
        scored: list[tuple[PathMask, float]],
        history: list[PathMask],
        round_idx: int,
    ) -> PathMask:
        """Pick the mask to play in the next round."""
        ...


class EnumerateSearcher:
    """v1: enumerate all 2^num_skips UNet-skip masks; no block drops.

    Pool size = 16 for our 4-skip backbone. Selection is either argmax
    over the round's scored masks (``schedule="adversarial"``) or a
    deterministic round-robin walk (``schedule="round_robin"``).
    """

    def __init__(
        self,
        num_skips: int,
        schedule: str = "adversarial",
        no_repeat_window: int = 0,
    ):
        if num_skips < 1:
            raise ValueError(f"num_skips must be >= 1, got {num_skips}")
        self._num_skips = num_skips
        self._pool = [
            PathMask(skip_drops=tuple(bits))
            for bits in itertools.product([False, True], repeat=num_skips)
        ]
        self._schedule = schedule
        self._no_repeat_window = no_repeat_window

    @property
    def num_skips(self) -> int:
        return self._num_skips

    @property
    def num_blocks(self) -> int:
        return 0

    def candidates_for_round(self, round_idx: int) -> list[PathMask]:
        return list(self._pool)

    def update(self, scored: list[tuple[PathMask, float]]) -> None:
        # Stateless — enumeration doesn't learn from past observations.
        return

    def select_next(
        self,
        scored: list[tuple[PathMask, float]],
        history: list[PathMask],
        round_idx: int,
    ) -> PathMask:
        if self._schedule == "round_robin":
            return self._pool[(round_idx + 1) % len(self._pool)]

        excluded: set[PathMask] = set()
        if self._no_repeat_window > 0:
            excluded.update(history[-self._no_repeat_window :])

        # Argmax over candidates not in the no-repeat window. Fall back to
        # the global argmax if the window excludes the entire pool.
        best: tuple[PathMask, float] | None = None
        for mask, score in scored:
            if mask in excluded:
                continue
            if best is None or score > best[1]:
                best = (mask, score)
        if best is None:
            best = max(scored, key=lambda x: x[1])
        return best[0]


class MixtureSearcher:
    """v2: mixture sampling with online linear surrogate.

    Each round samples ``num_candidates`` masks from a three-way mixture:

      * **Structured prior** (``prior_weight`` of the budget): per-bit
        Bernoulli with ``p_b = sigmoid(α · w_b)``, where ``w_b`` is the
        per-bit "importance" estimated from past rounds (high w_b = bit
        b, when set, predicts higher weighted MSE → dropping it is more
        likely to surface the argmax).
      * **Stratified by Hamming weight** (``stratified_weight``): pick
        a target number of drops ``k`` uniformly from {1, …, d-1}, then
        sample a mask with exactly k drops uniformly. Hedges against
        the linear surrogate underfitting bit-interaction effects.
      * **Uniform** (``uniform_weight``): per-bit Bernoulli(0.5).
        Exploration tail; prevents the prior from collapsing onto a
        narrow region of the lattice.

    The surrogate ``w_b`` is fit each round by linear regression of
    weighted MSE on the bit pattern, then EMA-blended into the running
    estimate (``surrogate_decay`` = retention of past w_b).

    Mask selection is always argmax over the sampled candidates.
    """

    def __init__(
        self,
        num_skips: int,
        num_blocks: int,
        num_candidates: int = 256,
        prior_weight: float = 0.6,
        stratified_weight: float = 0.25,
        uniform_weight: float = 0.15,
        surrogate_decay: float = 0.7,
        prior_temperature: float = 1.0,
        no_repeat_window: int = 0,
        seed: int = 0,
    ):
        if num_skips + num_blocks < 1:
            raise ValueError("MixtureSearcher needs at least 1 maskable bit.")
        total = prior_weight + stratified_weight + uniform_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"mixture weights must sum to 1; got {total} = "
                f"{prior_weight} + {stratified_weight} + {uniform_weight}"
            )
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if not (0.0 <= surrogate_decay <= 1.0):
            raise ValueError(
                f"surrogate_decay must be in [0, 1], got {surrogate_decay}"
            )

        self._num_skips = num_skips
        self._num_blocks = num_blocks
        self._d = num_skips + num_blocks
        self._num_candidates = num_candidates
        self._prior_weight = prior_weight
        self._stratified_weight = stratified_weight
        self._uniform_weight = uniform_weight
        self._surrogate_decay = surrogate_decay
        self._prior_temperature = prior_temperature
        self._no_repeat_window = no_repeat_window

        # Per-bit importance, init to zero (uniform Bernoulli prior).
        self._w = torch.zeros(self._d, dtype=torch.float64)
        # Reproducibility — torch generators per-rank diverge naturally if
        # callers seed differently; we only need within-rank determinism.
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    @property
    def num_skips(self) -> int:
        return self._num_skips

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def importance(self) -> torch.Tensor:
        """Read-only view of the current per-bit importance vector."""
        return self._w.detach().clone()

    def candidates_for_round(self, round_idx: int) -> list[PathMask]:
        n_prior = int(round(self._num_candidates * self._prior_weight))
        n_strat = int(round(self._num_candidates * self._stratified_weight))
        n_uniform = self._num_candidates - n_prior - n_strat

        masks: list[PathMask] = []
        masks.extend(self._sample_prior(n_prior))
        masks.extend(self._sample_stratified(n_strat))
        masks.extend(self._sample_uniform(n_uniform))

        # Dedupe across mixture branches (collisions are rare but harmless).
        seen: set[PathMask] = set()
        deduped: list[PathMask] = []
        for m in masks:
            if m not in seen:
                deduped.append(m)
                seen.add(m)
        return deduped

    def _sample_prior(self, n: int) -> list[PathMask]:
        if n <= 0:
            return []
        p = torch.sigmoid(self._prior_temperature * self._w).float()
        bits_batch = torch.bernoulli(p.expand(n, -1), generator=self._gen).bool()
        return [bits_to_mask(row.tolist(), self._num_skips) for row in bits_batch]

    def _sample_stratified(self, n: int) -> list[PathMask]:
        if n <= 0:
            return []
        masks: list[PathMask] = []
        for _ in range(n):
            # k ∈ {1, ..., d-1} — skip 0 (all-keep) and d (all-drop) as edge
            # cases that get coverage from the prior/uniform branches anyway.
            k = int(torch.randint(1, max(self._d, 2), (1,), generator=self._gen).item())
            k = max(1, min(k, self._d - 1)) if self._d > 1 else 1
            perm = torch.randperm(self._d, generator=self._gen)
            bits = torch.zeros(self._d, dtype=torch.bool)
            bits[perm[:k]] = True
            masks.append(bits_to_mask(bits.tolist(), self._num_skips))
        return masks

    def _sample_uniform(self, n: int) -> list[PathMask]:
        if n <= 0:
            return []
        bits_batch = torch.bernoulli(
            torch.full((n, self._d), 0.5), generator=self._gen
        ).bool()
        return [bits_to_mask(row.tolist(), self._num_skips) for row in bits_batch]

    def update(self, scored: list[tuple[PathMask, float]]) -> None:
        """Refit the per-bit importance vector from this round's scores."""
        if len(scored) < 2:
            return  # not enough samples for regression
        X = torch.zeros(len(scored), self._d, dtype=torch.float64)
        y = torch.zeros(len(scored), dtype=torch.float64)
        for i, (mask, score) in enumerate(scored):
            for j, b in enumerate(mask.to_bits()):
                X[i, j] = 1.0 if b else 0.0
            y[i] = score
        # Center y for numerical stability (intercept absorbed implicitly).
        y = y - y.mean()
        try:
            beta = torch.linalg.lstsq(X, y).solution
        except Exception:
            return  # singular X (rare); skip update silently
        self._w = self._surrogate_decay * self._w + (1.0 - self._surrogate_decay) * beta

    def select_next(
        self,
        scored: list[tuple[PathMask, float]],
        history: list[PathMask],
        round_idx: int,
    ) -> PathMask:
        excluded: set[PathMask] = set()
        if self._no_repeat_window > 0:
            excluded.update(history[-self._no_repeat_window :])
        candidates = [(m, s) for m, s in scored if m not in excluded]
        if not candidates:
            candidates = list(scored)
        return max(candidates, key=lambda x: x[1])[0]


def build_searcher(cfg: MaskSearcherConfig) -> MaskSearcher:
    """Construct a MaskSearcher from its discriminated-union config."""
    if isinstance(cfg, EnumerateSearcherConfig):
        return EnumerateSearcher(
            num_skips=cfg.num_skips,
            schedule=cfg.schedule,
            no_repeat_window=cfg.no_repeat_window,
        )
    if isinstance(cfg, MixtureSearcherConfig):
        return MixtureSearcher(
            num_skips=cfg.num_skips,
            num_blocks=cfg.num_blocks,
            num_candidates=cfg.num_candidates,
            prior_weight=cfg.prior_weight,
            stratified_weight=cfg.stratified_weight,
            uniform_weight=cfg.uniform_weight,
            surrogate_decay=cfg.surrogate_decay,
            prior_temperature=cfg.prior_temperature,
            no_repeat_window=cfg.no_repeat_window,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown MaskSearcherConfig type: {type(cfg).__name__}")


# -----------------------------------------------------------------------------
# Per-sample weights (D_t) — AdaBoost.R2-proper with EMA + clamp
# -----------------------------------------------------------------------------


@dataclass
class _R2Stats:
    """Per-round diagnostics returned by SampleWeights.update for logging."""

    L_bar: float
    beta: float
    max_weight: float
    min_weight: float
    entropy_bits: float


class SampleWeights:
    """Per-sample loss multipliers (D_t) for PCGB's R2 reweighting.

    Holds the full length-N weight tensor on every rank (N is small —
    thousands at most for ocean emulators). The tensor is identical
    across ranks at all times: per-round residual contributions are
    accumulated via :func:`torch.distributed.all_reduce` sum before the
    R2 update is applied, and the update itself is deterministic given
    full residuals.

    Reweighting is AdaBoost.R2-proper (Drucker, 1997) with two stability
    additions lifted from :class:`ocean_emulators.utils.loss.DynamicLoss`:

      * EMA smoothing — `D_{t+1} = (1-λ) · D_t + λ · D_R2` — to avoid
        per-round whipsaw when a single round's residuals happen to spike.
      * Max/min clamp — bounds the dynamic range of D_t to prevent
        collapse onto a tiny set of "hardest" samples.

    The R2 *abort-on-failure* rule (L̄ ≥ 0.5 → stop) is **not** enforced
    here; cold-start with random init can push L̄ above 0.5 in early
    rounds without the procedure being broken.
    """

    def __init__(
        self,
        n_samples: int,
        device: torch.device,
        ema_lambda: float,
        clamp_limit: float,
        beta_max: float = 1.0,
    ):
        if not (0.0 < ema_lambda <= 1.0):
            raise ValueError(f"ema_lambda must be in (0, 1], got {ema_lambda}")
        if clamp_limit <= 1.0:
            raise ValueError(f"clamp_limit must be > 1, got {clamp_limit}")
        if beta_max <= 0.0:
            raise ValueError(f"beta_max must be > 0, got {beta_max}")
        self._weights = torch.ones(n_samples, device=device)
        self._device = device
        self._n = n_samples
        self._ema_lambda = ema_lambda
        self._clamp_limit = clamp_limit
        # Clip the AdaBoost.R2 confidence β to prevent the sign-flip
        # pathology: when L̄ > 0.5, raw β > 1 would *up*-weight easy
        # examples and leave hard ones unchanged (since β^(1-L_i) is
        # largest at L_i=0). Capping at 1.0 keeps the reweighting
        # direction correct — easy examples can only be down-weighted or
        # held flat, never inverted into "easier than hard." This is the
        # natural extension of the canonical "abort if L̄ ≥ 0.5" rule.
        self._beta_max = beta_max

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def __len__(self) -> int:
        return self._n

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-sample weight multipliers for a batch's sample indices."""
        idx = indices.to(self._device, dtype=torch.long, non_blocking=True)
        return self._weights.index_select(0, idx)

    def update(
        self,
        local_indices: torch.Tensor,
        local_residuals: torch.Tensor,
    ) -> _R2Stats:
        if local_indices.shape != local_residuals.shape:
            raise ValueError(
                f"indices/residuals shape mismatch: {local_indices.shape} "
                f"vs {local_residuals.shape}"
            )

        global_resid = torch.zeros(self._n, device=self._device)
        global_count = torch.zeros(self._n, device=self._device)
        if local_indices.numel() > 0:
            idx = local_indices.to(self._device, dtype=torch.long)
            global_resid.scatter_add_(
                0, idx, local_residuals.to(self._device).abs().float()
            )
            global_count.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        if get_world_size() > 1:
            dist.all_reduce(global_resid)
            dist.all_reduce(global_count)

        seen = global_count > 0
        if not bool(seen.any()):
            raise RuntimeError(
                "No samples were scored this round; nothing to update D_t."
            )
        per_sample_resid = global_resid / global_count.clamp_min(1.0)
        mean_seen = per_sample_resid[seen].mean()
        L_raw = torch.where(seen, per_sample_resid, mean_seen)

        L_max = L_raw.max().clamp_min(1e-12)
        L = L_raw / L_max

        L_bar = float(((self._weights / self._n) * L).sum().item())
        L_bar_safe = min(max(L_bar, 1e-6), 1.0 - 1e-6)
        beta = L_bar_safe / (1.0 - L_bar_safe)
        # Sign-safe clip — see __init__ docstring.
        beta = min(beta, self._beta_max)

        D_prime = self._weights * (beta ** (1.0 - L))

        new_weights = (
            1.0 - self._ema_lambda
        ) * self._weights + self._ema_lambda * D_prime
        new_weights = self._renormalize(new_weights)
        self._weights = new_weights

        return _R2Stats(
            L_bar=L_bar,
            beta=beta,
            max_weight=float(self._weights.max().item()),
            min_weight=float(self._weights.min().item()),
            entropy_bits=self._entropy_bits(),
        )

    def _renormalize(self, w: torch.Tensor) -> torch.Tensor:
        w = w.clamp_min(1e-12)
        w_max = float(w.max().item())
        floor = w_max / self._clamp_limit
        w = w.clamp_min(floor)
        w = w * (self._n / w.sum())
        return w

    def _entropy_bits(self) -> float:
        p = self._weights / self._n
        log2p = torch.log2(p.clamp_min(1e-12))
        return float(-(p * log2p).sum().item())

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "weights": self._weights.detach().cpu(),
            "ema_lambda": self._ema_lambda,
            "clamp_limit": self._clamp_limit,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["weights"].numel() != self._n:
            raise ValueError(
                f"checkpoint has {state['weights'].numel()} weights but "
                f"this run expects {self._n}"
            )
        self._weights = state["weights"].to(self._device).float()


# -----------------------------------------------------------------------------
# Per-sample loss helper
# -----------------------------------------------------------------------------


def _per_sample_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, label_mask: torch.Tensor
) -> torch.Tensor:
    """Per-sample masked MSE, shape `(B,)`."""
    wet = label_mask.to(device=pred.device)
    pred = pred * wet
    target = target * wet
    return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))


# -----------------------------------------------------------------------------
# PCGB driver
# -----------------------------------------------------------------------------


class PCGB(Trainer):
    """Path-Cyclic Gradient Boosting driver.

    Subclasses :class:`Trainer` to inherit DDP, optimizer, scheduler,
    wandb, and ckpt-path machinery, then replaces the per-epoch loop in
    :meth:`run` with a per-round loop that:

    1. Runs ``K = round(steps_per_round_epochs * len(train_loader))`` SGD
       steps under the current deterministic path-mask, with per-sample
       loss weights drawn from ``D_t``.
    2. Streams a scoring subset of the calibration set, accumulating
       both per-sample residuals (under the *unmasked* network) and
       per-mask weighted MSE for the round's candidate set in one pass.
    3. Updates ``D_t`` via R2-proper, updates the searcher, and selects
       ``M_{t+1}`` via the searcher.

    EMA tracking is intentionally disabled — under PCGB's mask cycling,
    the optimization trajectory changes regime each round, so a single
    EMA over the whole trajectory averages over qualitatively different
    optima.
    """

    cfg: PCGBConfig  # narrowed type

    def __init__(self, cfg: PCGBConfig) -> None:
        if cfg.test_using_ema:
            logger.info("PCGB: forcing test_using_ema=False (EMA disabled).")
            cfg = cfg.model_copy(update={"test_using_ema": False})

        # Force single-step training — PCGB's per-sample-weighted loss is
        # defined for one-step regression.
        if cfg.steps != [1]:
            logger.info(f"PCGB: forcing steps=[1] (was {cfg.steps}).")
            cfg = cfg.model_copy(update={"steps": [1], "step_transition": []})

        super().__init__(cfg)
        self.cfg = cfg

        candidate: object = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if not isinstance(candidate, Samudra):
            raise TypeError(
                f"PCGB requires a Samudra model with a UNetBackbone; got "
                f"{type(candidate).__name__}"
            )
        # `candidate` is now narrowed to Samudra by the isinstance check.
        self._inner_model: Samudra = candidate
        self.backbone = self._inner_model.unet

        # Trainer.run() lazily initializes loaders per-epoch (train.py:436).
        # PCGB overrides run() with a per-round loop and sizes SampleWeights
        # by the train-set length below, so we need train_loader available
        # in __init__. Steps are forced to [1] above, so cur_step is 1.
        self.init_data_loaders(self.get_current_step(self.start_epoch))

        # MaskSearcher — v1 default is EnumerateSearcher(num_skips=4).
        self.searcher: MaskSearcher = build_searcher(cfg.mask_searcher)
        if self.searcher.num_skips != self.backbone.num_steps:
            raise ValueError(
                f"searcher.num_skips ({self.searcher.num_skips}) does not "
                f"match backbone num_steps ({self.backbone.num_steps})."
            )
        if self.searcher.num_blocks not in (0, self.backbone.num_blocks):
            raise ValueError(
                f"searcher.num_blocks ({self.searcher.num_blocks}) must be "
                f"either 0 (skip-only) or {self.backbone.num_blocks} (full "
                f"path lattice)."
            )
        logger.info(
            f"PCGB searcher: {type(self.searcher).__name__} "
            f"(num_skips={self.searcher.num_skips}, "
            f"num_blocks={self.searcher.num_blocks})"
        )

        # When block_drops can be active, dropping a CoreBlock's residual
        # (block_drops[i]=True) bypasses that block's trunk so its params
        # don't receive grad. DDP's reducer raises unless we rewrap with
        # find_unused_parameters=True. Skip-only masks don't hit this:
        # skip drops zero the skip tensor but every param still
        # participates in the forward pass.
        if self.searcher.num_blocks > 0 and self.distributed is not None:
            from torch import nn

            inner = self.model.module if hasattr(self.model, "module") else self.model
            self.model = nn.parallel.DistributedDataParallel(
                inner,
                device_ids=[self.distributed.gpu],
                find_unused_parameters=True,
            )
            logger.info(
                "PCGB: rewrapped DDP with find_unused_parameters=True "
                "(searcher uses block_drops)."
            )

        # Per-sample weights — sized to the train dataset.
        n_train = self._n_train_samples()
        self.sample_weights = SampleWeights(
            n_samples=n_train,
            device=self.device,
            ema_lambda=cfg.reweight_ema_lambda,
            clamp_limit=cfg.reweight_clamp_limit,
            beta_max=cfg.reweight_beta_max,
        )
        # log2(N) — used as the entropy ceiling for the D-collapse warning.
        import math

        self._max_entropy_bits = math.log2(n_train) if n_train > 1 else 1.0
        logger.info(
            f"PCGB: tracking D_t over {n_train} train samples "
            f"(reweight_enabled={cfg.reweight_enabled})."
        )

        # Calibration scoring loader — fixed subset of the calibration period.
        self.scoring_loader = self._build_scoring_loader()

        self._round_metrics: list[dict] = []
        self._mask_history: list[PathMask] = []
        self._current_mask: PathMask = all_kept(
            self.searcher.num_skips, self.searcher.num_blocks
        )

        self.steps_per_round = max(
            1,
            int(round(cfg.steps_per_round_epochs * len(self.train_loader))),
        )
        logger.info(
            f"PCGB: T={cfg.num_rounds} rounds × K={self.steps_per_round} "
            f"steps/round (≈{cfg.steps_per_round_epochs:.2f} epoch/round)."
        )

    # ------------------------------------------------------------------
    # Loader plumbing
    # ------------------------------------------------------------------

    def _n_train_samples(self) -> int:
        ds = self.train_loader.dataset
        if hasattr(ds, "datasets"):
            return sum(len(c) for c in ds.datasets)
        return len(ds)

    def _build_scoring_loader(self) -> TrainDataLoader:
        cfg = self.cfg
        src = self.primary_src.slice(cfg.calibration_time)
        ds = TorchTrainDataset(
            src=src,
            dst=None,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            hist=cfg.data.hist,
            steps=1,
            normalize_before_mask=cfg.data.normalize_before_mask,
            masked_fill_value=cfg.data.masked_fill_value,
            stride=1,
            concurrent_compute_=cfg.data.concurrent_compute,
        )
        n_total = len(ds)
        n_keep = max(1, int(round(n_total * cfg.scoring_subset_percent)))
        if n_keep < n_total:
            g = torch.Generator()
            g.manual_seed(cfg.experiment.rand_seed + 7919)
            indices = torch.randperm(n_total, generator=g)[:n_keep].tolist()
            indices.sort()
            subset: torch.utils.data.Dataset = Subset(ds, indices)
        else:
            subset = ds

        sampler: torch.utils.data.Sampler | None = None
        if self.distributed is not None:
            sampler = DistributedSampler(
                subset,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=False,
                drop_last=False,
            )

        loader = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_mem,
            collate_fn=collate_raw_train_data,
            multiprocessing_context=self.mp_context,
            drop_last=False,
        )
        logger.info(
            f"PCGB: scoring loader = {n_keep}/{n_total} samples of the "
            f"calibration period."
        )
        return TrainDataLoader(loader, [ds], self.device)

    # ------------------------------------------------------------------
    # Forward helpers — go through the DDP wrapper for gradient sync
    # ------------------------------------------------------------------

    def _forward_step_pred(self, data: TrainData) -> torch.Tensor:
        outputs = self.model(data)
        return outputs[0]

    def _enter_mask(self, mask: PathMask | None):
        """Backbone context for ``mask`` (None → unmasked / deployed network)."""
        block_drops = mask.block_drops if (mask and mask.block_drops) else None
        skip_drops = mask.skip_drops if mask is not None else None
        return self.backbone.with_path_mask(
            skip_drops=skip_drops, block_drops=block_drops
        )

    # ------------------------------------------------------------------
    # Per-round training
    # ------------------------------------------------------------------

    def _train_round(
        self, round_idx: int, mask: PathMask
    ) -> tuple[dict[str, float], Metrics]:
        """Run K SGD steps under ``mask`` with per-sample reweighting.

        Returns a (pcgb_scalars, train_aggregator_logs) pair. The aggregator
        logs use the same ``train/<var>`` key scheme as standard training, so
        plots line up with baseline runs.
        """
        self.model.train(True)
        self.optimizer.zero_grad()

        loss_running = 0.0
        n_steps = 0
        loader_iter: Iterator[TrainData] = self._cycling_train_iter()
        train_aggregator = Aggregator.get_train_aggregator(self.tensor_map)

        with self._enter_mask(mask):
            for k in range(self.steps_per_round):
                data = next(loader_iter)
                if data.sample_indices is None:
                    raise RuntimeError(
                        "PCGB requires sample_indices on each TrainData; "
                        "ensure the train loader uses the indexed collate."
                    )

                pred = self._forward_step_pred(data)
                label = data.get_label(0)
                per_sample = _per_sample_masked_mse(pred, label, data.ctx.label_mask)
                weights = self.sample_weights.lookup(data.sample_indices)
                loss = (weights * per_sample).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Per-channel masked MSE for the aggregator — same shape as
                # `decomposed_mse` so TrainAggregator can break out by
                # variable/depth. Detached: aggregator metrics never feed
                # back into autograd.
                with torch.no_grad():
                    wet = data.ctx.label_mask.to(device=pred.device)
                    loss_per_channel = F.mse_loss(
                        pred * wet, label * wet, reduction="none"
                    ).mean(dim=(0, 2, 3))
                train_aggregator.record_batch(
                    TrainBatchOutput(loss.detach(), loss_per_channel)
                )

                loss_running += float(loss.detach().item())
                n_steps += 1
                self.num_batches_seen += 1

                if self.is_wandb_enabled() and is_main_process() and (k % 50 == 0):
                    self.wandb_logger.log(
                        {
                            "pcgb/batch/loss_weighted": float(loss.detach().item()),
                            "pcgb/batch/round": round_idx,
                        },
                        step=self.num_batches_seen,
                    )

        pcgb_scalars = {"loss_weighted_mean": loss_running / max(n_steps, 1)}
        return pcgb_scalars, train_aggregator.get_logs("train")

    def _cycling_train_iter(self) -> Iterator[TrainData]:
        epoch = 0
        while True:
            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(self.num_batches_seen + epoch)
            yield from self.train_loader
            epoch += 1

    # ------------------------------------------------------------------
    # End-of-round validation (unmasked network)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_round(self, round_idx: int) -> Metrics:
        """Full validation pass under the *unmasked* (deployed) network.

        Mirrors ``Trainer.validate_one_epoch`` (train.py:677) so the
        ``val/<var>`` keys match standard training and plot directly
        against the baseline reference run.
        """
        self.model.eval()
        val_aggregator = Aggregator.get_validation_aggregator(
            self.primary_src.metadata,
            self.hist,
            self.primary_src.spherical_area_weights.to(self.device),
            self.num_out,
            self.tensor_map,
            self.normalize,
            include_image_aggregators=False,
        )
        metric_logger = MetricLogger(delimiter="  ")
        header = f"PCGB Validation [round {round_idx}]"

        with self._test_context(), self._enter_mask(None):
            for data in metric_logger.log_every(self.val_loader, 1, header):
                vo = validate_batch(self.model, data, self.loss_fn)
                val_aggregator.record_validation_batch(vo)
                metric_logger.update(loss=vo.loss)

        return val_aggregator.get_logs(label="val")

    # ------------------------------------------------------------------
    # End-of-round scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_round(
        self, candidates: list[PathMask]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[PathMask, float]]]:
        """One streaming pass over the scoring subset.

        For each batch:
          * Forward unmasked → contribute to per-sample residuals.
          * Forward under each candidate mask → accumulate per-mask
            weighted MSE.

        Returns:
          local_indices, local_residuals: rank-local pieces of the
            unmasked residuals (SampleWeights.update will all-reduce).
          scored: list of (mask, weighted_MSE) pairs, averaged across
            ranks. Same length and order as ``candidates``.
        """
        self.model.eval()

        n_masks = len(candidates)
        score_sum = torch.zeros(n_masks, dtype=torch.float64, device=self.device)
        score_count = torch.zeros(n_masks, dtype=torch.float64, device=self.device)
        all_indices: list[torch.Tensor] = []
        all_resid: list[torch.Tensor] = []

        for data in self.scoring_loader:
            assert data.sample_indices is not None
            label = data.get_label(0)
            weights = self.sample_weights.lookup(data.sample_indices)

            pred_unmasked = self._forward_step_pred(data)
            per_sample_unmasked = _per_sample_masked_mse(
                pred_unmasked, label, data.ctx.label_mask
            )
            all_indices.append(data.sample_indices.detach())
            all_resid.append(per_sample_unmasked.sqrt().detach())

            for m_idx, mask in enumerate(candidates):
                with self._enter_mask(mask):
                    pred_m = self._forward_step_pred(data)
                per_sample_m = _per_sample_masked_mse(
                    pred_m, label, data.ctx.label_mask
                )
                score_sum[m_idx] += (weights * per_sample_m).sum(dtype=torch.float64)
                score_count[m_idx] += per_sample_m.shape[0]

        if get_world_size() > 1:
            dist.all_reduce(score_sum)
            dist.all_reduce(score_count)
        scores = (score_sum / score_count.clamp_min(1.0)).cpu().tolist()
        scored = list(zip(candidates, scores))

        if all_indices:
            local_indices = torch.cat(all_indices)
            local_resid = torch.cat(all_resid)
        else:
            local_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            local_resid = torch.zeros(0, device=self.device)

        return local_indices, local_resid, scored

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            f"PCGB starting: T={self.cfg.num_rounds}, "
            f"K={self.steps_per_round}, "
            f"searcher={type(self.searcher).__name__}"
        )
        start = time.perf_counter()

        for t in range(1, self.cfg.num_rounds + 1):
            t_start = time.perf_counter()
            mask = self._current_mask
            self._mask_history.append(mask)

            train_metrics, train_agg_logs = self._train_round(t, mask)

            candidates = self.searcher.candidates_for_round(t)
            local_idx, local_resid, scored = self._score_round(candidates)

            # Full validation pass on cadence. Uses the *unmasked* network
            # — what we ship at inference — under the standard val
            # aggregator, so `val/<var>` keys line up with baseline runs.
            run_val = (
                t % self.cfg.validate_every_n_rounds == 0 or t == self.cfg.num_rounds
            )
            val_agg_logs = self._validate_round(t) if run_val else {}

            if self.cfg.reweight_enabled:
                r2_stats = self.sample_weights.update(local_idx, local_resid)
            else:
                # Ablation arm: skip the R2 update so D stays at 1 forever.
                # `sample_weights.lookup` keeps returning 1.0 for every
                # sample, making the SGD loss equivalent to plain mean MSE.
                # We still produce a stats record (with zero β) so the
                # logged metric schema is identical across runs.
                r2_stats = _R2Stats(
                    L_bar=0.0,
                    beta=0.0,
                    max_weight=1.0,
                    min_weight=1.0,
                    entropy_bits=self._max_entropy_bits,
                )

            self.searcher.update(scored)
            next_mask = self.searcher.select_next(scored, self._mask_history, t)
            self._current_mask = next_mask

            argmax_score = max(s for _, s in scored)
            metrics = {
                "round": t,
                "mask_played": str(mask),
                "mask_next": str(next_mask),
                "L_bar": r2_stats.L_bar,
                "beta": r2_stats.beta,
                "D_max": r2_stats.max_weight,
                "D_min": r2_stats.min_weight,
                "D_entropy_bits": r2_stats.entropy_bits,
                "loss_weighted_train_mean": train_metrics["loss_weighted_mean"],
                "n_candidates_scored": len(scored),
                "argmax_mask_score": argmax_score,
                "wall_seconds": time.perf_counter() - t_start,
            }
            self._round_metrics.append(metrics)

            # D-collapse warning — entropy ratio below 80% of uniform means
            # the reweighting has concentrated on a small fraction of
            # samples, which is the "Arctic-2018 collapse" failure mode.
            entropy_ratio = r2_stats.entropy_bits / max(self._max_entropy_bits, 1e-9)
            if self.cfg.reweight_enabled and entropy_ratio < 0.8 and is_main_process():
                logger.warning(
                    f"PCGB: D_t entropy at {entropy_ratio:.1%} of uniform "
                    f"(H={r2_stats.entropy_bits:.2f} / max={self._max_entropy_bits:.2f}) — "
                    f"sample reweighting may be collapsing onto a narrow "
                    f"subset. Consider tightening reweight_clamp_limit."
                )

            if is_main_process():
                logger.info(
                    f"[round {t}/{self.cfg.num_rounds}] "
                    f"played={metrics['mask_played']} "
                    f"next={metrics['mask_next']} "
                    f"L̄={metrics['L_bar']:.4f} β={metrics['beta']:.4f} "
                    f"H(D)={metrics['D_entropy_bits']:.3f}b "
                    f"({metrics['wall_seconds']:.0f}s)"
                )
                if self.is_wandb_enabled():
                    log_payload: dict = {
                        f"pcgb/round/{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }
                    log_payload.update(train_agg_logs)
                    log_payload.update(val_agg_logs)
                    self.wandb_logger.log(
                        log_payload,
                        step=self.num_batches_seen,
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            if t % self.cfg.save_round_freq == 0 or t == self.cfg.num_rounds:
                self._save_round_ckpt(t)

        if is_main_process():
            self._save_final_ckpt()
            self._save_round_metrics()

        total = time.perf_counter() - start
        logger.info(
            f"PCGB complete in {datetime.timedelta(seconds=int(total))}; "
            f"output: {self.output_dir}"
        )
        self.finish()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save_round_ckpt(self, round_idx: int) -> None:
        if not is_main_process():
            return
        path = self.nets_dir / f"pcgb_round_{round_idx:03d}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "sample_weights": self.sample_weights.state_dict(),
                "round": round_idx,
                "mask_history": [str(m) for m in self._mask_history],
                "current_mask": str(self._current_mask),
            },
            path,
        )
        logger.info(f"Saved per-round checkpoint to {path}")

    def _save_final_ckpt(self) -> None:
        path = self.nets_dir / "pcgb_final.pt"
        torch.save({"model": self.model.state_dict()}, path)
        logger.info(f"Saved final PCGB checkpoint to {path}")

    def _save_round_metrics(self) -> None:
        path = self.output_dir / "round_metrics.json"
        path.write_text(json.dumps(self._round_metrics, indent=2))
        logger.info(f"Wrote round_metrics to {path}")


def main() -> None:
    from ocean_emulators.utils.logging import handle_logging, handle_warnings

    cfg = PCGBConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    try:
        PCGB(cfg).run()
    except Exception:
        logger.exception("PCGB failed")
        raise


if __name__ == "__main__":
    main()
