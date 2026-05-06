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
"""

from __future__ import annotations

import datetime
import itertools
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ocean_emulators.config import MaskPoolConfig, PCGBConfig
from ocean_emulators.datasets import TorchTrainDataset, TrainData, TrainDataLoader
from ocean_emulators.models import Samudra
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import get_world_size, is_main_process
from ocean_emulators.utils.train import collate_raw_train_data

logger = logging.getLogger(__name__)


SkipMask = tuple[bool, ...]


# -----------------------------------------------------------------------------
# Mask pool helpers
# -----------------------------------------------------------------------------


def build_mask_pool(cfg: MaskPoolConfig) -> list[SkipMask]:
    """Generate the candidate skip-mask pool for boosting rounds."""
    if cfg.mode == "enumerate_all":
        return [
            tuple(bits)
            for bits in itertools.product([False, True], repeat=cfg.num_skips)
        ]
    raise ValueError(f"Unknown mask pool mode: {cfg.mode}")


def format_mask(mask: SkipMask) -> str:
    """Compact human-readable mask: '0101' = keep,drop,keep,drop."""
    return "".join("1" if drop else "0" for drop in mask)


def all_skips_kept(num_skips: int) -> SkipMask:
    """Round-1 default mask: every skip kept (no drops)."""
    return (False,) * num_skips


def pick_next_mask(
    mask_pool: list[SkipMask],
    mask_scores: torch.Tensor,
    schedule: str,
    history: list[SkipMask],
    no_repeat_window: int,
) -> SkipMask:
    """Choose the next round's mask given per-mask scores and schedule.

    Pulled out as a free function so it can be unit-tested in isolation.

    Args:
      mask_pool: ordered list of candidate masks; ``mask_scores[i]`` is
        the weighted MSE for ``mask_pool[i]``.
      mask_scores: (n_masks,) tensor of scores. Adversarial selection
        picks the argmax (the mask currently performing *worst* on the
        reweighted distribution — most slack to recover).
      schedule: "adversarial" or "round_robin".
      history: list of masks already played, oldest first. Length
        equals the number of completed rounds.
      no_repeat_window: if > 0 (and schedule="adversarial"), exclude the
        last ``no_repeat_window`` distinct masks from the argmax pool.
        If the window excludes everything, fall back to the global
        argmax.
    """
    if schedule == "round_robin":
        return mask_pool[len(history) % len(mask_pool)]

    excluded: set[SkipMask] = set()
    if no_repeat_window > 0:
        excluded.update(history[-no_repeat_window:])

    best_idx = -1
    best_score = -float("inf")
    for i, mask in enumerate(mask_pool):
        if mask in excluded:
            continue
        s = float(mask_scores[i].item())
        if s > best_score:
            best_score = s
            best_idx = i

    if best_idx < 0:
        best_idx = int(mask_scores.argmax().item())
    return mask_pool[best_idx]


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
    thousands at most for ocean emulators, tens of KB total). The tensor
    is identical across ranks at all times: per-round residual
    contributions are accumulated via :func:`torch.distributed.all_reduce`
    sum before the R2 update is applied, and the update itself is
    deterministic given full residuals so all ranks compute the same
    new D_{t+1}.

    Reweighting is AdaBoost.R2-proper (Drucker, 1997) with two stability
    additions lifted from :class:`ocean_emulators.utils.loss.DynamicLoss`:

      * EMA smoothing — `D_{t+1} = (1-λ) · D_t + λ · D_R2` — to avoid
        per-round whipsaw when a single round's residuals happen to spike.
      * Max/min clamp — bounds the dynamic range of D_t to prevent
        collapse onto a tiny set of "hardest" samples.

    The R2 *abort-on-failure* rule (L̄ ≥ 0.5 → stop) is **not** enforced
    here; cold-start with random init can push L̄ above 0.5 in early
    rounds without the procedure being broken. We log L̄ every round so
    pathological cases are visible.
    """

    def __init__(
        self,
        n_samples: int,
        device: torch.device,
        ema_lambda: float,
        clamp_limit: float,
    ):
        if not (0.0 < ema_lambda <= 1.0):
            raise ValueError(f"ema_lambda must be in (0, 1], got {ema_lambda}")
        if clamp_limit <= 1.0:
            raise ValueError(f"clamp_limit must be > 1, got {clamp_limit}")
        self._weights = torch.ones(n_samples, device=device)
        self._device = device
        self._n = n_samples
        self._ema_lambda = ema_lambda
        self._clamp_limit = clamp_limit

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def __len__(self) -> int:
        return self._n

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-sample weight multipliers for a batch's sample indices.

        Returned tensor matches `indices.shape` and lives on the weights'
        device. Used in the SGD step as a per-sample loss multiplier.
        """
        idx = indices.to(self._device, dtype=torch.long, non_blocking=True)
        return self._weights.index_select(0, idx)

    def update(
        self,
        local_indices: torch.Tensor,
        local_residuals: torch.Tensor,
    ) -> _R2Stats:
        """Apply R2-proper update from this rank's residual contributions.

        Args:
          local_indices: 1D tensor of sample indices this rank computed
            residuals for during the end-of-round scoring pass. Disjoint
            across ranks (DDP shards the calibration set non-overlappingly).
          local_residuals: 1D tensor of |y - F(x)| values, same shape as
            ``local_indices``, in the same order.

        Returns:
          A :class:`_R2Stats` for round-level logging.
        """
        if local_indices.shape != local_residuals.shape:
            raise ValueError(
                f"indices/residuals shape mismatch: {local_indices.shape} "
                f"vs {local_residuals.shape}"
            )

        # Scatter local residuals into a length-N buffer; ranks contribute
        # to disjoint indices, so all_reduce(sum) gives the global vector.
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

        # Samples never seen by any rank fall back to the global mean (so
        # they don't get artificially down-weighted by zero residual).
        seen = global_count > 0
        if not bool(seen.any()):
            raise RuntimeError(
                "No samples were scored this round; nothing to update D_t."
            )
        per_sample_resid = global_resid / global_count.clamp_min(1.0)
        mean_seen = per_sample_resid[seen].mean()
        L_raw = torch.where(seen, per_sample_resid, mean_seen)

        # R2 step 1: range-normalize losses to [0, 1].
        L_max = L_raw.max().clamp_min(1e-12)
        L = L_raw / L_max

        # R2 step 2: weighted mean loss (D is mean-1, so dividing by N
        # gives the proper expectation).
        L_bar = float(((self._weights / self._n) * L).sum().item())

        # R2 step 3: confidence β_t. Abort-on-failure intentionally not
        # enforced for cold-start; logged so the user can see it.
        L_bar_safe = min(max(L_bar, 1e-6), 1.0 - 1e-6)
        beta = L_bar_safe / (1.0 - L_bar_safe)

        # R2 step 4: D' ∝ D · β_t^(1 - L_i). Easy samples (L≈0) get
        # multiplied by β_t (typically <1, so down-weighted); hard
        # samples (L≈1) are multiplied by 1 (unchanged).
        D_prime = self._weights * (beta ** (1.0 - L))

        # EMA smoothing — lifted from DynamicLoss.
        new_weights = (
            1.0 - self._ema_lambda
        ) * self._weights + self._ema_lambda * D_prime

        # Clamp dynamic range, then renormalize so mean(D) = 1.
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
        """Apply the max/min clamp and renormalize so mean(w) = 1."""
        w = w.clamp_min(1e-12)
        # Bound the ratio max/min by raising the floor — same shape as
        # DynamicLoss._limit, applied to the per-sample axis.
        w_max = float(w.max().item())
        floor = w_max / self._clamp_limit
        w = w.clamp_min(floor)
        w = w * (self._n / w.sum())
        return w

    def _entropy_bits(self) -> float:
        # Treat D/N as a probability measure (Σ_i D_i/N = 1).
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
        # Hyperparameters travel with the run config, not the checkpoint.


# -----------------------------------------------------------------------------
# Per-sample loss helper
# -----------------------------------------------------------------------------


def _per_sample_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, label_mask: torch.Tensor
) -> torch.Tensor:
    """Per-sample masked MSE, shape `(B,)`.

    Mirrors :func:`ocean_emulators.utils.loss.decomposed_mse` but reduces
    over `(C, H, W)` instead of `(0, H, W)` so each sample's loss can be
    multiplied by its `D_t(i)` weight before the batch reduction.
    """
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
       steps under the current deterministic skip-mask, with per-sample
       loss weights drawn from `D_t`.
    2. Streams a scoring subset of the calibration set, accumulating
       both per-sample residuals (under the *unmasked* network) and
       per-mask weighted MSE scores in one pass.
    3. Updates `D_t` via R2-proper and selects `M_{t+1}` via argmax of
       the per-mask scores (or round-robin if so configured).

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
        # defined for one-step regression. Multi-step rollout would need a
        # different reweighting target.
        if cfg.steps != [1]:
            logger.info(f"PCGB: forcing steps=[1] (was {cfg.steps}).")
            cfg = cfg.model_copy(update={"steps": [1], "step_transition": []})

        super().__init__(cfg)
        self.cfg = cfg

        # Resolve the underlying Samudra (Trainer wraps in DDP).
        self._inner_model: Samudra = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if not isinstance(self._inner_model, Samudra):
            raise TypeError(
                f"PCGB requires a Samudra model with a UNetBackbone; got "
                f"{type(self._inner_model).__name__}"
            )
        self.backbone = self._inner_model.unet

        # Mask pool + sanity check vs. backbone topology.
        self.mask_pool: list[SkipMask] = build_mask_pool(cfg.mask_pool)
        if self.backbone.num_steps != cfg.mask_pool.num_skips:
            raise ValueError(
                f"mask_pool.num_skips ({cfg.mask_pool.num_skips}) does not "
                f"match backbone num_steps ({self.backbone.num_steps})."
            )

        # Per-sample weights — sized to the train dataset.
        n_train = self._n_train_samples()
        self.sample_weights = SampleWeights(
            n_samples=n_train,
            device=self.device,
            ema_lambda=cfg.reweight_ema_lambda,
            clamp_limit=cfg.reweight_clamp_limit,
        )
        logger.info(f"PCGB: tracking D_t over {n_train} train samples.")

        # Calibration scoring loader — fixed subset of the calibration
        # period, sampled once at startup so residuals/scores are computed
        # on a stable set across rounds.
        self.scoring_loader = self._build_scoring_loader()

        self._round_metrics: list[dict] = []
        self._mask_history: list[SkipMask] = []
        self._current_mask: SkipMask = all_skips_kept(self.backbone.num_steps)

        # K — total SGD steps per round, derived from epoch fraction.
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
        # ConcatDataset path — sum component lengths.
        if hasattr(ds, "datasets"):
            return sum(len(c) for c in ds.datasets)
        return len(ds)

    def _build_scoring_loader(self) -> TrainDataLoader:
        """Fixed-shard, single-step DataLoader over the scoring subset.

        The scoring subset is `scoring_subset_percent` of the calibration
        period. Sampled deterministically (seeded random subset) at
        startup; identical across rounds. Shards across DDP ranks via
        :class:`DistributedSampler`.
        """
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
        """Single-step prediction via the (possibly DDP-wrapped) model."""
        outputs = self.model(data)  # list[Tensor] of length steps==1
        return outputs[0]

    # ------------------------------------------------------------------
    # Per-round training
    # ------------------------------------------------------------------

    def _train_round(self, round_idx: int, mask: SkipMask) -> dict[str, float]:
        """K SGD steps under the fixed `mask` with per-sample weighted MSE."""
        self.model.train(True)
        self.optimizer.zero_grad()

        loss_running = 0.0
        n_steps = 0

        loader_iter: Iterator[TrainData] = self._cycling_train_iter()
        for k in range(self.steps_per_round):
            data = next(loader_iter)
            if data.sample_indices is None:
                raise RuntimeError(
                    "PCGB requires sample_indices on each TrainData; "
                    "ensure the train loader uses the indexed collate."
                )

            with self.backbone.with_skip_mask(mask):
                pred = self._forward_step_pred(data)

            label = data.get_label(0)
            per_sample = _per_sample_masked_mse(pred, label, data.ctx.label_mask)
            weights = self.sample_weights.lookup(data.sample_indices)
            loss = (weights * per_sample).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_running += float(loss.detach().item())
            n_steps += 1
            self.num_batches_seen += 1

            if self.is_wandb_enabled() and is_main_process() and (k % 50 == 0):
                self.wandb_logger.log(
                    {
                        "pcgb/batch/loss_weighted": float(loss.detach().item()),
                        "pcgb/batch/round": round_idx,
                        "pcgb/batch/mask": int(sum(b << i for i, b in enumerate(mask))),
                    },
                    step=self.num_batches_seen,
                )

        return {"loss_weighted_mean": loss_running / max(n_steps, 1)}

    def _cycling_train_iter(self) -> Iterator[TrainData]:
        """Infinite iterator over the train loader, re-shuffled on epoch wrap."""
        epoch = 0
        while True:
            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(self.num_batches_seen + epoch)
            yield from self.train_loader
            epoch += 1

    # ------------------------------------------------------------------
    # End-of-round scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_round(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming pass over the scoring subset.

        Returns:
          local_indices: (n_local,) sample idx scored on this rank.
          local_residuals: (n_local,) per-sample |y - F_unmasked(x)|.
          mask_scores: (n_masks,) global per-mask weighted MSE
            (already all-reduced across ranks).
        """
        self.model.eval()

        all_indices: list[torch.Tensor] = []
        all_resid: list[torch.Tensor] = []
        n_masks = len(self.mask_pool)
        mask_score_sum = torch.zeros(n_masks, dtype=torch.float64, device=self.device)
        mask_score_count = torch.zeros(n_masks, dtype=torch.float64, device=self.device)

        for data in self.scoring_loader:
            assert data.sample_indices is not None
            label = data.get_label(0)
            weights = self.sample_weights.lookup(data.sample_indices)

            # Unmasked residual — drives the reweighting target.
            pred_unmasked = self._forward_step_pred(data)
            per_sample_unmasked = _per_sample_masked_mse(
                pred_unmasked, label, data.ctx.label_mask
            )
            all_indices.append(data.sample_indices.detach())
            all_resid.append(per_sample_unmasked.sqrt().detach())  # |r| not r²

            # Per-mask weighted MSE — drives the adversarial selection.
            for m_idx, mask in enumerate(self.mask_pool):
                with self.backbone.with_skip_mask(mask):
                    pred_m = self._forward_step_pred(data)
                per_sample_m = _per_sample_masked_mse(
                    pred_m, label, data.ctx.label_mask
                )
                mask_score_sum[m_idx] += (weights * per_sample_m).sum(
                    dtype=torch.float64
                )
                mask_score_count[m_idx] += per_sample_m.shape[0]

        # All-reduce mask scores across ranks → identical on every rank.
        if get_world_size() > 1:
            dist.all_reduce(mask_score_sum)
            dist.all_reduce(mask_score_count)
        mask_scores = mask_score_sum / mask_score_count.clamp_min(1.0)

        if all_indices:
            local_indices = torch.cat(all_indices)
            local_resid = torch.cat(all_resid)
        else:
            local_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            local_resid = torch.zeros(0, device=self.device)

        return local_indices, local_resid, mask_scores

    def _pick_next_mask(self, mask_scores: torch.Tensor) -> SkipMask:
        """Method wrapper around :func:`pick_next_mask`."""
        return pick_next_mask(
            mask_pool=self.mask_pool,
            mask_scores=mask_scores,
            schedule=self.cfg.mask_schedule,
            history=self._mask_history,
            no_repeat_window=self.cfg.mask_no_repeat_window,
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            f"PCGB starting: T={self.cfg.num_rounds}, K={self.steps_per_round}, "
            f"mask_schedule={self.cfg.mask_schedule}, "
            f"#masks={len(self.mask_pool)}"
        )
        start = time.perf_counter()

        for t in range(1, self.cfg.num_rounds + 1):
            t_start = time.perf_counter()
            mask = self._current_mask
            self._mask_history.append(mask)

            train_metrics = self._train_round(t, mask)

            local_idx, local_resid, mask_scores = self._score_round()
            r2_stats = self.sample_weights.update(local_idx, local_resid)

            next_mask = self._pick_next_mask(mask_scores)
            self._current_mask = next_mask

            # Scheduler steps once per round (Trainer convention is once per
            # epoch — we treat a round as an epoch). Configure the scheduler
            # in YAML with epochs == num_rounds so the schedule lines up.
            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {
                "round": t,
                "mask_played": format_mask(mask),
                "mask_next": format_mask(next_mask),
                "L_bar": r2_stats.L_bar,
                "beta": r2_stats.beta,
                "D_max": r2_stats.max_weight,
                "D_min": r2_stats.min_weight,
                "D_entropy_bits": r2_stats.entropy_bits,
                "loss_weighted_train_mean": train_metrics["loss_weighted_mean"],
                "mask_scores": {
                    format_mask(m): float(mask_scores[i].item())
                    for i, m in enumerate(self.mask_pool)
                },
                "wall_seconds": time.perf_counter() - t_start,
            }
            self._round_metrics.append(metrics)

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
                    self.wandb_logger.log(
                        {
                            f"pcgb/round/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        },
                        step=self.num_batches_seen,
                    )

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
                "mask_history": [format_mask(m) for m in self._mask_history],
                "current_mask": format_mask(self._current_mask),
            },
            path,
        )
        logger.info(f"Saved per-round checkpoint to {path}")

    def _save_final_ckpt(self) -> None:
        path = self.nets_dir / "pcgb_final.pt"
        # Drop-in replacement for any Samudra ckpt — same key layout that
        # eval.py expects: {"model": state_dict}.
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
