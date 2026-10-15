# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""C's stochastic U-Net with explicit known/requested fields at fixed time slots."""

from typing import Literal

import torch
from score_model import ScoreModel, ScoreModelConfig, crps
from torch import nn

from samudra.config import SamudraConfig
from samudra.utils.device import autocast

TASKS = ("forecast", "surface", "variable", "gap")
TASK_PROBABILITIES = (0.5, 0.25, 0.125, 0.125)


def task_masks(task, wet, generator):
    """Same-time reconstruction masks shared within a rank minibatch.

    No history is supplied for reconstruction. Two target slots are distinct
    channels. Masked values never enter conditioning; only requested wet entries
    enter the loss. Gap centers are sampled from wet top-level cells.
    """
    wet = wet.bool()
    known = torch.zeros_like(wet)
    if task == "forecast":
        return wet, known, wet
    if task == "surface":
        known[[38, 76, 77 + 38, 77 + 76]] = wet[[38, 76, 77 + 38, 77 + 76]]
    elif task == "variable":
        family = int(torch.randint(5, (), generator=generator))
        missing = list(range(19 * family, 19 * (family + 1))) if family < 4 else [76]
        known = wet.clone()
        known[missing + [i + 77 for i in missing]] = False
    elif task == "gap":
        h, w = wet.shape[-2:]
        cells = wet[0].nonzero().cpu()
        y, x = cells[int(torch.randint(len(cells), (), generator=generator))].tolist()
        height, width = max(1, h // 4), max(1, w // 4)
        bottom = min(max(0, y - height // 2), h - height)
        columns = (torch.arange(width, device=wet.device) + x - width // 2) % w
        gap = torch.zeros_like(wet[0])
        gap[bottom : bottom + height, columns] = True
        known = wet & ~gap
    else:
        raise ValueError(f"Unknown completion task {task}")
    return torch.zeros_like(wet), known, wet & ~known


def requested_score(pred, target, request, wet):
    """Mean requested-entry CRPS per active channel, keeping C's wet fractions.

    Normalize the requested subset within each channel, then retain that
    channel's original full-grid wet fraction. Average only active channels;
    supplied channels have zero loss/gradient. With request=wet this is C's
    original land-inclusive per-channel objective.
    """
    score = crps(pred, target)
    count = request.sum((-2, -1))
    active = count > 0
    if not active.any():
        raise ValueError("No requested wet entries")
    values = (score * request).mean(0).sum((-2, -1)) / count.clamp_min(1)
    values = values * wet.float().mean((-2, -1))
    return values * active * (len(active) / active.sum())


class MultitaskModel(ScoreModel):
    def __init__(self, base, widths, mode):
        super().__init__(base, "C", widths)
        assert self.input_steps == 2 and self.out_channels == 154
        assert not self.pred_residuals
        self.mode = mode
        # Preserve the seeded backbone/noise initialization and initial forecast.
        with torch.random.fork_rng(devices=[]):
            self.condition = nn.Conv2d(4 * 154, 154, 1, bias=False)
            nn.init.zeros_(self.condition.weight)
        self.task_rng = torch.Generator().manual_seed(31000000)
        self.task_counts = None
        self.task_loss_sums = None

    def reset_tasks(self, epoch, rank, device):
        self.task_rng.manual_seed(31000000 + 1000 * epoch + rank)
        # Plain tensors: DDP must not broadcast accumulated per-rank statistics.
        self.task_counts = torch.zeros(4, device=device)
        self.task_loss_sums = torch.zeros(4, device=device)

    def complete(self, history, boundary, ctx, supplied, history_known, known, request):
        """One joint two-state draw per member; copy supplied output entries."""
        m, b = history.shape[:2]
        # Mask values here as well as at the caller: callers cannot leak targets
        # merely by handing this method an unmasked array.
        supplied = torch.where(known, supplied, 0)
        conditioning = torch.cat(
            [supplied]
            + [v[None].expand(b, -1, -1, -1) for v in (history_known, known, request)],
            dim=1,
        )
        with autocast(enabled=self.base.use_bfloat16, dtype=torch.bfloat16):
            adjustment = self.condition(conditioning)
        conditioned = torch.where(history_known, history, 0) + adjustment.float()[None]
        forcing = boundary[None].expand(m, *boundary.shape).flatten(0, 1)
        pred = self.base.forward_once(
            conditioned.flatten(0, 1), forcing, ctx
        ).unflatten(0, (m, b))
        return torch.where(known, supplied[None], pred)

    def sample_step(self, history, boundary, ctx):
        """Forecast API: never receives target fields or future ocean conditions."""
        wet = ctx.label_mask[:154].bool()
        known = torch.zeros_like(wet)
        supplied = torch.zeros_like(history[0])
        return self.complete(history, boundary, ctx, supplied, wet, known, wet)

    def forward_once(self, prognostic, boundary, ctx):
        return self.sample_step(prognostic[None], boundary, ctx)[0]

    def forward_task(self, batch, task, loss_fn):
        if task == "forecast":
            return super().forward(batch, loss_fn)
        initial, _ = batch.get_initial_input()
        history = initial[None].expand(self.members, *initial.shape)
        history_known, known, request = task_masks(
            task, batch.ctx.label_mask[:154], self.task_rng
        )
        losses = []
        for step in range(len(batch)):
            _, boundary = batch.get_input(step)
            target = batch.get_label(step)
            pred = self.complete(
                history, boundary, batch.ctx, target, history_known, known, request
            )
            losses.append(
                requested_score(pred, target, request, batch.ctx.label_mask[:154])
            )
        return torch.stack(losses).sum(0)

    def forward(self, batch, loss_fn=None):
        # Summary/ordinary evaluation forwards are causal and do not consume the
        # training task stream. Reconstruction evaluation is an explicit API.
        if loss_fn is None or not self.training:
            return super().forward(batch, loss_fn)
        index = 0
        if self.mode == "multitask":
            index = int(
                torch.multinomial(
                    torch.tensor(TASK_PROBABILITIES), 1, generator=self.task_rng
                )
            )
        losses = self.forward_task(batch, TASKS[index], loss_fn)
        if self.task_counts is not None:
            assert self.task_loss_sums is not None
            n = batch.get_label(0).shape[0]
            self.task_counts[index] += n
            self.task_loss_sums[index] += losses.detach().mean() * n
        return losses


class MultitaskModelConfig(ScoreModelConfig):
    arm: Literal["C"] = "C"
    mode: Literal["forecast", "multitask"] = "forecast"

    def build(self, *args, **kwargs):
        return MultitaskModel(
            SamudraConfig.build(self, *args, **kwargs), self.unet.ch_width, self.mode
        )
