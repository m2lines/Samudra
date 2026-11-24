# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed MSE-based objectives for the OM4 structure-loss experiment."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from samudra.utils.ctx import BatchGrid


class StructureLoss:
    def __init__(self, kind: str, coefficient: float, weights_path: str | None = None):
        self.kind = kind
        self.coefficient = coefficient
        self.weights = self.mask = None
        self._cached = None
        if kind == 'local':
            if not weights_path:
                raise ValueError('Local loss requires a frozen weights archive')
            with np.load(Path(weights_path), allow_pickle=False) as z:
                self.weights = torch.from_numpy(z['weights'].copy())
                self.mask = torch.from_numpy(z['mask'].copy()).bool()
            if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
                raise ValueError('Invalid local weights')

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, ctx: BatchGrid):
        wet = ctx.label_mask.to(device=pred.device).bool()
        if wet.ndim != 3 or pred.shape[1:] != wet.shape:
            raise ValueError('Expected a matching time-major channel mask')
        e = (pred - target) * wet
        state = e.square().mean((0, 2, 3))
        if self.kind == 'local':
            key = (pred.device, pred.dtype, pred.shape[1], ctx.label_mask.data_ptr())
            if self._cached is None or self._cached[0] != key:
                if pred.shape[1] % len(self.weights):
                    raise ValueError('Weight channels do not divide output channels')
                repeats = pred.shape[1] // len(self.weights)
                mask = self.mask.repeat(repeats, 1, 1).to(pred.device)
                if not torch.equal(mask, wet):
                    raise ValueError('Frozen weights mask differs from data mask')
                w = self.weights.repeat(repeats, 1, 1).to(device=pred.device, dtype=pred.dtype)
                self._cached = (key, w)
            return (e.square() * self._cached[1]).mean((0, 2, 3))
        if self.kind == 'curvature':
            inside = wet.clone()
            neighbors = torch.zeros_like(e)
            for axis, shift in ((-2,-1),(-2,1),(-1,-1),(-1,1)):
                inside &= wet.roll(shift, axis)
                neighbors = neighbors + e.roll(shift, axis)
            inside[:, (0,-1), :] = False
            hp = (e - neighbors / 4) * inside
            return state + self.coefficient * hp.square().mean((0, 2, 3))
        if self.kind == 'horizontal':
            dx = e.roll(-1, -1) - e
            dy = F.pad(e[...,1:,:] - e[...,:-1,:], (0,0,0,1))
            penalty = ((dx.abs()+dy.abs())*wet).sum((0,2,3))
            penalty = penalty / (2*pred.shape[0]*wet.sum((-2,-1)).clamp_min(1))
            return state + self.coefficient * penalty
        raise ValueError(f'Unknown structure loss: {self.kind}')
