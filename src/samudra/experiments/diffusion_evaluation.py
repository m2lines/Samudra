# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the frozen upstream observation metrics on the mean of evolved members."""

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_pilot import Pilot


class EnsembleMeanForecast(nn.Module):
    """Point-score interface only; this does not describe ensemble calibration.

    Members evolve separately before averaging. Fixed evaluation draws are reset
    per origin for reproducible checkpoint comparisons; they are independent of
    training draws. A deterministic model always uses exactly one member.
    """

    def __init__(self, model, *, members=8, seed=4041729):
        super().__init__()
        if members < 1:
            raise ValueError("Positive ensemble size required")
        self.model = model
        self.members = members if model.stochastic else 1
        self.seed = seed

    @torch.no_grad()
    def forecast(self, surface, atmosphere, contexts, mask, validity):
        trajectories, initial = self.model.forecast(
            surface,
            atmosphere,
            contexts,
            mask,
            validity,
            generator=torch.Generator(device=surface.device).manual_seed(self.seed),
            members=self.members,
        )
        return trajectories.float().mean(0), initial.float().mean(0)


def evaluate_point_metrics(
    model, data, paths, reference, *, members=8, seed=4041729, export=None
):
    """Return unchanged upstream metric records and the frozen validation score.

    Callers must supply validation paths for model selection; held-out paths are
    for reporting only. Independent member CRPS/coverage/structure are separate
    outputs and must accompany final point metrics.
    """
    evaluator = Pilot.__new__(Pilot)
    evaluator.args = SimpleNamespace(strict_velocity_support=True)
    evaluator.data = data
    # Pilot annotates its trainable model concretely; this read-only path calls
    # only eval()/forecast(), provided by the point-score adapter.
    cast(Any, evaluator).model = EnsembleMeanForecast(model, members=members, seed=seed)
    metrics = evaluator.evaluate(paths, export=export)
    value = selection_score(metrics, reference["control"], reference["spectral_keys"])
    return metrics, value
