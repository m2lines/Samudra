# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Continuous latent annual rollouts with direct calendar-month readouts."""

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_annual import inputs, month_weights
from samudra.experiments.observation_metrics import _field, weighted_rmse
from samudra.metrics import kernels


def month_layout(starts, month):
    weights = month_weights(starts, month).astype(np.float32)
    indices = np.flatnonzero(weights)
    duration = float(pd.Timestamp(month).days_in_month)
    position = (
        float((pd.Timestamp(month) - pd.Timestamp(starts[0])) / pd.Timedelta(days=1))
        + duration / 2
    )
    return indices, weights[indices], duration, position


def history_fields(surface, validity):
    """Lift surface history into masked physical inputs; sample masks may be floats."""
    if (
        surface.shape != validity.shape
        or surface.ndim != 5
        or surface.shape[1:3] != (19, 2)
    ):
        raise ValueError("Expected aligned 19-bin SST/SSH history and validity")
    fields = surface.new_zeros(surface.shape[0], 19, 77, *surface.shape[-2:])
    available = torch.zeros_like(fields, dtype=torch.bool)
    fields[:, :, [38, 76]] = surface
    available[:, :, [38, 76]] = validity.bool()
    return fields, available


@torch.no_grad()
def rollout(model, data, description, raw, *, origin_index, first_month_only=False):
    """Initialize once; never consume future ocean observations during evolution."""
    surface, atmosphere, contexts, _, valid = inputs(data, raw)
    fields, available = history_fields(surface, valid)
    monthly_sample = data.load(
        data.root / "test" / (description["origin"][:7] + ".npz")
    )
    prefix = monthly_sample["atmosphere"].shape[1]
    # Match the first month's adapter batch shape exactly for replay qualification.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        forcing = torch.cat(
            [model.adapt(atmosphere[:, :prefix]), model.adapt(atmosphere[:, prefix:])],
            1,
        )
        states = model.trajectory(fields, available, forcing, contexts)
    if len(states) != 74 or not all(torch.isfinite(s).all() for s in states):
        raise ValueError("Annual latent trajectory is incomplete or nonfinite")
    gen = torch.Generator(device=data.device).manual_seed(904001 + origin_index)
    native_support = torch.zeros_like(data.mask, dtype=torch.bool)
    native_support[[38, 76]] = data.mask[[38, 76]].bool()
    monthly_support = torch.zeros_like(native_support)
    monthly_support[data.ts_indices] = data.ts_mask.bool()
    native = {}
    monthly_ohc, months = [], []
    for record in description["monthly_interiors"][: 1 if first_month_only else 12]:
        month = record["file"][:7]
        indices, weights, duration, position = month_layout(raw["start"][19:], month)
        for index in indices:
            if int(index) in native:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                draw = model.decoder.sample(
                    states[1 + index].expand(8, -1, -1, -1, -1),
                    native_support,
                    gen,
                    steps=32,
                    position_days=2.5 + 5 * int(index),
                ).mean(0, keepdim=True)
            if not torch.isfinite(draw).all():
                raise ValueError("Nonfinite native readout")
            native[int(index)] = draw[:, [38, 76]].float().cpu()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.monthly_condition(
                [states[1 + i] for i in indices], data.tensor(weights)
            )
            monthly = model.decoder.sample(
                latent.expand(8, -1, -1, -1, -1),
                monthly_support,
                gen,
                steps=32,
                duration_days=duration,
                position_days=position,
            ).mean(0, keepdim=True)
        # Reconstruct the original native tensor layout and reduction order so
        # the first-month SST contribution is bitwise replayable.
        native_window = monthly.new_zeros(1, len(indices), 77, *model.shape)
        for j, index in enumerate(indices):
            native_window[:, j, [38, 76]] = native[int(index)].to(
                data.device, dtype=monthly.dtype
            )
        monthly[:, 38] = (
            native_window[:, :, 38] * data.tensor(weights)[None, :, None, None]
        ).sum(1)
        ohc = data.ohc(monthly)[0]
        if np.isinf(ohc).any() or not all(np.isfinite(layer).any() for layer in ohc):
            raise ValueError("Nonfinite OHC readout on model support")
        monthly_ohc.append(ohc)
        months.append(month)
    ordered = torch.stack([native[i] for i in sorted(native)], 1)
    physical = (
        (
            ordered.to(data.device) * data.std[:, :, [38, 76]]
            + data.mean[:, :, [38, 76]]
        )[0]
        .cpu()
        .numpy()
    )
    norms = [float(state.float().square().mean().sqrt()) for state in states]
    return dict(
        surface=physical,
        ohc=np.stack(monthly_ohc),
        months=np.asarray(months),
        latent_rms=np.asarray(norms),
    )


def lead_metrics(prediction, reference, data):
    """Pool matched cases per lead; EKE anomalies are across cases at fixed lead."""
    lat, lon = data.grid["lat"], data.grid["lon"]
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    domain = data.grid["mask"][0]
    results = []
    for lead in range(73):
        p, r = (
            np.where(domain, prediction[:, lead], np.nan),
            np.where(domain, reference[:, lead], np.nan),
        )
        pu_field, pv_field = kernels.geostrophic_velocity_from_zos(
            _field(p[:, 1], lat, lon), "lat", "lon"
        )
        ru_field, rv_field = kernels.geostrophic_velocity_from_zos(
            _field(r[:, 1], lat, lon), "lat", "lon"
        )
        pu, pv, ru, rv = [
            v.transpose("time", "lat", "lon").values
            for v in (pu_field, pv_field, ru_field, rv_field)
        ]
        support = (
            (np.abs(lat[:, None]) >= 5)
            & np.isfinite(ru)
            & np.isfinite(rv)
            & np.isfinite(r[:, 2])
            & np.isfinite(r[:, 3])
        )
        pu, pv, u, v = [
            np.where(support, x, np.nan) for x in (pu, pv, r[:, 2], r[:, 3])
        ]
        pe = kernels.instantaneous_surface_eke(
            _field(pu, lat, lon), _field(pv, lat, lon)
        ).values
        re = kernels.instantaneous_surface_eke(
            _field(u, lat, lon), _field(v, lat, lon)
        ).values
        results.append(
            dict(
                day=5 * (lead + 1),
                sst_rmse=weighted_rmse(p[:, 0], r[:, 0], area),
                adt_rmse=weighted_rmse(p[:, 1], r[:, 1], area),
                velocity_rmse=float(
                    np.sqrt(2)
                    * weighted_rmse(np.stack([pu, pv], 1), np.stack([u, v], 1), area)
                ),
                eke_rmse=weighted_rmse(pe, re, area),
            )
        )
    return results
