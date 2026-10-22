# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Continuous latent annual rollouts with direct calendar-month readouts."""

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_annual import inputs, month_weights
from samudra.experiments.observation_metrics import _field, weighted_rmse
from samudra.metrics import kernels


def require_endpoint_step(saved, row):
    """Keep annual reporting tied to each explicitly frozen training endpoint."""
    expected = row.get("step", 3000)
    if type(expected) is not int or expected <= 0:
        raise ValueError("Invalid fixed endpoint step")
    if saved["step"] != expected:
        raise ValueError(f"Expected fixed cooled endpoint at step {expected}")


def qualification_model(config):
    """Select the readout replay control without assuming a six-model cohort."""
    index = config.get("qualification_model_index", 5)
    if type(index) is not int or not 0 <= index < len(config["models"]):
        raise ValueError("Invalid annual qualification model index")
    return config["models"][index]


def month_layout(starts, month):
    # NPZ payloads carry numpy.str_ scalars; older pandas rejects those in
    # Timestamp even though DatetimeIndex accepts the containing array.
    starts = pd.DatetimeIndex(starts)
    month = str(month)
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
def rollout(
    model,
    data,
    description,
    raw,
    *,
    origin_index,
    first_month_only=False,
    export_profiles=False,
    export_member=None,
    model_batch=None,
):
    """Initialize once; never consume future ocean observations during evolution."""
    if export_member is not None and (
        type(export_member) is not int
        or not 0 <= export_member < 8
        or not export_profiles
    ):
        raise ValueError("A retained member must be 0–7 and include profile exports")
    if model_batch is None:
        surface, atmosphere, contexts, _, valid = inputs(data, raw)
        fields, available = history_fields(surface, valid)
        monthly_sample = data.load(
            data.root / "test" / (description["origin"][:7] + ".npz")
        )
        prefix = monthly_sample["atmosphere"].shape[1]
        # Match the first month's adapter batch shape exactly for replay qualification.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            forcing = torch.cat(
                [
                    model.adapt(atmosphere[:, :prefix]),
                    model.adapt(atmosphere[:, prefix:]),
                ],
                1,
            )
    else:
        fields, available, forcing, contexts = (
            model_batch[k] for k in ("fields", "available", "forcing", "contexts")
        )
        if (
            forcing.shape[1:3] != (92, 3)
            or available[:, :, :38].any()
            or available[:, :, 39:76].any()
        ):
            raise ValueError(
                "Expected OM4 native forcing and surface-only availability"
            )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        states = model.trajectory(fields, available, forcing, contexts)
    if len(states) != 74 or not all(torch.isfinite(s).all() for s in states):
        raise ValueError("Annual latent trajectory is incomplete or nonfinite")
    gen = torch.Generator(device=data.device).manual_seed(904001 + origin_index)
    native_support = torch.zeros_like(data.mask, dtype=torch.bool)
    native_support[[38, 76]] = data.mask[[38, 76]].bool()
    monthly_support = torch.zeros_like(native_support)
    monthly_support[data.ts_indices] = data.ts_mask.bool()
    if model_batch is not None:
        native_support = data.mask.bool()
        monthly_support = data.mask.bool()
    native, native_members = {}, {}
    monthly_ohc, months, temperatures = [], [], []
    member_ohc, member_temperatures = [], []
    for record in description["monthly_interiors"][: 1 if first_month_only else 12]:
        month = record["file"][:7]
        indices, weights, duration, position = month_layout(raw["start"][19:], month)
        for index in indices:
            if int(index) in native:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                draws = model.decoder.sample(
                    states[1 + index].expand(8, -1, -1, -1, -1),
                    native_support,
                    gen,
                    steps=32,
                    position_days=2.5 + 5 * int(index),
                )
                draw = draws.mean(0, keepdim=True)
            if not torch.isfinite(draw).all():
                raise ValueError("Nonfinite native readout")
            native[int(index)] = draw[:, [38, 76]].float().cpu()
            if export_member is not None:
                native_members[int(index)] = (
                    draws[export_member : export_member + 1, [38, 76]].float().cpu()
                )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.monthly_condition(
                [states[1 + i] for i in indices], data.tensor(weights)
            )
            monthly_draws = model.decoder.sample(
                latent.expand(8, -1, -1, -1, -1),
                monthly_support,
                gen,
                steps=32,
                duration_days=duration,
                position_days=position,
            )
            monthly = monthly_draws.mean(0, keepdim=True)
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
        if export_profiles:
            temperatures.append(
                data.physical(monthly[:, None])[0, 0, 38:52].cpu().numpy()
            )
        if export_member is not None:
            member = monthly_draws[export_member : export_member + 1].clone()
            member_window = torch.stack(
                [native_members[int(i)] for i in indices], 1
            ).to(data.device, dtype=member.dtype)
            # Retain the annual operator: top-layer T is the calendar-weighted
            # native SST, here using the same member index in every readout.
            member[:, 38] = (
                member_window[:, :, 0] * data.tensor(weights)[None, :, None, None]
            ).sum(1)
            member_ohc.append(data.ohc(member)[0])
            member_temperatures.append(
                data.physical(member[:, None])[0, 0, 38:52].cpu().numpy()
            )
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
    result = dict(
        surface=physical,
        ohc=np.stack(monthly_ohc),
        months=np.asarray(months),
        latent_rms=np.asarray(norms),
    )
    if export_profiles:
        # A separate stream and a final readout leave the annual readouts and
        # their noise sequence untouched. This is a decoded initialization,
        # not an observed interior or a correction to the forecast trajectory.
        initial_gen = torch.Generator(device=data.device).manual_seed(
            905001 + origin_index
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial_draws = model.decoder.sample(
                states[0].expand(8, -1, -1, -1, -1),
                data.mask.bool(),
                initial_gen,
                steps=32,
                position_days=-2.5,
            )
            initial = initial_draws.mean(0, keepdim=True)
        physical_initial = data.physical(initial[:, None])[0, 0].cpu().numpy()
        indices = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
        result.update(
            temperature=np.stack(temperatures),
            initial_interior=physical_initial[indices],
            initial_surface=physical_initial[[38, 76]],
            initial_ohc=data.ohc(initial)[0],
        )
        for key in ("temperature", "initial_interior", "initial_surface"):
            if not np.isfinite(result[key]).all():
                raise ValueError(f"Nonfinite exported {key}")
        if export_member is not None:
            member_initial = initial_draws[export_member : export_member + 1]
            physical_member = data.physical(member_initial[:, None])[0, 0].cpu().numpy()
            member_ordered = torch.stack(
                [native_members[i] for i in sorted(native_members)], 1
            ).to(data.device)
            result.update(
                member_surface=(
                    member_ordered * data.std[:, :, [38, 76]]
                    + data.mean[:, :, [38, 76]]
                )[0]
                .cpu()
                .numpy(),
                member_ohc=np.stack(member_ohc),
                member_temperature=np.stack(member_temperatures),
                member_initial_interior=physical_member[indices],
                member_initial_surface=physical_member[[38, 76]],
                member_initial_ohc=data.ohc(member_initial)[0],
            )
    return result


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


def fixed_ohc_support(grid):
    """Complete model temperature columns intersecting each fixed OHC layer."""
    from samudra.constants import build_om4_layout

    edges = np.r_[0.0, np.cumsum(build_om4_layout().depth_thickness)]
    wet = np.asarray(grid["mask"][38:57], dtype=bool)
    return np.stack(
        [
            wet[(edges[:-1] < layer.max_depth) & (edges[1:] > layer.min_depth)].all(0)
            for layer in kernels.OHC_LAYERS
        ]
    )


def monthly_ohc_metrics(prediction, reference, climatology, area, support):
    """Mask target by fixed geometry; missing predictions within it remain errors."""
    prediction, reference, climatology = map(
        np.asarray, (prediction, reference, climatology)
    )
    support = np.asarray(support, dtype=bool)
    if (
        prediction.shape != reference.shape
        or climatology.shape != prediction.shape
        or prediction.ndim != 5
        or prediction.shape[1:3] != (12, 2)
        or support.shape != prediction.shape[2:]
    ):
        raise ValueError("Annual OHC arrays or fixed geometry differ")
    if not np.isfinite(np.where(support, prediction, 0)).all():
        raise ValueError("Nonfinite OHC prediction on fixed model support")
    truth = np.where(support, reference, np.nan)
    result = []
    for month in range(12):
        entry = {"month": month + 1}
        for channel, label in enumerate(("0_700", "700_2000")):
            entry[label] = weighted_rmse(
                prediction[:, month, channel], truth[:, month, channel], area
            )
            entry[label + "_climatology"] = weighted_rmse(
                climatology[:, month, channel], truth[:, month, channel], area
            )
        result.append(entry)
    return result
