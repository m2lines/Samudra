# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Lead-aware adaptation of the integrated observation kernels for monthly origins.

No temporal spectra are computed across independently reinitialized forecasts.
Selection uses fixed validation controls and explicit common spatial support.
"""

from typing import Any

import numpy as np
import xarray as xr

from samudra.metrics import kernels, spectra

PROTOCOL: dict[str, Any] = {
    "version": 2,
    "selection": "0.5 * mean(normalized integrated errors) + 0.5 * mean(spatial spectral error in dex)",
    "integrated": [
        "sst_rmse",
        "velocity_rmse",
        "eke_rmse",
        "ohc_0_700_rmse",
        "ohc_700_2000_rmse",
    ],
    "normalization": "fixed validation seasonal-climatology control error; immutable once scored",
    "spatial_spectra": ["sst", "adt", "eke"],
    "regions": [r[0] for r in spectra.SPATIAL_REGIONS],
    "leads_days": [5, 15, 30],
    "minimum_wavelength_grid_cells": 4,
    "missing": "freeze available spectral keys from reference/control qualification; require same keys for every checkpoint",
    "temporal_spectra": "unavailable for selection: nine-month window and reinitialized origins",
    "variance": "report fixed-lead variance without short-record seasonal/trend fitting",
    "domain": "60S-60N, common finite observation and model wet support; geostrophy excludes 5S-5N",
    "ohc": "native-layer integral, complete-column common support, calendar-month means",
}


def weighted_rmse(prediction, reference, area):
    valid = np.isfinite(reference)
    if not valid.any() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Missing prediction on fixed reference support")
    weight = np.broadcast_to(area, reference.shape) * valid
    return float(
        np.sqrt(
            np.sum(np.where(valid, (prediction - reference) ** 2, 0) * weight)
            / weight.sum()
        )
    )


def _field(values, lat, lon):
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(len(values)), "lat": lat, "lon": lon},
    )


def spatial_error(prediction, reference, lat, lon, region):
    name, longitude, latitude = region
    ref = _field(reference, lat, lon)
    pred = _field(prediction, lat, lon).where(np.isfinite(ref))
    k, power = spectra.region_spectrum(ref, longitude, latitude, name=name)
    kp, pp = spectra.region_spectrum(pred, longitude, latitude, name=name)
    if not len(k) or not len(kp):
        return None
    # The coarsened model/reference have the same nominal spacing. Avoid claiming
    # resolution at the Nyquist limit of a heavily coarsened analysis.
    maximum_k = (
        2 * np.pi / (4 * spectra.METRES_PER_DEGREE / 1000 * np.max(np.diff(lat)))
    )
    keep, keep_pred = k <= maximum_k, kp <= maximum_k
    if keep.sum() < 2 or keep_pred.sum() < 2:
        return None
    error = spectra.log10_rmse_between_curves(
        k[keep], power[keep], kp[keep_pred], pp[keep_pred]
    )
    return {
        "error_dex": float(error),
        "k_rad_km": k[keep].tolist(),
        "reference_power": power[keep].tolist(),
        "prediction_power": pp[keep_pred].tolist(),
    }


def protocol(strict_velocity_support=False):
    if not strict_velocity_support:
        return PROTOCOL
    return {
        **PROTOCOL,
        "version": 3,
        "velocity_support": "Both observed velocity components and complete observed ADT derivative stencil; prediction-independent; EKE uses identical per-origin support",
    }


def score(
    prediction,
    reference,
    predicted_ohc,
    reference_ohc,
    lat,
    lon,
    mask,
    strict_velocity_support=False,
):
    """Arrays: surface [origins, leads, 2, y, x], velocity refs at slots 2:4.

    Prediction only needs SST and ADT. Reference slots are SST, ADT, ugos, vgos.
    Monthly OHC arrays have shape [origins, 2, y, x].
    """
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    domain = mask & (np.abs(lat[:, None]) <= 60)
    if not np.isfinite(np.where(domain, prediction, 0)).all():
        raise ValueError("Nonfinite prediction on the fixed model domain")
    metrics, curves = {}, {}
    accumulated: dict[str, list[float]] = {
        name: [] for name in ("sst_rmse", "velocity_rmse", "eke_rmse")
    }
    for lead in (0, 2, 5):
        p = np.where(domain, prediction[:, lead], np.nan)
        r = np.where(domain, reference[:, lead], np.nan)
        sst = r[:, 0]
        accumulated["sst_rmse"].append(weighted_rmse(p[:, 0], sst, area))
        pu, pv = kernels.geostrophic_velocity_from_zos(
            _field(p[:, 1], lat, lon), "lat", "lon"
        )
        pu = pu.transpose("time", "lat", "lon")
        pv = pv.transpose("time", "lat", "lon")
        u, v = r[:, 2], r[:, 3]
        velocity_support = (
            (np.abs(lat[:, None]) >= 5)
            & np.isfinite(pu.values).all(0)
            & np.isfinite(pv.values).all(0)
        )
        if strict_velocity_support:
            observed_u, observed_v = kernels.geostrophic_velocity_from_zos(
                _field(r[:, 1], lat, lon), "lat", "lon"
            )
            velocity_support = (
                velocity_support[None]
                & np.isfinite(observed_u.transpose("time", "lat", "lon").values)
                & np.isfinite(observed_v.transpose("time", "lat", "lon").values)
                & np.isfinite(u)
                & np.isfinite(v)
            )
            support = _field(velocity_support, lat, lon)
            pu, pv = pu.where(support), pv.where(support)
        # Support is geometrically determined by static model mask and supplied refs.
        u, v = (
            np.where(velocity_support, u, np.nan),
            np.where(velocity_support, v, np.nan),
        )
        velocity = np.stack([u, v], 1)
        accumulated["velocity_rmse"].append(
            np.sqrt(2)
            * weighted_rmse(np.stack([pu.values, pv.values], 1), velocity, area)
        )
        pe = kernels.instantaneous_surface_eke(pu, pv).values
        re = kernels.instantaneous_surface_eke(
            _field(u, lat, lon), _field(v, lat, lon)
        ).values
        accumulated["eke_rmse"].append(weighted_rmse(pe, re, area))
        for label, predicted, truth in [
            ("sst", p[:, 0], sst),
            ("adt", p[:, 1], r[:, 1]),
            ("eke", pe, re),
        ]:
            for region in spectra.SPATIAL_REGIONS:
                key = f"{label}/{region[0]}/day{5 * (lead + 1)}"
                result = spatial_error(predicted, truth, lat, lon, region)
                if result is not None:
                    curves[key] = result
        for key in accumulated:
            metrics[f"{key}/day{5 * (lead + 1)}"] = float(accumulated[key][-1])
        variance_ref = np.nanvar(sst, axis=0)
        variance_pred = np.nanvar(p[:, 0], axis=0)
        metrics[f"sst_variance_map_rmse/day{5 * (lead + 1)}"] = weighted_rmse(
            variance_pred, variance_ref, area
        )
    for key, values in accumulated.items():
        metrics[key] = float(np.mean(values))
    for channel, label in enumerate(("0_700", "700_2000")):
        truth = np.where(domain, reference_ohc[:, channel], np.nan)
        metrics[f"ohc_{label}_rmse"] = weighted_rmse(
            predicted_ohc[:, channel], truth, area
        )
    return {"metrics": metrics, "spectra": curves}


def selection_score(result, control, spectral_keys):
    if not spectral_keys:
        raise ValueError("No qualified spectral metrics: refuse RMSE-only selection")
    ratios = []
    for name in PROTOCOL["integrated"]:
        denominator = control["metrics"][name]
        numerator = result["metrics"][name]
        if (
            not np.isfinite(denominator)
            or denominator <= 0
            or not np.isfinite(numerator)
        ):
            raise ValueError(f"Invalid fixed selection component: {name}")
        ratios.append(numerator / denominator)
    errors = [result["spectra"][key]["error_dex"] for key in spectral_keys]
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite required spectral comparison")
    return float(0.5 * np.mean(ratios) + 0.5 * np.mean(errors))
