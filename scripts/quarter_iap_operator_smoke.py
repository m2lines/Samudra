#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU geometry/gradient diagnostic; no training or observed label reads.

Use the exported real OM4 grid/masks and raw argo-iap.zarr coordinates. Synthetic
physical fields exercise NativeIapOperator and an experimental dense separable
equivalent with 28 interior channels (or two for a smaller diagnostic). Compare
forward means and the area-mean loss adjoint to an independent CPU overlap
integral. Repeated CUDA outputs AND gradients must
match bitwise under both ordinary and strict deterministic settings, all inside
outer bf16 autocast. A failure writes diagnostic JSON and exits nonzero.

Example (inside the qualified ARM image, source on PYTHONPATH):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python quarter_iap_operator_smoke.py \
    --grid /prepared/om4-quarter-grid.npz --iap /raw/argo-iap.zarr \
    --output /diagnostics/iap-operator.json

This does not qualify observed values, temporal/depth processing, the training
loss integration, or determinism across devices/framework versions.
"""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.special import roots_legendre

from samudra.experiments.quarter_observation_operator import NativeIapOperator


def axis_integrals(source, target, latitude=False):
    """Independent spherical cell intersections, with explicit seam copies."""

    def edges(centers):
        centers = np.asarray(centers, np.float64)
        result = np.empty(len(centers) + 1)
        result[1:-1] = (centers[1:] + centers[:-1]) / 2
        result[0] = centers[0] - (centers[1] - centers[0]) / 2
        result[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
        if latitude:
            result = result.clip(-90, 90)
            result[0], result[-1] = -90, 90
            result = np.sin(np.deg2rad(result))
        return result

    source_edges, target_edges = edges(source), edges(target)
    rows, columns, values = [], [], []
    for row in range(len(target)):
        low, high = target_edges[row : row + 2]
        for shift in [0] if latitude else [-360, 0, 360]:
            for col in np.flatnonzero(
                (source_edges[:-1] + shift < high) & (source_edges[1:] + shift > low)
            ):
                width = min(high, source_edges[col + 1] + shift) - max(
                    low, source_edges[col] + shift
                )
                rows.append(row)
                columns.append(col)
                values.append(width)
    return (
        csr_matrix((values, (rows, columns)), shape=(len(target), len(source))),
        np.diff(target_edges),
    )


def cpu_reference(lat, lon, target_lat, target_lon, mask, physical):
    """Float64 wet-area means and exact adjoint of their global area mean."""
    y, dy = axis_integrals(lat, target_lat, latitude=True)
    x, dx = axis_integrals(lon, target_lon)
    area = dy[:, None] * dx[None, :]

    def integrate(field):
        return x.dot(y.dot(field).T).T

    wet_area = np.stack([integrate(field.astype(np.float64)) for field in mask])
    coverage = (wet_area / area).astype(np.float32)
    # Match the documented operator support tolerance; retain double arithmetic
    # for the physical numerator, denominator and independent gradient.
    supported = coverage >= 0.9 - 1e-7
    numerator = np.stack(
        [
            integrate(np.where(wet, field, 0).astype(np.float64))
            for wet, field in zip(mask, physical, strict=True)
        ]
    )
    means = np.divide(
        numerator, wet_area, out=np.zeros_like(numerator), where=wet_area > 0
    )
    weights = supported * area
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("No supported native IAP cells")
    loss = (means * weights).sum() / denominator
    adjoint = []
    for wet, support, wet_cell_area in zip(mask, supported, wet_area, strict=True):
        sensitivity = np.divide(
            support * area,
            wet_cell_area * denominator,
            out=np.zeros_like(area),
            where=wet_cell_area > 0,
        )
        adjoint.append(x.T.dot(y.T.dot(sensitivity).T).T * wet)
    means[~supported] = np.nan
    return means, coverage, np.stack(adjoint), loss


def load_geometry(grid_path, iap_path, channels):
    with np.load(grid_path) as grid:
        lat, lon = grid["lat"].copy(), grid["lon"].copy()
        if (len(lat), len(lon)) != (720, 1440):
            raise ValueError("Expected the real quarter OM4 grid")
        np.testing.assert_allclose(
            lat, np.rad2deg(np.arcsin(roots_legendre(720)[0])), atol=1e-5
        )
        names = [f"{base}_{level}" for base in ("thetao", "so") for level in range(14)]
        if channels == 2:
            names = ["thetao_0", "so_0"]
        all_names = list(grid["names"].astype(str))
        mask = grid["mask"][[all_names.index(name) for name in names]].copy()
    # Only raw coordinate arrays are materialized, never temp/salt or a month.
    with xr.open_zarr(iap_path, chunks=None, decode_times=False) as dataset:
        target_lat, target_lon = dataset.lat.values.copy(), dataset.lon.values.copy()
    if (len(target_lat), len(target_lon)) != (360, 720):
        raise ValueError("Expected native IAP 360x720 coordinates")
    return lat, lon, target_lat, target_lon, mask, names


class DiagnosticDenseIap(NativeIapOperator):
    """Experimental implementation only: retain exact production float32 weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latitude_weights = self.latitude_weights.to_dense()
        self.longitude_weights = self.longitude_weights.to_dense()

    def forward(self, physical):
        if physical.ndim < 3 or physical.shape[-3:] != self.wet_mask.shape:
            raise ValueError("Prediction channel/grid shape differs from wet_mask")
        if physical.dtype != torch.float32:
            raise ValueError("IAP projection requires physical float32 predictions")
        if not torch.isfinite(physical.masked_select(self.wet_mask)).all():
            raise ValueError("Nonfinite model prediction on wet support")
        leading = physical.shape[:-2]
        safe = torch.where(self.wet_mask, physical, 0).reshape(-1, *self.source_shape)
        count, _, width = safe.shape
        with torch.autocast(device_type=physical.device.type, enabled=False):
            latitude = torch.mm(
                self.latitude_weights,
                safe.permute(1, 0, 2).reshape(self.source_shape[0], -1),
            ).reshape(self.target_shape[0], count, width)
            longitude = torch.mm(
                self.longitude_weights,
                latitude.permute(2, 1, 0).reshape(width, -1),
            )
        integral = longitude.reshape(self.target_shape[1], count, self.target_shape[0])
        integral = integral.permute(1, 2, 0).reshape(*leading, *self.target_shape)
        mean = integral / self.coverage.clamp_min(1e-12)
        return torch.where(self.supported, mean, torch.nan)


def check_variant(cpu_operator, physical, reference, adjoint, expected_loss, repeats):
    """Run one implementation; release its tensors before profiling the next."""
    operator = cpu_operator.cuda()
    values = torch.from_numpy(physical).cuda().requires_grad_()
    weights = operator.area * operator.supported
    denominator = weights.sum()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results: list[dict] = []
    evidence: dict = dict(status="running", policies=results)
    try:
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic, warn_only=False)
            first_output = first_gradient = None
            timings = []
            bitwise_forward = bitwise_gradient = True
            for repeat in range(repeats):
                evidence["stage"] = dict(deterministic=deterministic, repeat=repeat)
                values.grad = None
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    projected = operator(values)
                    safe = torch.where(operator.supported, projected, 0)
                    loss = (safe * weights).sum() / denominator
                loss.backward()
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                if projected.dtype != torch.float32 or values.grad is None:
                    raise AssertionError("Projection/gradient dtype contract failed")
                if not torch.isfinite(values.grad).all():
                    raise AssertionError("Nonfinite gradient")
                if torch.count_nonzero(values.grad[~operator.wet_mask]).item():
                    raise AssertionError("Dry cells received gradients")
                actual: np.ndarray = projected.detach().cpu().numpy()
                gradient: np.ndarray = values.grad.detach().cpu().numpy()
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
                np.testing.assert_allclose(
                    actual, reference, rtol=3e-6, atol=3e-5, equal_nan=True
                )
                np.testing.assert_allclose(gradient, adjoint, rtol=1e-5, atol=1e-12)
                np.testing.assert_allclose(loss.item(), expected_loss, rtol=3e-6)
                if repeat == 0:
                    first_output = projected.detach().clone().view(torch.int32)
                    first_gradient = values.grad.detach().clone().view(torch.int32)
                else:
                    if not torch.equal(
                        first_output, projected.detach().view(torch.int32)
                    ):
                        bitwise_forward = False
                    if not torch.equal(
                        first_gradient, values.grad.detach().view(torch.int32)
                    ):
                        bitwise_gradient = False
            results.append(
                dict(
                    deterministic=deterministic,
                    repeats=repeats,
                    seconds=timings,
                    loss=loss.item(),
                    expected_loss=float(expected_loss),
                    max_forward_absolute_error=float(
                        np.nanmax(np.abs(actual - reference))
                    ),
                    max_gradient_absolute_error=float(
                        np.max(np.abs(gradient - adjoint))
                    ),
                    bitwise_forward=bitwise_forward,
                    bitwise_gradient=bitwise_gradient,
                )
            )
        # Explicitly prove wet NaNs fail loudly on the actual GPU path.
        with torch.no_grad():
            wet_index = int(np.argmax(np.isfinite(physical)))
            values.view(-1)[wet_index] = torch.nan
        try:
            operator(values)
        except ValueError as error:
            if "Nonfinite model prediction" not in str(error):
                raise
        else:
            raise AssertionError("Wet NaN was accepted")
        if not all(p["bitwise_forward"] and p["bitwise_gradient"] for p in results):
            raise AssertionError("Non-bitwise GPU repeats; see per-policy diagnostics")
        evidence["status"] = "passed"
    except Exception as error:
        # Preserve failure and still try the other implementation. diagnose()
        # raises after both, so a failed qualification never exits successfully.
        evidence.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
    finally:
        evidence["max_memory_allocated"] = torch.cuda.max_memory_allocated()
        evidence["max_memory_reserved"] = torch.cuda.max_memory_reserved()
    return evidence


def diagnose(
    lat, lon, target_lat, target_lon, mask, names, *, repeats=3, evidence=None
):
    if evidence is None:
        evidence = {}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python/CUDA"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU")
    if repeats < 2:
        raise ValueError("At least two repetitions are required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This diagnostic requires bf16 autocast support")
    evidence.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        channels=names,
        synthetic_physical_fields=True,
        model_shape=list(mask.shape[-2:]),
    )
    pattern = (
        2 * np.sin(np.deg2rad(lat))[:, None] + np.cos(2 * np.deg2rad(lon))[None, :]
    )
    physical = np.stack(
        [
            (10 if name.startswith("thetao") else 34) + 0.125 * i + pattern
            for i, name in enumerate(names)
        ]
    ).astype(np.float32)
    physical[~mask.astype(bool)] = np.nan
    reference, coverage, adjoint, expected_loss = cpu_reference(
        lat, lon, target_lat, target_lon, mask, physical
    )
    evidence.update(
        iap_shape=list(reference.shape[-2:]),
        supported_cells=int(np.isfinite(reference).sum()),
        unsupported_cells=int(np.isnan(reference).sum()),
        variants={},
    )
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        evidence["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        for variant, operator_class in (
            ("sparse", NativeIapOperator),
            ("dense", DiagnosticDenseIap),
        ):
            cpu_operator = operator_class(lat, lon, target_lat, target_lon, mask)
            np.testing.assert_allclose(
                cpu_operator.coverage.numpy(), coverage, atol=2e-7
            )
            np.testing.assert_array_equal(
                cpu_operator.supported.numpy(), np.isfinite(reference)
            )
            torch.cuda.empty_cache()
            evidence["variants"][variant] = check_variant(
                cpu_operator, physical, reference, adjoint, expected_loss, repeats
            )
            del cpu_operator
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.set_float32_matmul_precision(old_precision)
    if any(variant["status"] != "passed" for variant in evidence["variants"].values()):
        raise AssertionError(
            "GPU qualification failed; see sparse/dense variant results"
        )
    evidence["status"] = "passed"
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--iap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, choices=[2, 28], default=28)
    args = parser.parse_args()
    evidence = dict(grid=str(args.grid), iap=str(args.iap), scope=__doc__)
    try:
        evidence["script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        evidence["operator_sha256"] = hashlib.sha256(
            Path(inspect.getfile(NativeIapOperator)).read_bytes()
        ).hexdigest()
        geometry = load_geometry(args.grid, args.iap, args.channels)
        evidence["grid_sha256"] = hashlib.sha256(args.grid.read_bytes()).hexdigest()
        evidence["iap_metadata_sha256"] = hashlib.sha256(
            (args.iap / ".zmetadata").read_bytes()
        ).hexdigest()
        evidence.update(diagnose(*geometry, evidence=evidence))
    except Exception as error:
        evidence.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
