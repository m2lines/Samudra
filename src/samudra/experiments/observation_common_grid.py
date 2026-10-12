# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Reporting-only native model -> archived 1-degree observation comparison.

Construct one CommonGrid from the native and archived model grids. Its coverage
and supports depend only on geometry, never forecasts. Use the same canonical
legacy truth, origins and training-only climatology for every call to score().
Truth validity may vary by origin/lead; it must not vary between candidates.
This module does not select checkpoints or change native training metrics.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from samudra.constants import build_om4_layout
from samudra.experiments.observation_metrics import (
    score,
    selection_score,
    weighted_rmse,
)
from samudra.experiments.observation_prepare import ConservativeRemap, save_atomic
from samudra.metrics import kernels


def sidecar_hashes(directory, expected):
    """Require the exact sidecar inventory before recording completion."""
    directory = Path(directory)
    expected = list(expected)
    actual = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    if len(expected) != len(set(expected)) or actual != set(expected):
        raise ValueError(
            f"Common-grid sidecar inventory differs: missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    hashes = {}
    for name in sorted(expected):
        with (directory / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def reporting_masks(grid):
    """SST, ADT and two complete-column OHC masks in physical channel order."""
    mask = np.asarray(grid["mask"])
    if mask.shape != (77, len(grid["lat"]), len(grid["lon"])):
        raise ValueError("Expected canonical 77-channel physical model grid")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Model masks must be binary")
    mask = mask.astype(bool)
    edges = np.r_[0.0, np.cumsum(build_om4_layout().depth_thickness)]
    columns = []
    for layer in kernels.OHC_LAYERS:
        keep = (edges[:-1] < layer.max_depth) & (edges[1:] > layer.min_depth)
        columns.append(mask[38:57][keep].all(axis=0))
    return np.stack([mask[38], mask[76], *columns])


class CommonGrid:
    """Fixed wet-area geometry and legacy scoring support.

    ``coverage`` is [SST, ADT, OHC0-700, OHC700-2000, target_y, target_x].
    ``surface_support`` deliberately intersects SST and ADT support so the
    unchanged legacy scorer can use one shared domain. This can exclude more
    SST cells than an independently optimized per-variable domain.

    The 90% threshold follows observation preparation. These inferred spherical
    cell bounds are the repository's established geometry, not external bounds.
    Call geometry_arrays() for a bounded CPU coverage audit before reporting.
    """

    def __init__(
        self, source_grid, legacy_grid, *, min_coverage=0.9, global_observations=True
    ):
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must lie in (0, 1]")
        self.remap = ConservativeRemap(
            source_grid["lat"],
            source_grid["lon"],
            legacy_grid["lat"],
            legacy_grid["lon"],
        )
        self.lat = np.array(legacy_grid["lat"], copy=True)
        self.lon = np.array(legacy_grid["lon"], copy=True)
        self.source_lat = np.array(source_grid["lat"], copy=True)
        self.source_lon = np.array(source_grid["lon"], copy=True)
        self.wet = reporting_masks(source_grid)
        self.min_coverage = min_coverage
        self.global_observations = global_observations
        self.wet_area = np.stack(
            [self.remap.integral(m.astype(float)) for m in self.wet]
        )
        self.coverage = self.wet_area / self.remap.area
        self.covered = self.coverage >= min_coverage - 1e-10
        legacy_masks = reporting_masks(legacy_grid)
        domain = np.asarray(legacy_grid["mask"])[0].astype(bool).copy()
        if not global_observations:
            domain &= np.abs(self.lat[:, None]) <= 60
        self.legacy_domain = domain
        self.surface_support = domain & (legacy_masks[:2] & self.covered[:2]).all(0)
        self.ohc_support = domain & legacy_masks[2:] & self.covered[2:]
        if (
            not self.surface_support.any()
            or not self.ohc_support.any(axis=(-2, -1)).all()
        ):
            raise ValueError("Empty common-grid surface or OHC support")
        # Keep scoring weights identical to the legacy metric. Remapping above
        # uses spherical overlap, not these cosine weights.
        self.area = np.broadcast_to(
            np.cos(np.deg2rad(self.lat))[:, None], self.remap.shape
        ).copy()

    def project(self, physical, *, ohc=False):
        """Strict physical float fields [..., 2, y, x] -> wet means on target.

        Dry cells may contain NaN or arbitrary values. Every fixed wet source
        cell must be finite, even if its target cell is below coverage threshold.
        Iteration is one field at a time; no full source-period copy is made.
        """
        values = np.asarray(physical)
        if values.shape[-3:] != (2, *self.remap.source_shape):
            raise ValueError("Expected two physical fields on the source grid")
        offset = 2 if ohc else 0
        output = np.full(
            (*values.shape[:-2], *self.remap.shape), np.nan, dtype=np.float32
        )
        for i, (field, target) in enumerate(
            zip(
                values.reshape((-1, *self.remap.source_shape)),
                output.reshape((-1, *self.remap.shape)),
                strict=True,
            )
        ):
            channel = offset + i % 2
            wet = self.wet[channel]
            if not np.isfinite(field[wet]).all():
                raise ValueError("Nonfinite prediction on fixed source wet support")
            numerator = self.remap.integral(np.where(wet, field, 0).astype(np.float64))
            np.divide(
                numerator,
                self.wet_area[channel],
                out=target,
                where=self.covered[channel],
            )
            target[~self.covered[channel]] = np.nan
        return output

    def geometry_arrays(self):
        """Arrays for an explicit support/coverage audit; no predictions required."""
        return dict(
            lat=self.lat,
            lon=self.lon,
            source_lat=self.source_lat,
            source_lon=self.source_lon,
            coverage=self.coverage,
            surface_support=self.surface_support,
            ohc_support=self.ohc_support,
            legacy_domain=self.legacy_domain,
            min_coverage=np.array(self.min_coverage),
            global_observations=np.array(self.global_observations),
        )

    def validate_source(self, grid):
        """Reject accidental reuse with a different source coordinate or mask."""
        for actual, expected in (
            (grid["lat"], self.source_lat),
            (grid["lon"], self.source_lon),
            (reporting_masks(grid), self.wet),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError("Common-grid source geometry differs from model")

    def write_sidecar(self, path, *, surface, model_ohc, origins, months=()):
        """Write physical common-grid predictions, independently of native exports.

        Monthly: surface [6,2,H,W], model_ohc [2,H,W], one origin.
        Annual: surface [73,2,H,W], model_ohc [12,2,H,W], one origin,
        with ordered YYYY-MM month labels. Inputs are pre-IAP model-grid maps.
        No native climatology is exported here: scoring uses the legacy control.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_atomic(
            path,
            prediction=self.project(surface),
            predicted_ohc=self.project(model_ohc, ohc=True),
            origins=np.asarray(origins),
            months=np.asarray(months, dtype=str),
            **self.geometry_arrays(),
        )

    def score(
        self,
        prediction,
        predicted_ohc,
        *,
        reference,
        reference_ohc,
        climatology_surface,
        climatology_ohc,
    ):
        """Rescore a monthly cohort and one frozen legacy climatology together.

        Predictions are already on the target grid (project quarter outputs;
        pass archived 1-degree outputs unchanged). All six arrays have identical
        origin ordering. Surface predictions/control: [N,6,2,H,W]; references:
        [N,6,4,H,W]; OHC: [N,2,H,W]. Use canonical legacy truth, not legacy
        exported truth that may have been masked using a candidate's finiteness.

        The returned reporting ratio uses freshly rescored climatology metrics,
        never the archived denominators or the native quarter climatology.
        """
        n = len(prediction)
        shapes = [
            (prediction, (n, 6, 2, *self.remap.shape)),
            (climatology_surface, (n, 6, 2, *self.remap.shape)),
            (reference, (n, 6, 4, *self.remap.shape)),
            *[
                (a, (n, 2, *self.remap.shape))
                for a in (predicted_ohc, reference_ohc, climatology_ohc)
            ],
        ]
        if n < 2 or any(np.shape(a) != shape for a, shape in shapes):
            raise ValueError(
                "Expected matching multi-origin monthly common-grid arrays"
            )
        kwargs = dict(
            reference=reference,
            reference_ohc=reference_ohc,
            lat=self.lat,
            lon=self.lon,
            mask=self.surface_support,
            strict_velocity_support=True,
            global_observations=True,
            compute_spectra=False,
            climatology_surface=climatology_surface,
            ohc_area=self.area,
            ohc_support=self.ohc_support,
        )
        candidate = score(prediction=prediction, predicted_ohc=predicted_ohc, **kwargs)
        control = score(
            prediction=climatology_surface, predicted_ohc=climatology_ohc, **kwargs
        )
        return dict(
            candidate=candidate,
            control=control,
            common_grid_rmse_ratio=selection_score(
                candidate, control, [], metric="rmse-only"
            ),
        )

    def score_annual(
        self, prediction, predicted_ohc, *, reference, reference_ohc, months
    ):
        """One-year legacy metrics on the same intersection; no annual blend.

        Inputs are target-grid physical maps, 73 surface steps and 12 exact
        calendar-month OHC maps. Call separately for each candidate/control
        with identical canonical references and month ordering.
        """
        from samudra.experiments.observation_annual import surface_metrics

        shapes = [
            (prediction, (73, 2, *self.remap.shape)),
            (reference, (73, 4, *self.remap.shape)),
            (predicted_ohc, (12, 2, *self.remap.shape)),
            (reference_ohc, (12, 2, *self.remap.shape)),
        ]
        if (
            len(months) != 12
            or len(set(months)) != 12
            or any(np.shape(a) != shape for a, shape in shapes)
        ):
            raise ValueError("Expected 73 steps and 12 ordered monthly OHC maps")
        month_labels = np.asarray(months, dtype=str)
        month_dates = month_labels.astype("datetime64[M]")
        if (
            not np.array_equal(month_labels, month_dates.astype(str))
            or not (np.diff(month_dates.astype(np.int64)) == 1).all()
        ):
            raise ValueError("OHC month labels must be consecutive ordered YYYY-MM")
        if not np.isfinite(np.where(self.surface_support, prediction, 0)).all():
            raise ValueError("Nonfinite prediction on fixed common surface support")
        if not np.isfinite(np.where(self.ohc_support, predicted_ohc, 0)).all():
            raise ValueError("Nonfinite prediction on fixed common OHC support")
        data = SimpleNamespace(
            grid=dict(lat=self.lat, lon=self.lon, mask=self.surface_support[None]),
            global_observations=True,
        )
        result = surface_metrics(prediction, reference, data)
        truth = np.where(self.ohc_support, reference_ohc, np.nan)
        result["monthly_ohc"] = {
            month: {
                label: weighted_rmse(predicted_ohc[i, c], truth[i, c], self.area)
                for c, label in enumerate(("0_700", "700_2000"))
            }
            for i, month in enumerate(months)
        }
        return result
