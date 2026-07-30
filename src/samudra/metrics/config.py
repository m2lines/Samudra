# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the observation-metrics phase of an eval job."""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field, model_validator

from samudra.metrics import kernels, observations
from samudra.utils.location import Location, ResolvedLocation, UnresolvedLocation

logger = logging.getLogger(__name__)


class ObsMetricsConfig(BaseModel):
    """Where the observation products live, and over what period to score.

    Disabled by default. Enabling it makes an eval job compare its rollout
    against DUACS, OISST, and ARGO-IAP once the rollout is on disk.
    """

    enabled: bool = Field(
        default=False,
        description="Compute observation metrics after the rollout finishes.",
    )
    duacs_location: Location = Field(
        default=UnresolvedLocation(path="obs/duacs.zarr"),
        description="DUACS surface geostrophic velocity, on its native grid.",
    )
    oisst_location: Location = Field(
        default=UnresolvedLocation(path="obs/oisst.zarr"),
        description="OISST sea-surface temperature, on its native grid.",
    )
    argo_iap_location: Location = Field(
        default=UnresolvedLocation(path="obs/argo-iap.zarr"),
        description="ARGO-IAP gridded temperature, on its native grid.",
    )
    rmse_start: str = Field(
        default="2015-01-01",
        description=(
            "Start of the primary scoring window. Must begin a complete calendar "
            "year: the score and its bootstrap both use equal-year blocks."
        ),
    )
    rmse_end: str = Field(
        default="2022-12-31",
        description="End of the primary scoring window; must end a complete calendar year.",
    )
    bootstrap_samples: int = Field(
        default=kernels.DEFAULT_BOOTSTRAP_SAMPLES,
        ge=0,
        description="Calendar-year block-bootstrap draws for the 95% CI; 0 disables.",
    )
    velocity_kind: str = Field(
        default="absolute",
        description=(
            "Which DUACS geostrophic velocity to compare against. 'absolute' "
            "matches the model velocity derived from zos."
        ),
    )
    baselines: list[str] = Field(
        default_factory=lambda: ["om4"],
        description=(
            "Reference rollouts scored alongside the model. 'om4' scores the "
            "ground-truth OM4 data already staged for the eval, which is what "
            "makes the model's own numbers interpretable."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> ObsMetricsConfig:
        if self.velocity_kind not in observations.DUACS_VELOCITY_COMPONENTS:
            raise ValueError(
                f"velocity_kind must be one of "
                f"{sorted(observations.DUACS_VELOCITY_COMPONENTS)}, "
                f"got {self.velocity_kind!r}"
            )
        unknown = set(self.baselines) - {"om4"}
        if unknown:
            raise ValueError(f"Unknown baselines {sorted(unknown)}; supported: ['om4']")
        if self.window[0] > self.window[1]:
            raise ValueError(
                f"rmse_start ({self.rmse_start}) must not be after rmse_end "
                f"({self.rmse_end})"
            )
        return self

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.rmse_start), pd.Timestamp(self.rmse_end)

    def open_products(self, data_root: ResolvedLocation) -> dict[str, xr.Dataset]:
        """Open and standardize the three observation products."""
        return {
            "duacs": observations.open_product(
                data_root.resolve(self.duacs_location), "DUACS"
            ),
            "oisst": observations.open_product(
                data_root.resolve(self.oisst_location), "OISST"
            ),
            "argo": observations.open_product(
                data_root.resolve(self.argo_iap_location), "ARGO-IAP"
            ),
        }
