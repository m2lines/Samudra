# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation training config, composed from Samudra's existing config types."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from samudra.config import ExperimentConfig, RustDataLoadingConfig, TrainBackendConfig
from samudra.config_base import BaseConfig, TopLevelConfig
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.utils.location import LocalLocation, Location, UnresolvedLocation

MIXES = {
    "obs": ("obs",),
    "om4-obs": ("obs", "om4"),
    "om4-cm4-obs": ("obs", "om4", "obs", "cm4"),
    "om4-cm4": ("om4", "cm4"),
}
Mix = Literal["obs", "om4-obs", "om4-cm4-obs", "om4-cm4"]


class ObservationDataConfig(BaseConfig):
    """Prepared archives; relative paths use experiment.data_root."""

    observations: Location = UnresolvedLocation(path="obs/samples")
    daily: Location = UnresolvedLocation(path="obs/daily")
    normalization: Location = UnresolvedLocation(path="obs/normalization")
    om4: Location | None = None
    cm4: Location | None = None
    era5_cache: Location | None = None
    era5_proof_sha256: str | None = None
    native_observations: Location | None = None
    native_observations_proof_sha256: str | None = None
    native_om4: Location | None = None
    native_om4_proof_sha256: str | None = None
    loading: RustDataLoadingConfig | None = None

    def local_path(self, name: str, experiment: ExperimentConfig) -> Path:
        location = getattr(self, name)
        if location is None:
            raise ValueError(f"data.{name} is required for this source mix")
        resolved = experiment.resolved_data_root.resolve(location)
        if not isinstance(resolved, LocalLocation):
            raise ValueError(
                "Observation archives currently require local filesystem paths"
            )
        return resolved.path


class ObservationReplayConfig(BaseConfig):
    per_month: int = Field(default=2, ge=1)
    max_age: int = Field(
        default=250, ge=0, description="Maximum age in optimizer updates"
    )


class ObservationLossConfig(BaseConfig):
    monthly_ts: float = Field(default=0.8, ge=0)
    sst: float = Field(default=0.1, ge=0)
    ssh: float = Field(default=0.1, ge=0)
    reconstruction: float = Field(default=0.1, ge=0)
    completion: float = Field(default=0.1, ge=0)


class ObservationTrainConfig(TopLevelConfig):
    """Train the observation-initialized model from YAML and CLI overrides.

    Update counts and cadence count optimizer steps. One microbatch is a monthly
    source window; gradient_accumulation_steps sets the global effective batch
    size, divided across ranks when update_prefetch enables distributed training.
    """

    experiment: ExperimentConfig
    data: ObservationDataConfig = ObservationDataConfig()
    model: SurfaceInitializedConfig = SurfaceInitializedConfig()
    mix: Mix = "obs"
    main_updates: int = Field(default=2000, ge=1)
    cooldown_updates: int = Field(default=1000, ge=0)
    batch_size: Literal[1] = 1
    gradient_accumulation_steps: int = Field(
        default=8, ge=1, description="Global examples per optimizer update"
    )
    fresh_examples: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    final_learning_rate: float = Field(default=1e-6, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    save_freq: int = Field(
        default=25, ge=1, description="Optimizer updates between saves"
    )
    validation_freq: int = Field(default=250, ge=1)
    validation_origin: str = "2013-11-01"
    ema_decay: float = Field(default=0.5 ** (1 / 100), ge=0, lt=1)
    faster_decay_at_start: bool = False
    replay: ObservationReplayConfig = ObservationReplayConfig()
    loss: ObservationLossConfig = ObservationLossConfig()
    milestone_updates: list[int] = Field(default_factory=list)
    measure_timing: bool = False
    update_prefetch: bool = False
    prefetch_enabled: bool = True
    backend: TrainBackendConfig = "auto"
    debug: bool = False
    resume_ckpt_path: str | None = None
    fork_ckpt_path: str | None = None
    stop_after: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        if (self.data.native_om4 is None) != (
            self.data.native_om4_proof_sha256 is None
        ):
            raise ValueError("Packed OM4 requires both a cache and its proof checksum")
        if self.data.loading is not None:
            if self.data.loading.prefetch_batches != 1:
                raise ValueError(
                    "Observation replay supports exactly one prefetched update"
                )
            if (
                self.data.native_observations is None
                or self.data.native_observations_proof_sha256 is None
            ):
                raise ValueError(
                    "Rust observation loading requires a verified interval cache"
                )
            if "cm4" in MIXES[self.mix]:
                raise ValueError(
                    "Rust observation training currently supports obs and OM4"
                )
        if self.fresh_examples > self.gradient_accumulation_steps:
            raise ValueError("fresh_examples exceeds gradient_accumulation_steps")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final_learning_rate must not exceed learning_rate")
        if self.resume_ckpt_path and self.fork_ckpt_path:
            raise ValueError("Specify resume or an uncooled main checkpoint fork")
        if any(i < 1 or i > self.main_updates for i in self.milestone_updates):
            raise ValueError("Milestones must lie within the main trajectory")
        if self.model.forcing_mode == "shared_era5":
            if "cm4" in MIXES[self.mix]:
                raise ValueError(
                    "Historical ERA5 conditioning is defined only for obs/OM4"
                )
            if "om4" in MIXES[self.mix] and (
                self.data.era5_cache is None or self.data.era5_proof_sha256 is None
            ):
                raise ValueError("Shared ERA5 OM4 requires a verified interval cache")
        for source in MIXES[self.mix]:
            if source != "obs" and getattr(self.data, source) is None:
                raise ValueError(f"mix={self.mix} requires data.{source}")
        return self

    @property
    def total_updates(self) -> int:
        return self.main_updates + self.cooldown_updates

    def source_at(self, update: int) -> str:
        if not 0 <= update < self.total_updates:
            raise ValueError("Update outside the training plan")
        cycle = MIXES[self.mix]
        return "obs" if update >= self.main_updates else cycle[update % len(cycle)]

    def learning_rate_at(self, update: int) -> float:
        if update < self.main_updates:
            return self.learning_rate
        fraction = (update - self.main_updates + 1) / self.cooldown_updates
        return self.learning_rate + fraction * (
            self.final_learning_rate - self.learning_rate
        )

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)

    def scientific_config(self) -> dict:
        """Operational overrides can change on resume; scientific choices cannot."""
        values = self.model_dump(
            mode="json",
            exclude={
                "experiment",
                "backend",
                "debug",
                "resume_ckpt_path",
                "fork_ckpt_path",
                "stop_after",
                "measure_timing",
                "prefetch_enabled",
            },
        )
        values["seed"] = self.experiment.rand_seed
        return values
