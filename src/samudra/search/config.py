# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated configuration for architecture searches."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, model_validator

from samudra.config_base import BaseConfig, TopLevelConfig
from samudra.utils.location import LocalLocation, S3Location

if TYPE_CHECKING:
    from samudra.search.successive_halving import SuccessiveHalving


def resource_slug(value: str) -> str:
    """Convert a display name to the identifier used for search resources."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError(f"Name has no usable characters: {value!r}")
    return slug


class ObjectiveConfig(BaseConfig):
    metric: str = "validation_loss"
    mode: Literal["min", "max"] = "min"


class CandidateConfig(BaseConfig):
    name: str
    config: str
    args: list[str] = Field(default_factory=list)
    fixed: bool = False


class SuccessiveHalvingConfig(BaseConfig):
    type: Literal["successive_halving"] = "successive_halving"
    rungs: list[int] = Field(min_length=1)
    promotion_fraction: float = Field(default=0.5, gt=0, le=1)
    minimum_promoted: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _rungs_increase(self) -> Self:
        if self.rungs != sorted(set(self.rungs)):
            raise ValueError("rungs must be strictly increasing and unique")
        if any(epoch < 1 for epoch in self.rungs):
            raise ValueError("rungs must contain positive total epoch budgets")
        return self


class LocalExecutorConfig(BaseConfig):
    type: Literal["local"] = "local"
    output_dir: Path
    dry_run: bool = False


class SlurmExecutorConfig(BaseConfig):
    type: Literal["slurm"] = "slurm"
    output_dir: Path
    harness: Path
    account: str
    partition: str
    controller_partition: str = "cs"
    python: str = "python"
    cpus_per_task: int = Field(default=4, ge=1)
    memory: str = "32G"
    gres: str = "gpu:1"
    time: str = "04:00:00"
    time_by_rung: list[str] | None = None
    max_concurrent: int = Field(default=8, ge=1)
    data_root: Path | None = None
    scratch_dir: Path | None = None
    sif_path: Path | None = None
    image_ref: str | None = None
    code_layer: Path | None = None
    apptainer_module: str | None = None
    dry_run: bool = False


ExecutorConfig = Annotated[
    LocalExecutorConfig | SlurmExecutorConfig,
    Field(discriminator="type"),
]
AlgorithmConfig = SuccessiveHalvingConfig

ArtifactDestination = Annotated[
    LocalLocation | S3Location,
    Field(discriminator="type"),
]


class ArtifactConfig(BaseConfig):
    """Durable, executor-independent publication of a search record."""

    destination: ArtifactDestination
    checkpoints: Literal["none", "final", "all"] = "final"
    public_url: str | None = None


class SearchConfig(TopLevelConfig):
    """Compare training configurations using adaptive resource allocation."""

    name: str
    run_id: str | None = None
    algorithm: AlgorithmConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    metrics: list[str] = Field(default_factory=lambda: ["validation_loss"])
    candidates: list[CandidateConfig] = Field(min_length=1)
    executor: ExecutorConfig
    artifacts: ArtifactConfig | None = None
    allow_dirty: bool = False

    def build(self) -> "SuccessiveHalving":
        """Build the configured search algorithm."""
        from samudra.search.successive_halving import SuccessiveHalving

        if self.algorithm.type == "successive_halving":
            return SuccessiveHalving(self)
        raise AssertionError(f"Unhandled search algorithm: {self.algorithm.type}")

    @model_validator(mode="after")
    def _validate_search(self) -> Self:
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")
        slugs = [resource_slug(name) for name in names]
        if len(slugs) != len(set(slugs)):
            raise ValueError("candidate names must be unique after slug normalization")
        if all(candidate.fixed for candidate in self.candidates):
            raise ValueError("at least one candidate must participate in promotion")
        if self.objective.metric not in self.metrics:
            raise ValueError("objective.metric must be included in metrics")
        if (
            isinstance(self.executor, SlurmExecutorConfig)
            and self.executor.time_by_rung is not None
            and len(self.executor.time_by_rung) != len(self.algorithm.rungs)
        ):
            raise ValueError("executor.time_by_rung must have one value per rung")
        return self
