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


class SingleGpuResourceConfig(BaseConfig):
    strategy: Literal["single_gpu"] = "single_gpu"


class AdaptiveDataParallelResourceConfig(BaseConfig):
    strategy: Literal["adaptive_data_parallel"] = "adaptive_data_parallel"
    max_gpus_per_candidate: int = Field(ge=2)
    effective_global_batch_size: int = Field(ge=1)
    allowed_world_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])

    @model_validator(mode="after")
    def _validate_world_sizes(self) -> Self:
        if self.allowed_world_sizes != sorted(set(self.allowed_world_sizes)):
            raise ValueError("allowed_world_sizes must be increasing and unique")
        if not self.allowed_world_sizes or self.allowed_world_sizes[0] != 1:
            raise ValueError("allowed_world_sizes must start with 1")
        if any(size < 1 for size in self.allowed_world_sizes):
            raise ValueError("allowed_world_sizes must contain positive values")
        return self


ResourceConfig = Annotated[
    SingleGpuResourceConfig | AdaptiveDataParallelResourceConfig,
    Field(discriminator="strategy"),
]


class LocalExecutorConfig(BaseConfig):
    type: Literal["local"] = "local"
    output_dir: Path
    max_concurrent: int | None = Field(default=None, ge=1)
    dry_run: bool = False


class SlurmAllocationExecutorConfig(BaseConfig):
    type: Literal["slurm_allocation"] = "slurm_allocation"
    output_dir: Path
    max_concurrent: int | None = Field(default=None, ge=1)
    dry_run: bool = False


class SlurmExecutorConfig(BaseConfig):
    type: Literal["slurm"] = "slurm"
    output_dir: Path
    harness: Path
    account: str
    partition: str
    controller_partition: str = "cs"
    controller_cpus_per_task: int = Field(default=1, ge=1)
    controller_memory: str = "4G"
    controller_time: str = "01:00:00"
    qos: str | None = None
    constraint: str | None = None
    controller_qos: str | None = None
    controller_constraint: str | None = None
    controller_gres: str | None = None
    python: str = "python"
    cpus_per_task: int = Field(default=4, ge=1)
    memory: str = "32G"
    gres: str = "gpu:1"
    time: str = "04:00:00"
    time_by_rung: list[str] | None = None
    max_concurrent: int = Field(default=8, ge=1)
    rung0_probe: bool = False
    data_root: Path | None = None
    scratch_dir: Path | None = None
    sif_path: Path | None = None
    image_ref: str | None = None
    code_layer: Path | None = None
    code_dir: Path | None = None
    apptainer_module: str | None = None
    dry_run: bool = False

    @model_validator(mode="after")
    def _code_source_is_unambiguous(self) -> Self:
        if self.code_layer is not None and self.code_dir is not None:
            raise ValueError("Specify only one of code_layer or code_dir")
        return self


ExecutorConfig = Annotated[
    LocalExecutorConfig | SlurmAllocationExecutorConfig | SlurmExecutorConfig,
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
    logs: Literal["none", "all"] = "none"
    public_url: str | None = None


class SearchConfig(TopLevelConfig):
    """Compare training configurations using adaptive resource allocation."""

    name: str
    run_id: str | None = None
    algorithm: AlgorithmConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    metrics: list[str] = Field(default_factory=lambda: ["validation_loss"])
    resources: ResourceConfig = Field(default_factory=SingleGpuResourceConfig)
    candidates: list[CandidateConfig] = Field(min_length=1)
    executor: ExecutorConfig
    artifacts: ArtifactConfig | None = None
    allow_dirty: bool = False

    def build(self) -> "SuccessiveHalving":
        """Build the configured search algorithm."""
        from samudra.search.successive_halving import SuccessiveHalving

        return SuccessiveHalving(self)

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
        competing = sum(not candidate.fixed for candidate in self.candidates)
        if self.algorithm.minimum_promoted > competing:
            raise ValueError(
                "algorithm.minimum_promoted cannot exceed the number of "
                f"non-fixed candidates ({competing})"
            )
        if self.objective.metric not in self.metrics:
            raise ValueError("objective.metric must be included in metrics")
        if (
            isinstance(self.executor, SlurmExecutorConfig)
            and self.executor.time_by_rung is not None
            and len(self.executor.time_by_rung) != len(self.algorithm.rungs)
        ):
            raise ValueError("executor.time_by_rung must have one value per rung")
        if isinstance(self.executor, SlurmExecutorConfig) and self.allow_dirty:
            raise ValueError(
                "allow_dirty is not supported by the submitting Slurm executor; "
                "queued searches require immutable code provenance"
            )
        if isinstance(
            self.resources, AdaptiveDataParallelResourceConfig
        ) and isinstance(self.executor, SlurmExecutorConfig):
            raise ValueError(
                "adaptive_data_parallel requires the local or slurm_allocation "
                "executor; submitted Slurm arrays do not share an allocation"
            )
        return self
