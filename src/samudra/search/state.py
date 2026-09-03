# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated durable state for resumable architecture searches."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceState(BaseModel):
    commit: str
    dirty: bool
    package_version: str


class ProbeState(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: str
    job_id: str
    controller_job_id: str | None = None
    status: Literal["array_submitted", "submitted", "complete", "failed"]


class CandidateResourceState(BaseModel):
    world_size: int = Field(ge=1)
    local_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    effective_global_batch_size: int = Field(ge=1)


class RungState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    epochs: int = Field(ge=1)
    candidates: list[str]
    results: list[dict[str, Any]]
    promoted: list[str]
    advanced: bool
    job_id: str | None = None
    controller_job_id: str | None = None
    submission_stage: Literal["array_submitted", "controller_submitted"] | None = None
    probe: ProbeState | None = None
    resources: dict[str, CandidateResourceState] = Field(default_factory=dict)


class AnchorState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str]
    results: list[dict[str, Any]]
    job_id: str | None = None
    resources: dict[str, CandidateResourceState] = Field(default_factory=dict)


class FailureState(BaseModel):
    model_config = ConfigDict(extra="allow")

    stage: str
    type: str
    message: str
    failed_at: str


class SearchState(BaseModel):
    """Versioned controller state persisted across workers and machines."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    run_id: str
    created_at: str
    status: Literal[
        "prepared", "validating", "running", "partial", "complete", "failed"
    ]
    provenance: ProvenanceState
    anchors: AnchorState
    rungs: list[RungState]
    failure: FailureState | None = None
