# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validated durable state for resumable architecture searches."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProvenanceState(BaseModel):
    """Identity of the code a search runs.

    ``git`` provenance names a commit (from a source checkout or the immutable
    ``SAMUDRA_CODE_COMMIT``). ``package`` provenance is used only for local
    searches run from a regular, non-editable install such as ``pip install
    samudra`` on Colab: the installed distribution version identifies the code,
    and there is no commit.
    """

    source: Literal["git", "package"] = "git"
    commit: str | None = None
    dirty: bool
    package_version: str

    @model_validator(mode="after")
    def _commit_matches_source(self) -> "ProvenanceState":
        if self.source == "git" and self.commit is None:
            raise ValueError("git provenance requires a commit")
        if self.source == "package" and self.commit is not None:
            raise ValueError("package provenance must not claim a commit")
        return self


class ProbeState(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: str
    job_id: str
    controller_job_id: str | None = None
    status: Literal["array_submitted", "submitted", "complete", "failed"]


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


class AnchorState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str]
    results: list[dict[str, Any]]
    job_id: str | None = None


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
