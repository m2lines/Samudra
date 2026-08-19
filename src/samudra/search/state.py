# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Typed, validated durable state for resumable architecture searches."""

import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SearchStatus(StringEnum):
    PREPARED = "prepared"
    VALIDATING = "validating"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


class ProbeStatus(StringEnum):
    ARRAY_SUBMITTED = "array_submitted"
    SUBMITTED = "submitted"
    COMPLETE = "complete"
    FAILED = "failed"


class SubmissionStage(StringEnum):
    ARRAY_SUBMITTED = "array_submitted"
    CONTROLLER_SUBMITTED = "controller_submitted"


class StrictStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProvenanceState(StrictStateModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    package_version: str


class CandidateResult(BaseModel):
    """One candidate/rung outcome plus configured scalar metric columns."""

    model_config = ConfigDict(extra="allow")

    search: str
    search_run: str
    candidate: str
    rung: int = Field(ge=0)
    epochs: int = Field(ge=1)
    fixed: bool
    eligible: bool = False
    error: str | None = None
    output_dir: Path
    code_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")

    def row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class ProbeState(StrictStateModel):
    candidate: str
    job_id: str
    controller_job_id: str | None = None
    status: ProbeStatus
    optimizer_steps: int | None = Field(default=None, ge=0)
    batches_seen: int | None = Field(default=None, ge=0)
    validated_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _completed_probe_updated_parameters(self) -> Self:
        if self.status == ProbeStatus.COMPLETE and not self.optimizer_steps:
            raise ValueError("a complete probe must record an optimizer update")
        return self


class RungState(StrictStateModel):
    index: int = Field(ge=0)
    epochs: int = Field(ge=1)
    candidates: list[str]
    results: list[CandidateResult]
    promoted: list[str]
    advanced: bool
    job_id: str | None = None
    controller_job_id: str | None = None
    submission_stage: SubmissionStage | None = None
    probe: ProbeState | None = None

    @model_validator(mode="after")
    def _validate_progress(self) -> Self:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("rung candidates must be unique")
        if len(self.promoted) != len(set(self.promoted)):
            raise ValueError("promoted candidates must be unique")
        if not set(self.promoted).issubset(self.candidates):
            raise ValueError("promoted candidates must belong to their rung")
        if self.submission_stage is not None and self.job_id is None:
            raise ValueError("a submission stage requires an array job ID")
        if (
            self.submission_stage == SubmissionStage.CONTROLLER_SUBMITTED
            and self.controller_job_id is None
        ):
            raise ValueError("controller submission requires a controller job ID")
        if self.advanced:
            result_names = [result.candidate for result in self.results]
            if sorted(result_names) != sorted(self.candidates):
                raise ValueError("an advanced rung requires one result per candidate")
            if any(
                result.rung != self.index
                or result.epochs != self.epochs
                or result.fixed
                for result in self.results
            ):
                raise ValueError("rung results must match their rung budget and role")
            eligible = {result.candidate for result in self.results if result.eligible}
            if not set(self.promoted).issubset(eligible):
                raise ValueError("only eligible candidates may be promoted")
        return self


class AnchorState(StrictStateModel):
    candidates: list[str]
    results: list[CandidateResult]
    job_id: str | None = None

    @model_validator(mode="after")
    def _validate_candidates(self) -> Self:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("anchor candidates must be unique")
        result_names = [result.candidate for result in self.results]
        if not set(result_names).issubset(self.candidates):
            raise ValueError("anchor results must belong to configured anchors")
        return self


class FailureState(StrictStateModel):
    stage: str
    type: str
    message: str
    failed_at: AwareDatetime
    candidate: str | None = None


class SearchState(StrictStateModel):
    """Versioned controller state persisted across workers and machines."""

    schema_version: Literal[1] = 1
    name: str
    run_id: str
    created_at: AwareDatetime
    status: SearchStatus
    provenance: ProvenanceState
    anchors: AnchorState
    rungs: list[RungState] = Field(min_length=1)
    failure: FailureState | None = None

    _TRANSITIONS: ClassVar[dict[SearchStatus, set[SearchStatus]]] = {
        SearchStatus.PREPARED: {
            SearchStatus.VALIDATING,
            SearchStatus.RUNNING,
            SearchStatus.FAILED,
        },
        SearchStatus.VALIDATING: {SearchStatus.RUNNING, SearchStatus.FAILED},
        SearchStatus.RUNNING: {
            SearchStatus.RUNNING,
            SearchStatus.PARTIAL,
            SearchStatus.COMPLETE,
            SearchStatus.FAILED,
        },
        SearchStatus.PARTIAL: set(),
        SearchStatus.COMPLETE: set(),
        SearchStatus.FAILED: set(),
    }

    @model_validator(mode="after")
    def _validate_search(self) -> Self:
        if [rung.index for rung in self.rungs] != list(range(len(self.rungs))):
            raise ValueError("rung indices must be contiguous and zero-based")
        epochs = [rung.epochs for rung in self.rungs]
        if epochs != sorted(set(epochs)):
            raise ValueError("rung epoch budgets must be strictly increasing")
        if set(self.anchors.candidates).intersection(self.rungs[0].candidates):
            raise ValueError("fixed anchors cannot participate in promotion")
        for current, following in zip(self.rungs, self.rungs[1:], strict=False):
            expected = current.promoted if current.advanced else []
            if following.candidates != expected:
                raise ValueError("each rung must contain exactly the prior promotions")
        for result in self.results():
            if result.search != self.name or result.search_run != self.run_id:
                raise ValueError("candidate result identity must match its search")
        final = self.rungs[-1]
        if any(
            not result.fixed
            or result.rung != final.index
            or result.epochs != final.epochs
            for result in self.anchors.results
        ):
            raise ValueError("anchor results must use the final budget and fixed role")
        if self.status == SearchStatus.FAILED and self.failure is None:
            raise ValueError("a failed search must record its failure")
        if self.status in {SearchStatus.COMPLETE, SearchStatus.PARTIAL}:
            if not self.rungs[-1].advanced:
                raise ValueError("a terminal search requires an advanced final rung")
            if sorted(result.candidate for result in self.anchors.results) != sorted(
                self.anchors.candidates
            ):
                raise ValueError("a terminal search requires one result per anchor")
            rows = self.results()
            if not rows:
                raise ValueError("a terminal search requires candidate results")
            all_eligible = all(result.eligible for result in rows)
            if self.status == SearchStatus.COMPLETE and not all_eligible:
                raise ValueError("a complete search cannot contain ineligible results")
            if self.status == SearchStatus.PARTIAL and all_eligible:
                raise ValueError("a partial search must contain an ineligible result")
        return self

    def results(self) -> list[CandidateResult]:
        return [result for rung in self.rungs for result in rung.results] + [
            *self.anchors.results
        ]

    def transition(self, status: SearchStatus) -> None:
        if status not in self._TRANSITIONS[self.status]:
            raise ValueError(f"Invalid search transition: {self.status} -> {status}")
        self.status = status

    def fail(
        self,
        *,
        stage: str,
        error: BaseException,
        candidate: str | None = None,
    ) -> None:
        self.transition(SearchStatus.FAILED)
        self.failure = FailureState(
            stage=stage,
            type=type(error).__name__,
            message=str(error),
            candidate=candidate,
            failed_at=datetime.datetime.now(datetime.UTC),
        )
