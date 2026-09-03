# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Successive halving from Li et al., JMLR 18 (2018), arXiv:1603.06560."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from samudra.config import SearchRunConfig, TrainConfig
from samudra.search.artifacts import ArtifactPublisher, atomic_local_parquet
from samudra.search.config import (
    AdaptiveDataParallelResourceConfig,
    CandidateConfig,
    SearchConfig,
    SlurmExecutorConfig,
    resource_slug,
)
from samudra.search.executors import (
    Executor,
    LocalExecutor,
    SlurmAllocationExecutor,
    SlurmExecutor,
)
from samudra.search.report import write_search_report
from samudra.search.resources import plan_candidate_resources
from samudra.search.state import SearchState
from samudra.train import Trainer
from samudra.utils.atomic import atomic_path
from samudra.utils.distributed import is_main_process
from samudra.utils.logging import handle_logging, handle_warnings
from samudra.utils.training_summary import (
    SEARCH_WORKER_STATUS_NAME,
    TRAINING_SUMMARY_NAME,
    write_search_worker_status,
)

CHECKPOINT = Path("saved_nets/ckpt.pt")
EXECUTORS: dict[str, type[Executor]] = {
    "local": LocalExecutor,
    "slurm_allocation": SlurmAllocationExecutor,
    "slurm": SlurmExecutor,
}


def _new_run_id(name: str) -> str:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{resource_slug(name)}--{timestamp}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    immutable_commit = os.environ.get("SAMUDRA_CODE_COMMIT")
    if immutable_commit is not None:
        if re.fullmatch(r"[0-9a-fA-F]{40}", immutable_commit) is None:
            raise ValueError("SAMUDRA_CODE_COMMIT must be a full 40-character Git SHA")
        return {
            "commit": immutable_commit.lower(),
            "dirty": False,
            "package_version": importlib.metadata.version("samudra"),
        }

    root = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if dirty and not allow_dirty:
        raise ValueError("Commit tracked changes or set allow_dirty=true")
    return {
        "commit": commit,
        "dirty": dirty,
        "package_version": importlib.metadata.version("samudra"),
    }


class SuccessiveHalving:
    """Train all candidates cheaply and promote the best through larger budgets."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.slug = resource_slug(config.name)
        if config.run_id is None:
            config.run_id = _new_run_id(config.name)
        self.run_id = resource_slug(config.run_id)
        self.search_dir = config.executor.output_dir / self.run_id
        self.config_path = self.search_dir / "config.yaml"
        self.state_path = self.search_dir / "state.json"
        self.results_path = self.search_dir / "results.csv"
        self.results_parquet_path = self.search_dir / "results.parquet"
        self.checkpoint = CHECKPOINT
        self.rungs = config.algorithm.rungs
        self.executor = EXECUTORS[config.executor.type](self)
        self.publisher = (
            ArtifactPublisher(self, config.artifacts)
            if config.artifacts is not None
            else None
        )

    def start(self) -> Path:
        if self.search_dir.exists():
            raise ValueError(f"Search output already exists: {self.search_dir}")
        if self.publisher is not None:
            self.publisher.prepare()
        competing = [item.name for item in self.config.candidates if not item.fixed]
        anchors = [item.name for item in self.config.candidates if item.fixed]
        provenance = _git_provenance(allow_dirty=self.config.allow_dirty)
        code_layer_metadata: dict[str, Any] | None = None
        if isinstance(self.config.executor, SlurmExecutorConfig):
            code_layer = self.config.executor.code_layer
            if code_layer is not None:
                metadata_path = Path(f"{code_layer}.json")
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_metadata, dict):
                    raise ValueError(f"Invalid code-layer manifest: {metadata_path}")
                code_layer_metadata = loaded_metadata
                if code_layer_metadata.get("code_commit") != provenance["commit"]:
                    raise ValueError(
                        f"Code layer contains {code_layer_metadata.get('code_commit')!r}; "
                        f"search controller is {provenance['commit']!r}"
                    )
        resolved_candidates = [
            (
                candidate,
                TrainConfig.from_yaml_and_cli([candidate.config, *candidate.args]),
            )
            for candidate in self.config.candidates
        ]
        self.search_dir.mkdir(parents=True)
        if code_layer_metadata is not None:
            _atomic_json(self.search_dir / "code-layer.json", code_layer_metadata)
        candidate_dir = self.search_dir / "candidates"
        candidate_dir.mkdir()
        for candidate, train_config in resolved_candidates:
            destination = candidate_dir / f"{resource_slug(candidate.name)}.yaml"
            destination.write_text(
                yaml.safe_dump(train_config.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )
            candidate.config = str(destination)
            candidate.args = []
        self.config_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        state = {
            "name": self.config.name,
            "run_id": self.run_id,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "prepared",
            "provenance": provenance,
            "anchors": {"candidates": anchors, "results": []},
            "rungs": [
                {
                    "index": index,
                    "epochs": epochs,
                    "candidates": competing if index == 0 else [],
                    "results": [],
                    "promoted": [],
                    "advanced": False,
                }
                for index, epochs in enumerate(self.rungs)
            ],
        }
        self.write_state(state)
        # Publish stable, queryable schemas immediately. Without these files,
        # the public results URL returns 404 until the first rung completes.
        self._write_results(state)
        self.publish(state)
        if self.publisher is not None and is_main_process():
            print(f"Published search record: {self.publisher.root}", flush=True)
        try:
            gate_anchors = (
                isinstance(self.config.executor, SlurmExecutorConfig)
                and self.config.executor.rung0_probe
            )
            if not gate_anchors:
                self.executor.submit_initial(state)
            else:
                self.executor.submit_rung(self.read_state(), 0)
        except Exception as error:
            state = self.read_state()
            state["status"] = "failed"
            state["failure"] = {
                "stage": "submission",
                "type": type(error).__name__,
                "message": str(error),
                "failed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            self.write_state(state)
            try:
                self._write_results(state)
                self.publish(state)
                self._write_report(state)
                self.publish(state)
            except Exception as publication_error:
                error.add_note(
                    "Failed to render or publish the terminal search record: "
                    f"{publication_error!r}"
                )
            raise
        state = self.read_state()
        self._write_report(state)
        self.publish(state)
        return self.state_path

    def read_state(self) -> dict[str, Any]:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = SearchState.model_validate(value)
        if state.name != self.config.name or state.run_id != self.run_id:
            raise ValueError("Search state identity does not match its configuration")
        if [item.epochs for item in state.rungs] != self.rungs or [
            item.index for item in state.rungs
        ] != list(range(len(self.rungs))):
            raise ValueError("Search state rungs do not match its configuration")
        return state.model_dump(mode="json", exclude_none=True)

    def write_state(self, state: dict[str, Any]) -> None:
        validated = SearchState.model_validate(state)
        _atomic_json(
            self.state_path, validated.model_dump(mode="json", exclude_none=True)
        )

    def publish(self, state: dict[str, Any]) -> None:
        if self.publisher is not None:
            self.publisher.publish(state)

    def candidate(self, name: str) -> CandidateConfig:
        return next(item for item in self.config.candidates if item.name == name)

    def plan_resources(
        self,
        candidates: list[str],
        *,
        gpu_capacity: int,
        candidate_concurrency: int | None = None,
        placeable_world_sizes: set[int] | None = None,
        force_single_gpu: bool = False,
    ) -> dict[str, dict[str, int]]:
        configs = {
            name: TrainConfig.from_yaml_and_cli([self.candidate(name).config])
            for name in candidates
        }
        plans = plan_candidate_resources(
            self.config.resources,
            configs,
            gpu_capacity=gpu_capacity,
            candidate_concurrency=candidate_concurrency,
            placeable_world_sizes=placeable_world_sizes,
            force_single_gpu=force_single_gpu,
        )
        return {name: plan.model_dump(mode="json") for name, plan in plans.items()}

    def output_dir(self, candidate: str, rung: int) -> Path:
        return self.config.executor.output_dir / (
            f"{self.run_id}--{resource_slug(candidate)}--e{self.rungs[rung]}"
        )

    def train_task(
        self, rung: int, task: int, *, anchor: bool, probe: bool = False
    ) -> None:
        state = self.read_state()
        candidates = (
            state["anchors"]["candidates"]
            if anchor
            else state["rungs"][rung]["candidates"]
        )
        try:
            name = candidates[task]
        except IndexError as error:
            raise ValueError(f"Task {task} is outside this rung") from error
        candidate = self.candidate(name)
        if probe and anchor:
            raise ValueError("Fixed anchors cannot be used as rung probes")
        output = (
            self.search_dir / "probe" / resource_slug(name)
            if probe
            else self.output_dir(name, rung)
        )
        if output.exists() and int(os.environ.get("RANK", "0")) == 0:
            raise ValueError(f"Refusing to overwrite {output}")
        args = [candidate.config, *candidate.args]
        train_config = TrainConfig.from_yaml_and_cli(args)
        resource_group = (
            state["anchors"]["resources"]
            if anchor
            else state["rungs"][rung]["resources"]
        )
        resource_values = resource_group.get(name)
        if resource_values is None:
            resource_values = {
                "world_size": 1,
                "local_batch_size": train_config.batch_size,
                "gradient_accumulation_steps": (
                    train_config.gradient_accumulation_steps
                ),
                "effective_global_batch_size": (
                    train_config.batch_size * train_config.gradient_accumulation_steps
                ),
            }
        train_config.gradient_accumulation_steps = resource_values[
            "gradient_accumulation_steps"
        ]
        train_config.epochs = self.rungs[rung]
        if (
            train_config.scheduler is not None
            and train_config.scheduler.target_epochs is None
        ):
            # Every rung and fixed anchor must share one LR horizon. Otherwise
            # restoring a scheduler checkpoint also restores the short rung's
            # T_max and corrupts the promoted candidate's learning rate.
            train_config.scheduler.target_epochs = self.rungs[-1]
        train_config.experiment.name = output.name
        train_config.experiment.base_output_dir = str(output.parent)
        parent_checkpoint: Path | None = None
        if rung > 0 and not anchor and not probe:
            parent_checkpoint = self.output_dir(name, rung - 1) / CHECKPOINT
            if not parent_checkpoint.is_file():
                raise ValueError(f"Missing promotion checkpoint: {parent_checkpoint}")
            train_config.resume_ckpt_path = str(parent_checkpoint)
        train_config.experiment.search = SearchRunConfig(
            name=self.config.name,
            run_id=self.run_id,
            candidate=name,
            rung=rung,
            target_epochs=self.rungs[rung],
            objective=self.config.objective.metric,
            executor=self.config.executor.type,
            code_commit=os.environ.get(
                "SAMUDRA_CODE_COMMIT",
                state.get("provenance", {}).get("commit"),
            ),
            code_layer_sha256=os.environ.get("SAMUDRA_CODE_LAYER_SHA256"),
            container_image_ref=os.environ.get("SAMUDRA_CONTAINER_IMAGE_REF"),
            container_git_commit=os.environ.get("SAMUDRA_CONTAINER_GIT_COMMIT"),
            artifacts_uri=(self.publisher.root if self.publisher is not None else None),
            job_id=os.environ.get("SLURM_JOB_ID", self.config.executor.type),
            parent_checkpoint=(
                str(parent_checkpoint) if parent_checkpoint is not None else None
            ),
            world_size=resource_values["world_size"],
            local_batch_size=resource_values["local_batch_size"],
            gradient_accumulation_steps=resource_values["gradient_accumulation_steps"],
            effective_global_batch_size=resource_values["effective_global_batch_size"],
            adaptive_data_parallel=isinstance(
                self.config.resources, AdaptiveDataParallelResourceConfig
            ),
        )
        train_config.experiment.wandb.group = self.run_id
        if probe:
            train_config.experiment.wandb.mode = "disabled"
        tags = train_config.experiment.wandb.tags or []
        train_config.experiment.wandb.tags = list(
            dict.fromkeys([*tags, "search", self.slug, resource_slug(name)])
        )
        train_config.prepare_output_dirs()
        handle_logging(train_config.debug, train_config.experiment.output_dir)
        handle_warnings()
        status_details = {
            "candidate": name,
            "rung": rung,
            "target_epochs": self.rungs[rung],
            "job_id": os.environ.get("SLURM_JOB_ID", self.config.executor.type),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "probe": probe,
        }
        if is_main_process():
            write_search_worker_status(output, "launched", **status_details)
        try:
            trainer = Trainer(train_config)
            if is_main_process():
                write_search_worker_status(output, "initialized", **status_details)
            if probe:
                trainer.probe_optimizer_step()
            else:
                trainer.run()
        except Exception as error:
            if is_main_process():
                write_search_worker_status(
                    output,
                    "failed",
                    **status_details,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            raise
        if is_main_process():
            write_search_worker_status(
                output,
                "completed",
                **status_details,
                batches_seen=trainer.num_batches_seen,
                optimizer_steps=trainer.train_progress.optimizer_steps,
            )

    def release_probe(self, rung: int) -> None:
        """Release a Slurm rung only after its probe proves an optimizer update."""
        state = self.read_state()
        controller_commit = os.environ.get("SAMUDRA_CODE_COMMIT")
        expected_commit = state.get("provenance", {}).get("commit")
        if controller_commit != expected_commit:
            raise ValueError(
                f"Search was started at {expected_commit}; probe controller "
                f"reports {controller_commit!r}"
            )
        current = state["rungs"][rung]
        probe = current.get("probe")
        if not isinstance(probe, dict):
            raise ValueError(f"Rung {rung} has no configured probe")
        name = probe["candidate"]
        status_path = (
            self.search_dir / "probe" / resource_slug(name) / SEARCH_WORKER_STATUS_NAME
        )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("stage") != "completed":
                raise ValueError(
                    f"worker stopped at stage {status.get('stage')!r}: "
                    f"{status.get('error', 'no worker error recorded')}"
                )
            if int(status.get("optimizer_steps", 0)) < 1:
                raise ValueError("worker completed without an optimizer update")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            state["status"] = "failed"
            state["failure"] = {
                "stage": "rung_probe",
                "type": type(error).__name__,
                "message": str(error),
                "candidate": name,
                "failed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            probe["status"] = "failed"
            self.write_state(state)
            self._write_results(state)
            self.publish(state)
            try:
                self._write_report(state)
                self.publish(state)
            except Exception as report_error:
                error.add_note(f"Failed to render or publish report: {report_error!r}")
            raise ValueError(f"Rung {rung} probe failed: {error}") from error
        probe.update(
            status="complete",
            optimizer_steps=status["optimizer_steps"],
            batches_seen=status.get("batches_seen"),
            validated_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        self.write_state(state)
        if not isinstance(self.executor, SlurmExecutor):
            raise TypeError("Rung probes are only supported by the Slurm executor")
        self.executor.submit_anchors(state)
        self.executor.submit_validated_rung(state, rung)
        self.publish(self.read_state())

    def _result(self, name: str, rung: int) -> dict[str, Any]:
        output = self.output_dir(name, rung)
        result: dict[str, Any] = {
            "search": self.config.name,
            "search_run": self.run_id,
            "candidate": name,
            "rung": rung,
            "epochs": self.rungs[rung],
            "fixed": self.candidate(name).fixed,
            "eligible": False,
            "error": None,
            "output_dir": str(output),
            "code_commit": self.read_state().get("provenance", {}).get("commit"),
        }
        try:
            summary = json.loads(
                (output / TRAINING_SUMMARY_NAME).read_text(encoding="utf-8")
            )
            if summary["epoch"] != self.rungs[rung] or not summary["complete"]:
                raise ValueError("target epoch was not completed")
            expected_commit = self.read_state().get("provenance", {}).get("commit")
            if summary.get("code_commit") != expected_commit:
                raise ValueError(
                    f"training commit {summary.get('code_commit')!r} does not match "
                    f"search commit {expected_commit!r}"
                )
            if not (output / CHECKPOINT).is_file():
                raise ValueError("checkpoint is missing")
            for metric in self.config.metrics:
                score = summary[metric]
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"metric {metric!r} is not finite")
                result[metric] = float(score)
            result.update(
                eligible=True,
                optimizer_steps=summary.get("optimizer_steps"),
                wandb_id=summary.get("wandb_id"),
                wandb_name=summary.get("wandb_name"),
                executor=summary.get("executor"),
                job_id=summary.get("job_id"),
                code_layer_sha256=summary.get("code_layer_sha256"),
                container_image_ref=summary.get("container_image_ref"),
                container_git_commit=summary.get("container_git_commit"),
                hostname=summary.get("hostname"),
                torch_version=summary.get("torch_version"),
                device=summary.get("device"),
                cuda_device_name=summary.get("cuda_device_name"),
                world_size=summary.get("world_size"),
                completed_at=summary.get("completed_at"),
                artifacts_uri=summary.get("artifacts_uri"),
                parent_checkpoint=summary.get("parent_checkpoint"),
                train_seconds=summary.get("epoch_train_seconds"),
                validation_seconds=summary.get("epoch_validation_seconds"),
                total_seconds=summary.get("epoch_total_seconds"),
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            result["error"] = str(error)
            result.update(self._scheduler_failure_context(name, rung))
        status_path = output / SEARCH_WORKER_STATUS_NAME
        if status_path.is_file():
            worker_status = json.loads(status_path.read_text(encoding="utf-8"))
            result.update(
                worker_stage=worker_status.get("stage"),
                worker_updated_at=worker_status.get("updated_at"),
                worker_optimizer_steps=worker_status.get("optimizer_steps"),
                worker_batches_seen=worker_status.get("batches_seen"),
                worker_error_type=worker_status.get("error_type"),
                worker_error=worker_status.get("error"),
                worker_status_log=f"runs/{output.name}/{SEARCH_WORKER_STATUS_NAME}",
            )
        return result

    def _scheduler_failure_context(self, name: str, rung: int) -> dict[str, Any]:
        """Locate the bounded task logs that explain an ineligible result."""
        state = self.read_state()
        fixed = self.candidate(name).fixed
        group = state["anchors"] if fixed else state["rungs"][rung]
        candidates = group.get("candidates", [])
        job_id = group.get("job_id")
        if job_id is None or name not in candidates:
            return {}
        task = candidates.index(name)
        label = "anchors" if fixed else f"r{rung}"
        context: dict[str, Any] = {"scheduler_task_id": f"{job_id}_{task}"}
        for suffix, field in (("out", "stdout"), ("err", "stderr")):
            path = self.search_dir / "logs" / f"{label}-{job_id}_{task}.{suffix}"
            if not path.is_file():
                continue
            context[f"scheduler_{field}_log"] = str(path.relative_to(self.search_dir))
        return context

    def _write_results(self, state: dict[str, Any]) -> None:
        rows = self.result_rows(state)
        columns = list(
            dict.fromkeys(
                [
                    "search",
                    "search_run",
                    "candidate",
                    "rung",
                    "epochs",
                    "fixed",
                    "eligible",
                    "error",
                    "output_dir",
                    "code_commit",
                    *self.config.metrics,
                    *(key for row in rows for key in row),
                ]
            )
        )
        frame = pd.DataFrame(rows, columns=columns)
        atomic_local_parquet(frame, self.results_parquet_path)
        with atomic_path(self.results_path, suffix=".csv") as temporary:
            frame.to_csv(temporary, index=False)

    def _write_report(self, state: dict[str, Any]) -> None:
        """Render the human-facing report after critical scheduler operations."""
        write_search_report(self, state)

    @staticmethod
    def result_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            result for rung in state["rungs"] for result in rung.get("results", [])
        ] + state["anchors"].get("results", [])

    def advance(self, rung: int) -> None:
        state = self.read_state()
        if isinstance(self.config.executor, SlurmExecutorConfig):
            controller_commit = os.environ.get("SAMUDRA_CODE_COMMIT")
            expected_commit = state.get("provenance", {}).get("commit")
            if controller_commit != expected_commit:
                raise ValueError(
                    f"Search was started at {expected_commit}; promotion controller "
                    f"reports {controller_commit!r}"
                )
        current = state["rungs"][rung]
        if current["advanced"]:
            # Publication and scheduler submission are external operations. A
            # controller retry must be able to finish either after a transient
            # upload/submission failure without recomputing the ranking.
            self._write_results(state)
            self.publish(state)
            next_rung = rung + 1
            if next_rung == len(self.rungs):
                self._write_report(state)
                self.publish(state)
                return
            if "controller_job_id" not in state["rungs"][next_rung]:
                self.executor.submit_rung(state, next_rung)
            state = self.read_state()
            self._write_report(state)
            self.publish(state)
            return
        results = [self._result(name, rung) for name in current["candidates"]]
        eligible = pd.DataFrame([row for row in results if row["eligible"]])
        if eligible.empty:
            current["results"] = results
            state["status"] = "failed"
            state["failure"] = {
                "stage": f"rung_{rung}",
                "type": "NoEligibleCandidates",
                "message": f"No candidate completed rung {rung}",
                "failed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            self.write_state(state)
            self._write_results(state)
            self.publish(state)
            self._write_report(state)
            self.publish(state)
            raise ValueError(f"No candidate completed rung {rung}")
        ascending = self.config.objective.mode == "min"
        ranked = eligible.sort_values(self.config.objective.metric, ascending=ascending)
        keep = min(
            len(ranked),
            max(
                self.config.algorithm.minimum_promoted,
                math.ceil(len(ranked) * self.config.algorithm.promotion_fraction),
            ),
        )
        promoted = ranked.head(keep)["candidate"].tolist()
        current.update(results=results, promoted=promoted, advanced=True)
        next_rung = rung + 1
        if next_rung == len(self.rungs):
            state["anchors"]["results"] = [
                self._result(name, rung) for name in state["anchors"]["candidates"]
            ]
            all_results = self.result_rows(state)
            state["status"] = (
                "complete"
                if all(result["eligible"] for result in all_results)
                else "partial"
            )
            self.write_state(state)
            self._write_results(state)
            self.publish(state)
            self._write_report(state)
            self.publish(state)
            if is_main_process():
                print(f"Search {state['status']}: {self.results_path}", flush=True)
            return
        state["rungs"][next_rung]["candidates"] = promoted
        self.write_state(state)
        self._write_results(state)
        self.publish(state)
        self.executor.submit_rung(state, next_rung)
        state = self.read_state()
        self._write_report(state)
        self.publish(state)
