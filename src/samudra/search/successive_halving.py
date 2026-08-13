# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Successive halving from Li et al., JMLR 18 (2018), arXiv:1603.06560."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from samudra.config import SearchRunConfig, TrainConfig
from samudra.search.config import CandidateConfig, SearchConfig, SlurmExecutorConfig
from samudra.search.executors import Executor, LocalExecutor, SlurmExecutor
from samudra.train import Trainer
from samudra.utils.distributed import is_main_process
from samudra.utils.logging import handle_logging, handle_warnings

SUMMARY_NAME = "training_summary.json"
CHECKPOINT = Path("saved_nets/ckpt.pt")
EXECUTORS: dict[str, type[Executor]] = {
    "local": LocalExecutor,
    "slurm": SlurmExecutor,
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError(f"Name has no usable characters: {value!r}")
    return slug


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
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
        self.slug = _slug(config.name)
        self.search_dir = config.executor.output_dir / self.slug
        self.config_path = self.search_dir / "config.yaml"
        self.state_path = self.search_dir / "state.json"
        self.results_path = self.search_dir / "results.csv"
        self.rungs = config.algorithm.rungs
        self.executor = EXECUTORS[config.executor.type](self)

    def start(self) -> Path:
        if self.search_dir.exists():
            raise ValueError(f"Search output already exists: {self.search_dir}")
        competing = [item.name for item in self.config.candidates if not item.fixed]
        anchors = [item.name for item in self.config.candidates if item.fixed]
        provenance = _git_provenance(allow_dirty=self.config.allow_dirty)
        if isinstance(self.config.executor, SlurmExecutorConfig):
            code_layer = self.config.executor.code_layer
            if code_layer is not None:
                metadata_path = Path(f"{code_layer}.json")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("code_commit") != provenance["commit"]:
                    raise ValueError(
                        f"Code layer contains {metadata.get('code_commit')!r}; "
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
        candidate_dir = self.search_dir / "candidates"
        candidate_dir.mkdir()
        for candidate, train_config in resolved_candidates:
            destination = candidate_dir / f"{_slug(candidate.name)}.yaml"
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
        self.executor.submit_anchors(state)
        self.executor.submit_rung(self.read_state(), 0)
        return self.state_path

    def read_state(self) -> dict[str, Any]:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid search state: {self.state_path}")
        return value

    def write_state(self, state: dict[str, Any]) -> None:
        _atomic_json(self.state_path, state)

    def candidate(self, name: str) -> CandidateConfig:
        return next(item for item in self.config.candidates if item.name == name)

    def output_dir(self, candidate: str, rung: int) -> Path:
        return self.config.executor.output_dir / (
            f"{self.slug}--{_slug(candidate)}--e{self.rungs[rung]}"
        )

    def train_task(self, rung: int, task: int, *, anchor: bool) -> None:
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
        output = self.output_dir(name, rung)
        if output.exists() and "RANK" not in os.environ:
            raise ValueError(f"Refusing to overwrite {output}")
        args = [candidate.config, *candidate.args]
        train_config = TrainConfig.from_yaml_and_cli(args)
        train_config.epochs = self.rungs[rung]
        train_config.experiment.name = output.name
        train_config.experiment.base_output_dir = str(output.parent)
        parent_checkpoint: Path | None = None
        if rung > 0 and not anchor:
            parent_checkpoint = self.output_dir(name, rung - 1) / CHECKPOINT
            if not parent_checkpoint.is_file():
                raise ValueError(f"Missing promotion checkpoint: {parent_checkpoint}")
            train_config.resume_ckpt_path = str(parent_checkpoint)
        train_config.experiment.search = SearchRunConfig(
            name=self.config.name,
            candidate=name,
            rung=rung,
            target_epochs=self.rungs[rung],
            objective=self.config.objective.metric,
            executor=self.config.executor.type,
            code_commit=os.environ.get(
                "SAMUDRA_CODE_COMMIT",
                state.get("provenance", {}).get("commit"),
            ),
            job_id=os.environ.get("SLURM_JOB_ID", self.config.executor.type),
            parent_checkpoint=(
                str(parent_checkpoint) if parent_checkpoint is not None else None
            ),
        )
        train_config.experiment.wandb.group = self.config.name
        tags = train_config.experiment.wandb.tags or []
        train_config.experiment.wandb.tags = list(
            dict.fromkeys([*tags, "search", self.slug, _slug(name)])
        )
        train_config.prepare_output_dirs()
        handle_logging(train_config.debug, train_config.experiment.output_dir)
        handle_warnings()
        trainer = Trainer(train_config)
        trainer.run()

    def _result(self, name: str, rung: int) -> dict[str, Any]:
        output = self.output_dir(name, rung)
        result: dict[str, Any] = {
            "search": self.config.name,
            "candidate": name,
            "rung": rung,
            "epochs": self.rungs[rung],
            "fixed": self.candidate(name).fixed,
            "eligible": False,
            "output_dir": str(output),
            "code_commit": self.read_state().get("provenance", {}).get("commit"),
        }
        try:
            summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
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
        return result

    def _write_results(self, state: dict[str, Any]) -> None:
        rows = [
            result for rung in state["rungs"] for result in rung.get("results", [])
        ] + state["anchors"].get("results", [])
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.search_dir,
            prefix=".results.",
            suffix=".csv",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pd.DataFrame(rows).to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.results_path)

    def advance(self, rung: int) -> None:
        state = self.read_state()
        if isinstance(self.config.executor, SlurmExecutorConfig):
            controller = _git_provenance(allow_dirty=False)
            expected_commit = state.get("provenance", {}).get("commit")
            if controller["commit"] != expected_commit:
                raise ValueError(
                    f"Search was started at {expected_commit}; promotion controller "
                    f"is running {controller['commit']}"
                )
        current = state["rungs"][rung]
        if current["advanced"]:
            raise ValueError(f"Rung {rung} was already advanced")
        results = [self._result(name, rung) for name in current["candidates"]]
        eligible = pd.DataFrame([row for row in results if row["eligible"]])
        if eligible.empty:
            current["results"] = results
            state["status"] = "failed"
            self.write_state(state)
            self._write_results(state)
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
            state["status"] = "complete"
            self.write_state(state)
            self._write_results(state)
            if is_main_process():
                print(f"Search complete: {self.results_path}", flush=True)
            return
        state["rungs"][next_rung]["candidates"] = promoted
        self.write_state(state)
        self._write_results(state)
        self.executor.submit_rung(state, next_rung)


def build_search(config: SearchConfig) -> SuccessiveHalving:
    """Build the configured algorithm; add future algorithms at this boundary."""
    if config.algorithm.type == "successive_halving":
        return SuccessiveHalving(config)
    raise AssertionError(f"Unhandled search algorithm: {config.algorithm.type}")
