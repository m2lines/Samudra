# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Launch and promote Samudra candidates through successive-halving rungs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
SUMMARY_NAME = "training_summary.json"
CHECKPOINT_RELATIVE_PATH = Path("saved_nets/ckpt.pt")


def _die(message: str) -> None:
    raise ValueError(message)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not slug:
        _die(f"Name has no usable characters: {value!r}")
    return slug


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _die(f"Manifest must contain a mapping: {path}")
    return data


def _require(mapping: dict[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        _die(f"{key!r} must be a {kind.__name__}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _die(f"schema_version must be {SCHEMA_VERSION}")
    _slug(_require(manifest, "name", str))
    rungs = _require(manifest, "rungs", list)
    if not rungs or any(not isinstance(epoch, int) or epoch < 1 for epoch in rungs):
        _die("rungs must be a non-empty list of positive integer total epochs")
    if rungs != sorted(set(rungs)):
        _die("rungs must be strictly increasing and unique")
    fraction = manifest.get("promotion_fraction", 0.5)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        _die("promotion_fraction must be in (0, 1]")
    minimum = manifest.get("minimum_promoted", 1)
    if not isinstance(minimum, int) or minimum < 1:
        _die("minimum_promoted must be a positive integer")
    if manifest.get("mode", "min") not in ("min", "max"):
        _die("mode must be 'min' or 'max'")
    metric = manifest.get("metric", "validation_loss")
    if not isinstance(metric, str) or not metric:
        _die("metric must be a non-empty training-summary key")

    runtime = _require(manifest, "runtime", dict)
    for key in ("output_base", "train_harness", "worker_harness"):
        _require(runtime, key, str)
    slurm = _require(manifest, "slurm", dict)
    for key in ("account", "partition"):
        _require(slurm, key, str)
    time_by_rung = slurm.get("time_by_rung")
    if time_by_rung is not None and (
        not isinstance(time_by_rung, list)
        or len(time_by_rung) != len(rungs)
        or any(not isinstance(value, str) or not value for value in time_by_rung)
    ):
        _die("slurm.time_by_rung must contain one walltime string per rung")

    candidates = _require(manifest, "candidates", list)
    if not candidates:
        _die("candidates must not be empty")
    names: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            _die(f"candidates[{index}] must be a mapping")
        name = _require(candidate, "name", str)
        _slug(name)
        if name in names:
            _die(f"Duplicate candidate name: {name}")
        names.add(name)
        _require(candidate, "config", str)
        if not any(candidate.get(key) for key in ("code_layer", "image_ref")):
            _die(f"Candidate {name!r} needs code_layer or image_ref")
        args = candidate.get("args", [])
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            _die(f"Candidate {name!r} args must be a list of strings")
        commit = candidate.get("code_commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            _die(f"Candidate {name!r} code_commit must be a full 40-character SHA")
    if all(candidate.get("fixed") for candidate in candidates):
        _die("At least one non-fixed candidate is required for promotion")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _die(f"Invalid state file: {path}")
    return data


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _orchestrator_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    """Identify the exact search controller source being snapshotted."""
    environment_commit = os.environ.get("SAMUDRA_CODE_COMMIT")
    try:
        root = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--show-toplevel",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        )
        if dirty and not allow_dirty:
            _die(
                "The Samudra checkout containing the search controller has tracked "
                "changes. Commit them or pass --allow-dirty; the bundle still "
                "records a source checksum when dirty execution is intentional."
            )
        if environment_commit is not None and environment_commit != commit:
            _die(
                f"SAMUDRA_CODE_COMMIT={environment_commit} does not match the "
                f"controller checkout at {commit}"
            )
        return {
            "commit": commit,
            "git_commit": commit,
            "git_root": root,
            "dirty": dirty,
            "package_version": importlib.metadata.version("samudra"),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        if (
            environment_commit is None
            or re.fullmatch(r"[0-9a-f]{40}", environment_commit) is None
        ):
            _die(
                "Could not identify the search controller Git commit. Run from a "
                "checkout or set SAMUDRA_CODE_COMMIT to the immutable package commit."
            )
        return {
            "commit": environment_commit,
            "git_commit": None,
            "git_root": None,
            "dirty": None,
            "package_version": importlib.metadata.version("samudra"),
        }


def _candidate_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["name"]: candidate for candidate in manifest["candidates"]}


def _run_name(search_name: str, candidate_name: str, epoch: int) -> str:
    return f"{_slug(search_name)}--{_slug(candidate_name)}--e{epoch}"


def _rung_output(
    manifest: dict[str, Any], candidate_name: str, rung_index: int
) -> Path:
    epoch = manifest["rungs"][rung_index]
    return Path(manifest["runtime"]["output_base"]) / _run_name(
        manifest["name"], candidate_name, epoch
    )


def _sbatch_output(args: list[str], *, dry_run: bool) -> str:
    print(shlex.join(args))
    if dry_run:
        return "DRY-RUN"
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";")[0]


def _submit_rung(
    manifest_path: Path,
    state_path: Path,
    state: dict[str, Any],
    rung_index: int,
    *,
    dry_run: bool,
) -> None:
    active = state["rungs"][rung_index]["candidates"]
    slurm = _load_yaml(manifest_path)["slurm"]
    bundle_dir = state_path.parent
    worker = bundle_dir / "successive_halving_worker.sbatch"
    controller = bundle_dir / "search_controller.py"
    python = str(slurm.get("python", sys.executable))
    maximum = int(slurm.get("max_concurrent", len(active)))
    array = f"0-{len(active) - 1}%{max(1, maximum)}"
    walltime = (
        slurm["time_by_rung"][rung_index]
        if slurm.get("time_by_rung")
        else slurm.get("time", "04:00:00")
    )
    logs = bundle_dir / "logs"
    logs.mkdir(exist_ok=True)
    exports = ",".join(
        (
            "ALL",
            f"HALVING_PYTHON={python}",
            f"HALVING_SCRIPT={controller}",
            f"HALVING_MANIFEST={manifest_path}",
            f"HALVING_STATE={state_path}",
            f"HALVING_RUNG={rung_index}",
        )
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={_slug(state['name'])}-r{rung_index}",
        f"--array={array}",
        f"--account={slurm['account']}",
        f"--partition={slurm['partition']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={slurm.get('cpus_per_task', 4)}",
        f"--mem={slurm.get('memory', '32G')}",
        f"--gres={slurm.get('gres', 'gpu:rtx6000:1')}",
        f"--time={walltime}",
        f"--output={logs}/r{rung_index}-%A_%a.out",
        f"--error={logs}/r{rung_index}-%A_%a.err",
        f"--export={exports}",
        str(worker),
    ]
    array_job = _sbatch_output(command, dry_run=dry_run)
    state["rungs"][rung_index]["array_job_id"] = array_job
    state["status"] = "running"
    _atomic_json(state_path, state)

    advance = shlex.join(
        [
            python,
            str(controller),
            "advance",
            str(manifest_path),
            str(state_path),
            str(rung_index),
        ]
    )
    dependency = f"afterany:{array_job}"
    if rung_index == len(state["rungs"]) - 1 and state["anchors"].get("array_job_id"):
        dependency += f":{state['anchors']['array_job_id']}"
    controller_command = [
        "sbatch",
        "--parsable",
        f"--job-name={_slug(state['name'])}-promote-r{rung_index}",
        f"--dependency={dependency}",
        f"--account={slurm.get('controller_account', slurm['account'])}",
        f"--partition={slurm.get('controller_partition', 'cs')}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:10:00",
        f"--output={logs}/promote-r{rung_index}-%j.out",
        f"--error={logs}/promote-r{rung_index}-%j.err",
        f"--wrap={advance}",
    ]
    controller_job = _sbatch_output(controller_command, dry_run=dry_run)
    state = _load_state(state_path)
    state["rungs"][rung_index]["controller_job_id"] = controller_job
    _atomic_json(state_path, state)


def _submit_anchors(
    manifest_path: Path,
    state_path: Path,
    state: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    anchors = state["anchors"]["candidates"]
    if not anchors:
        return
    manifest = _load_yaml(manifest_path)
    slurm = manifest["slurm"]
    bundle_dir = state_path.parent
    logs = bundle_dir / "logs"
    logs.mkdir(exist_ok=True)
    maximum = min(len(anchors), int(slurm.get("max_concurrent", len(anchors))))
    python = str(slurm.get("python", sys.executable))
    exports = ",".join(
        (
            "ALL",
            f"HALVING_PYTHON={python}",
            f"HALVING_SCRIPT={bundle_dir / 'search_controller.py'}",
            f"HALVING_MANIFEST={manifest_path}",
            f"HALVING_STATE={state_path}",
            f"HALVING_RUNG={len(manifest['rungs']) - 1}",
            "HALVING_ANCHOR=1",
        )
    )
    walltime = (
        slurm["time_by_rung"][-1]
        if slurm.get("time_by_rung")
        else slurm.get("time", "04:00:00")
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={_slug(state['name'])}-anchors",
        f"--array=0-{len(anchors) - 1}%{max(1, maximum)}",
        f"--account={slurm['account']}",
        f"--partition={slurm['partition']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={slurm.get('cpus_per_task', 4)}",
        f"--mem={slurm.get('memory', '32G')}",
        f"--gres={slurm.get('gres', 'gpu:rtx6000:1')}",
        f"--time={walltime}",
        f"--output={logs}/anchors-%A_%a.out",
        f"--error={logs}/anchors-%A_%a.err",
        f"--export={exports}",
        str(bundle_dir / "successive_halving_worker.sbatch"),
    ]
    state["anchors"]["array_job_id"] = _sbatch_output(command, dry_run=dry_run)
    _atomic_json(state_path, state)


def start(
    manifest_source: Path,
    state_root: Path,
    *,
    dry_run: bool,
    allow_dirty: bool = False,
) -> Path:
    manifest = _load_yaml(manifest_source)
    validate_manifest(manifest)
    provenance = _orchestrator_provenance(allow_dirty=allow_dirty)
    worker_source = Path(manifest["runtime"]["worker_harness"])
    if not worker_source.is_file():
        _die(f"Worker harness does not exist: {worker_source}")
    search_dir = state_root / _slug(manifest["name"])
    if search_dir.exists():
        _die(f"Search state already exists: {search_dir}")
    search_dir.mkdir(parents=True)
    manifest_path = search_dir / "manifest.yaml"
    shutil.copy2(manifest_source, manifest_path)
    controller_path = search_dir / "search_controller.py"
    worker_path = search_dir / "successive_halving_worker.sbatch"
    shutil.copy2(Path(__file__), controller_path)
    shutil.copy2(worker_source, worker_path)
    provenance.update(
        controller_sha256=_file_hash(controller_path),
        worker_sha256=_file_hash(worker_path),
    )
    state_path = search_dir / "state.json"
    initial = [
        candidate["name"]
        for candidate in manifest["candidates"]
        if not candidate.get("fixed")
    ]
    anchors = [
        candidate["name"]
        for candidate in manifest["candidates"]
        if candidate.get("fixed")
    ]
    state = {
        "schema_version": SCHEMA_VERSION,
        "name": manifest["name"],
        "manifest_sha256": _manifest_hash(manifest_path),
        "orchestrator": provenance,
        "status": "prepared",
        "anchors": {"candidates": anchors, "results": []},
        "rungs": [
            {
                "index": index,
                "epochs": epoch,
                "candidates": initial if index == 0 else [],
                "results": [],
                "promoted": [],
                "advanced": False,
            }
            for index, epoch in enumerate(manifest["rungs"])
        ],
    }
    _atomic_json(state_path, state)
    _submit_anchors(manifest_path, state_path, state, dry_run=dry_run)
    state = _load_state(state_path)
    _submit_rung(manifest_path, state_path, state, 0, dry_run=dry_run)
    return state_path


def run_task(
    manifest_path: Path,
    state_path: Path,
    rung_index: int,
    task_index: int,
    *,
    anchor: bool,
) -> None:
    manifest = _load_yaml(manifest_path)
    validate_manifest(manifest)
    state = _load_state(state_path)
    names = (
        state["anchors"]["candidates"]
        if anchor
        else state["rungs"][rung_index]["candidates"]
    )
    if not 0 <= task_index < len(names):
        _die(f"Array index {task_index} is outside 0..{len(names) - 1}")
    candidate_name = names[task_index]
    candidate = _candidate_map(manifest)[candidate_name]
    if code_layer := candidate.get("code_layer"):
        layer_manifest = Path(f"{code_layer}.json")
        if not layer_manifest.is_file():
            _die(f"Code-layer manifest does not exist: {layer_manifest}")
        metadata = json.loads(layer_manifest.read_text(encoding="utf-8"))
        expected_commit = candidate.get("code_commit")
        if expected_commit and metadata.get("code_commit") != expected_commit:
            _die(
                f"Candidate {candidate_name!r} expected commit {expected_commit}, "
                f"but {layer_manifest} records {metadata.get('code_commit')!r}"
            )
    epoch = manifest["rungs"][rung_index]
    output_dir = _rung_output(manifest, candidate_name, rung_index)
    if output_dir.exists():
        _die(f"Refusing to overwrite rung output: {output_dir}")

    runtime = manifest["runtime"]
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIG": candidate["config"],
            "NAME": output_dir.name,
            "OUTPUT_BASE": str(output_dir.parent),
            "WANDB_MODE": str(runtime.get("wandb_mode", "online")),
            "SAMUDRA_SEARCH_NAME": manifest["name"],
            "SAMUDRA_SEARCH_MANIFEST_SHA256": state["manifest_sha256"],
            "SAMUDRA_SEARCH_ORCHESTRATOR_COMMIT": str(state["orchestrator"]["commit"]),
            "SAMUDRA_SEARCH_CANDIDATE": candidate_name,
            "SAMUDRA_SEARCH_RUNG": str(rung_index),
            "SAMUDRA_SEARCH_TARGET_EPOCHS": str(epoch),
        }
    )
    if commit := candidate.get("code_commit"):
        environment["SAMUDRA_SEARCH_CANDIDATE_COMMIT"] = commit
    for key in (
        "data_root",
        "scratch_dir",
        "sif_path",
        "container_tag",
        "container_hash",
    ):
        if value := runtime.get(key):
            environment[key.upper()] = str(value)
    if value := candidate.get("code_layer"):
        environment["CODE_LAYER"] = str(value)
    if value := candidate.get("image_ref"):
        environment["IMAGE_REF"] = str(value)

    overrides = list(candidate.get("args", []))
    overrides.extend(
        (
            f"--epochs={epoch}",
            f"--experiment.wandb.group={manifest['name']}",
        )
    )
    if rung_index > 0 and not anchor:
        previous_output = _rung_output(manifest, candidate_name, rung_index - 1)
        checkpoint = previous_output / CHECKPOINT_RELATIVE_PATH
        if not checkpoint.is_file():
            _die(f"Promotion checkpoint does not exist: {checkpoint}")
        overrides.append(f"--resume_ckpt_path={checkpoint}")
        environment["SAMUDRA_SEARCH_PARENT_CHECKPOINT"] = str(checkpoint)
    environment["ARGS"] = shlex.join(overrides)
    print(
        f"candidate={candidate_name} rung={rung_index} epochs={epoch} "
        f"output={output_dir}",
        flush=True,
    )
    subprocess.run([str(runtime["train_harness"])], env=environment, check=True)


def _read_results(
    manifest: dict[str, Any], state: dict[str, Any], rung_index: int
) -> list[dict[str, Any]]:
    metric = manifest.get("metric", "validation_loss")
    epoch = manifest["rungs"][rung_index]
    results = []
    for name in state["rungs"][rung_index]["candidates"]:
        output = _rung_output(manifest, name, rung_index)
        summary_path = output / SUMMARY_NAME
        result: dict[str, Any] = {
            "candidate": name,
            "code_commit": _candidate_map(manifest)[name].get("code_commit", ""),
            "epoch": epoch,
            "output_dir": str(output),
            "summary": str(summary_path),
            "eligible": False,
        }
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score = summary[metric]
            if summary.get("schema_version") != SCHEMA_VERSION:
                _die("unsupported summary schema")
            if summary.get("epoch") != epoch or not summary.get("complete"):
                _die("rung did not finish its target epoch")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                _die("ranking metric is not finite")
            if not (output / CHECKPOINT_RELATIVE_PATH).is_file():
                _die("latest checkpoint is missing")
            actual_commit = summary.get("provenance", {}).get("code_commit")
            expected_commit = _candidate_map(manifest)[name]["code_commit"]
            if actual_commit != expected_commit:
                _die(
                    f"completed run reports code commit {actual_commit!r}, "
                    f"expected {expected_commit}"
                )
            result.update(eligible=True, score=float(score))
            result.update(
                wandb_id=summary.get("wandb_id"),
                wandb_name=summary.get("wandb_name"),
                slurm_job_id=summary.get("search", {}).get("slurm_job_id"),
                parent_checkpoint=summary.get("search", {}).get("parent_checkpoint"),
                optimizer_steps=summary.get("progress", {}).get("optimizer_steps"),
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            result["error"] = str(error)
        results.append(result)
    return results


def _read_anchor_results(
    manifest: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, Any]]:
    synthetic_state = {
        "rungs": [
            {"candidates": state["anchors"]["candidates"]} for _ in manifest["rungs"]
        ]
    }
    return _read_results(manifest, synthetic_state, len(manifest["rungs"]) - 1)


def _write_leaderboard(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "candidate",
                "code_commit",
                "epoch",
                "score",
                "eligible",
                "output_dir",
                "summary",
                "wandb_id",
                "wandb_name",
                "slurm_job_id",
                "parent_checkpoint",
                "optimizer_steps",
                "error",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)


def advance(
    manifest_path: Path, state_path: Path, rung_index: int, *, dry_run: bool
) -> None:
    manifest = _load_yaml(manifest_path)
    validate_manifest(manifest)
    state = _load_state(state_path)
    if state.get("manifest_sha256") != _manifest_hash(manifest_path):
        _die("Manifest changed after the search started")
    rung = state["rungs"][rung_index]
    if rung.get("advanced"):
        _die(f"Rung {rung_index} was already advanced")
    results = _read_results(manifest, state, rung_index)
    reverse = manifest.get("mode", "min") == "max"
    eligible = sorted(
        (result for result in results if result["eligible"]),
        key=lambda result: result["score"],
        reverse=reverse,
    )
    leaderboard = eligible + [result for result in results if not result["eligible"]]
    if not eligible:
        state["status"] = "failed"
        rung["results"] = results
        _atomic_json(state_path, state)
        _write_leaderboard(
            state_path.parent / f"leaderboard-r{rung_index}.csv", leaderboard
        )
        _die(f"No candidate completed rung {rung_index}")

    competing = eligible
    keep = min(
        len(competing),
        max(
            int(manifest.get("minimum_promoted", 1)),
            math.ceil(len(competing) * float(manifest.get("promotion_fraction", 0.5))),
        ),
    )
    promoted = [result["candidate"] for result in competing[:keep]]
    rung.update(results=results, promoted=promoted, advanced=True)
    _write_leaderboard(
        state_path.parent / f"leaderboard-r{rung_index}.csv", leaderboard
    )

    next_index = rung_index + 1
    if next_index == len(state["rungs"]):
        anchor_results = _read_anchor_results(manifest, state)
        state["anchors"]["results"] = anchor_results
        ranked_anchors = sorted(
            (result for result in anchor_results if result["eligible"]),
            key=lambda result: result["score"],
            reverse=reverse,
        ) + [result for result in anchor_results if not result["eligible"]]
        _write_leaderboard(
            state_path.parent / "leaderboard-anchors.csv", ranked_anchors
        )
        state["status"] = "complete"
        _atomic_json(state_path, state)
        print(f"Search complete. Results: {state_path.parent}")
        return
    if not promoted:
        state["status"] = "failed"
        _atomic_json(state_path, state)
        _die("No candidate was promoted")
    state["rungs"][next_index]["candidates"] = promoted
    _atomic_json(state_path, state)
    _submit_rung(manifest_path, state_path, state, next_index, dry_run=dry_run)


def plan(manifest_path: Path) -> None:
    manifest = _load_yaml(manifest_path)
    validate_manifest(manifest)
    fixed = sum(bool(candidate.get("fixed")) for candidate in manifest["candidates"])
    candidates = len(manifest["candidates"]) - fixed
    fraction = float(manifest.get("promotion_fraction", 0.5))
    minimum = int(manifest.get("minimum_promoted", 1))
    print(f"search={manifest['name']} competing={candidates} fixed_anchors={fixed}")
    for index, epoch in enumerate(manifest["rungs"]):
        print(f"rung={index} total_epochs={epoch} maximum_competing={candidates}")
        candidates = min(candidates, max(minimum, math.ceil(candidates * fraction)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("manifest", type=Path)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("manifest", type=Path)
    start_parser.add_argument("--state-root", type=Path, required=True)
    start_parser.add_argument("--dry-run", action="store_true")
    start_parser.add_argument("--allow-dirty", action="store_true")
    run_parser = subparsers.add_parser("run-task")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("state", type=Path)
    run_parser.add_argument("rung", type=int)
    run_parser.add_argument("task", type=int)
    run_parser.add_argument("--anchor", action="store_true")
    advance_parser = subparsers.add_parser("advance")
    advance_parser.add_argument("manifest", type=Path)
    advance_parser.add_argument("state", type=Path)
    advance_parser.add_argument("rung", type=int)
    advance_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        plan(args.manifest)
    elif args.command == "start":
        print(
            start(
                args.manifest,
                args.state_root,
                dry_run=args.dry_run,
                allow_dirty=args.allow_dirty,
            )
        )
    elif args.command == "run-task":
        run_task(
            args.manifest,
            args.state,
            args.rung,
            args.task,
            anchor=args.anchor,
        )
    elif args.command == "advance":
        advance(args.manifest, args.state, args.rung, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
