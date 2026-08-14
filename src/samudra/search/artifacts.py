# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Portable research records for human and agent analysis."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import s3fs

from samudra.search.config import ArtifactConfig
from samudra.utils.location import LocalLocation, S3Location, UnresolvedLocation

CATALOG_NAME = "artifacts.parquet"
EPOCHS_NAME = "epochs.parquet"
RUN_FILES = (
    "config.yaml",
    "training_summary.json",
    "search_metrics.parquet",
    "experiment.log",
    "error.log",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactSearchConfig(Protocol):
    """Configuration fields shared by publishable search algorithms."""

    name: str


class ArtifactSearch(Protocol):
    """Search-algorithm surface required by artifact publication."""

    @property
    def config(self) -> ArtifactSearchConfig: ...

    run_id: str
    search_dir: Path
    checkpoint: Path

    @staticmethod
    def result_rows(state: dict[str, Any]) -> list[dict[str, Any]]: ...


def atomic_local_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace one local Parquet snapshot used for publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".parquet"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactPublisher:
    """Materialize and mirror the complete inspectable record of a search."""

    def __init__(self, search: ArtifactSearch, config: ArtifactConfig) -> None:
        self.search = search
        self.config = config
        self.destination = config.destination.resolve(
            UnresolvedLocation(path=self.search.run_id)
        )

    def prepare(self) -> None:
        """Refuse to overwrite a previously published search of the same name."""
        if isinstance(self.destination, LocalLocation):
            exists = self.destination.path.exists()
        else:
            assert isinstance(self.destination, S3Location)
            exists = self._s3(self.destination).exists(
                f"{self.destination.bucket}/{self.destination.path}"
            )
        if exists:
            raise ValueError(f"Artifact destination already exists: {self.destination}")

    @property
    def root(self) -> str:
        if self.config.public_url is not None:
            return f"{self.config.public_url.rstrip('/')}/{self.search.run_id}"
        return str(self.destination)

    def publish(self, state: dict[str, Any]) -> None:
        files = self._files(state)
        self._write_epochs(files)
        files = self._files(state)
        catalog = pd.DataFrame(self._catalog_rows(files, state))
        atomic_local_parquet(catalog, self.search.search_dir / CATALOG_NAME)
        files.append((self.search.search_dir / CATALOG_NAME, CATALOG_NAME))
        for source, relative in files:
            self._put(source, relative)

    def _files(self, state: dict[str, Any]) -> list[tuple[Path, str]]:
        """List local search files to publish and their destination-relative keys."""
        files: list[tuple[Path, str]] = []
        for name in (
            "config.yaml",
            "code-layer.json",
            "state.json",
            "results.csv",
            "results.parquet",
            EPOCHS_NAME,
        ):
            path = self.search.search_dir / name
            if path.is_file():
                files.append((path, name))
        for path in sorted((self.search.search_dir / "candidates").glob("*.yaml")):
            files.append((path, f"candidates/{path.name}"))
        for path in sorted((self.search.search_dir / "logs").glob("*")):
            if path.is_file():
                files.append((path, f"logs/{path.name}"))
        for path in sorted((self.search.search_dir / "analysis").rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.search.search_dir)
                files.append((path, str(relative)))

        rows = self.search.result_rows(state)
        for row in rows:
            output = Path(row["output_dir"])
            prefix = f"runs/{output.name}"
            for name in RUN_FILES:
                path = output / name
                if path.is_file():
                    files.append((path, f"{prefix}/{name}"))
            for path in sorted((output / "analysis").rglob("*")):
                if path.is_file():
                    relative = path.relative_to(output)
                    files.append((path, f"{prefix}/{relative}"))
            for checkpoint in self._checkpoints(output, row, state):
                relative = checkpoint.relative_to(output)
                files.append((checkpoint, f"{prefix}/{relative}"))
        return files

    def _checkpoints(
        self, output: Path, row: dict[str, Any], state: dict[str, Any]
    ) -> list[Path]:
        if self.config.checkpoints == "none":
            return []
        if self.config.checkpoints == "all":
            return sorted((output / "saved_nets").glob("*.pt"))
        final_rung = len(state["rungs"]) - 1
        if not row.get("eligible") or int(row["rung"]) != final_rung:
            return []
        best = output / "saved_nets/best_validation_ckpt.pt"
        latest = output / self.search.checkpoint
        return [best] if best.is_file() else [latest]

    def _write_epochs(self, files: list[tuple[Path, str]]) -> None:
        frames = []
        for path, _ in files:
            if path.name == "search_metrics.parquet":
                frames.append(pd.read_parquet(path))
        if frames:
            atomic_local_parquet(
                pd.concat(frames, ignore_index=True),
                self.search.search_dir / EPOCHS_NAME,
            )

    def _catalog_rows(
        self, files: list[tuple[Path, str]], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        runs = {
            Path(row["output_dir"]).name: row for row in self.search.result_rows(state)
        }
        rows = []
        for source, relative in files:
            parts = Path(relative).parts
            run = runs.get(parts[1]) if len(parts) > 1 and parts[0] == "runs" else None
            rows.append(
                {
                    "search": self.search.config.name,
                    "search_run": self.search.run_id,
                    "artifact": relative,
                    "kind": self._kind(relative),
                    "candidate": run.get("candidate") if run else None,
                    "rung": run.get("rung") if run else None,
                    "eligible": run.get("eligible") if run else None,
                    "media_type": self._media_type(source),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                    "uri": self._uri(relative),
                    "public_url": self._public_url(relative),
                }
            )
        return rows

    @staticmethod
    def _kind(relative: str) -> str:
        if relative.endswith("ckpt.pt"):
            return "checkpoint"
        if Path(relative).suffix == ".md":
            return "report"
        if Path(relative).suffix in {".pdf", ".png", ".svg"}:
            return "figure"
        if relative.endswith(".log") or relative.startswith("logs/"):
            return "log"
        if relative.endswith(".parquet") or relative.endswith(".csv"):
            return "metrics"
        if relative.endswith(".yaml") or relative.endswith(".json"):
            return "provenance"
        return "other"

    @staticmethod
    def _media_type(path: Path) -> str:
        return {
            ".csv": "text/csv",
            ".json": "application/json",
            ".log": "text/plain",
            ".md": "text/markdown",
            ".parquet": "application/vnd.apache.parquet",
            ".pt": "application/x-pytorch",
            ".yaml": "application/yaml",
        }.get(path.suffix, "application/octet-stream")

    def _uri(self, relative: str) -> str:
        return str(self.destination.resolve(UnresolvedLocation(path=relative)))

    def _public_url(self, relative: str) -> str | None:
        if self.config.public_url is None:
            return None
        return f"{self.config.public_url.rstrip('/')}/{self.search.run_id}/{relative}"

    def _put(self, source: Path, relative: str) -> None:
        destination = self.destination.resolve(UnresolvedLocation(path=relative))
        if isinstance(destination, LocalLocation):
            destination.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination.path)
            return
        assert isinstance(destination, S3Location)
        filesystem = self._s3(destination)
        filesystem.put_file(source, f"{destination.bucket}/{destination.path}")

    @staticmethod
    def _s3(destination: S3Location) -> s3fs.S3FileSystem:
        return s3fs.S3FileSystem(
            anon=destination.anon,
            endpoint_url=destination.endpoint_url,
        )
