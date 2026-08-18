# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Slurm arrays and dependent promotion jobs."""

import shlex
import subprocess
from typing import Any

from samudra.search.config import SlurmExecutorConfig
from samudra.search.executors.base import Executor


class SlurmExecutor(Executor):
    @property
    def config(self) -> SlurmExecutorConfig:
        config = self.search.config.executor
        assert isinstance(config, SlurmExecutorConfig)
        return config

    def _submit(self, command: list[str]) -> str:
        print(shlex.join(command), flush=True)
        if self.config.dry_run:
            return "DRY-RUN"
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        return result.stdout.strip().split(";")[0]

    def _controller_command(self, *args: str) -> str:
        """Build a controller command with the worker's container runtime."""
        command = shlex.join([self.config.python, *args])
        if self.config.apptainer_module is None:
            return command
        module = shlex.quote(self.config.apptainer_module)
        return shlex.join(["bash", "-lc", f"module load {module} && {command}"])

    def _exports(self, rung: int, *, anchor: bool, worker_command: str = "task") -> str:
        values = {
            "ALL": None,
            "CONFIG": str(self.search.config_path),
            "OUTPUT_BASE": str(self.config.output_dir),
            "NAME": f"{self.search.run_id}-r{rung}",
            "SAMUDRA_MODULE": "samudra.search.worker",
            "SAMUDRA_MODULE_ARGS": shlex.join(
                [
                    worker_command,
                    str(self.search.config_path),
                    str(self.search.state_path),
                    str(rung),
                    *(["--anchor"] if anchor and worker_command == "task" else []),
                ]
            ),
            "SAMUDRA_MANAGE_RUN_DIR": "0",
        }
        optional = {
            "DATA_ROOT": self.config.data_root,
            "SCRATCH_DIR": self.config.scratch_dir,
            "SIF_PATH": self.config.sif_path,
            "IMAGE_REF": self.config.image_ref,
            "CODE_LAYER": self.config.code_layer,
            "APPTAINER_MODULE": self.config.apptainer_module,
        }
        values.update({key: str(value) for key, value in optional.items() if value})
        return ",".join(
            key if value is None else f"{key}={value}" for key, value in values.items()
        )

    def _array(
        self,
        state: dict[str, Any],
        rung: int,
        *,
        anchor: bool,
        task_start: int = 0,
        task_end: int | None = None,
        label: str | None = None,
        worker_command: str = "task",
    ) -> str:
        candidates = (
            state["anchors"]["candidates"]
            if anchor
            else state["rungs"][rung]["candidates"]
        )
        if task_end is None:
            task_end = len(candidates) - 1
        if not 0 <= task_start <= task_end < len(candidates):
            raise ValueError(
                f"Invalid task range {task_start}-{task_end} for "
                f"{len(candidates)} candidates"
            )
        maximum = min(task_end - task_start + 1, self.config.max_concurrent)
        walltime = (
            self.config.time_by_rung[rung]
            if self.config.time_by_rung is not None
            else self.config.time
        )
        label = label or ("anchors" if anchor else f"r{rung}")
        logs = self.search.search_dir / "logs"
        logs.mkdir(exist_ok=True)
        working_directory = self.config.scratch_dir or self.config.output_dir
        return self._submit(
            [
                "sbatch",
                "--parsable",
                f"--job-name={self.search.run_id}-{label}",
                f"--chdir={working_directory}",
                f"--array={task_start}-{task_end}%{maximum}",
                f"--account={self.config.account}",
                f"--partition={self.config.partition}",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={self.config.cpus_per_task}",
                f"--mem={self.config.memory}",
                f"--gres={self.config.gres}",
                f"--time={walltime}",
                f"--output={logs}/{label}-%A_%a.out",
                f"--error={logs}/{label}-%A_%a.err",
                f"--export={self._exports(rung, anchor=anchor, worker_command=worker_command)}",
                str(self.config.harness),
            ]
        )

    def submit_anchors(self, state: dict[str, Any]) -> None:
        if not state["anchors"]["candidates"]:
            return
        state["anchors"]["job_id"] = self._array(
            state, len(self.search.rungs) - 1, anchor=True
        )
        self.search.write_state(state)

    def submit_rung(self, state: dict[str, Any], rung: int) -> None:
        if rung == 0 and self.config.rung0_probe:
            self._submit_probe(state, rung)
            return
        self.submit_validated_rung(state, rung)

    def _submit_probe(self, state: dict[str, Any], rung: int) -> None:
        """Gate the expensive first array on one real optimizer update."""
        candidate = state["rungs"][rung]["candidates"][0]
        job_id = self._array(
            state,
            rung,
            anchor=False,
            task_start=0,
            task_end=0,
            label=f"r{rung}-probe",
            worker_command="probe",
        )
        command = self._controller_command(
            "-m",
            "samudra.search.worker",
            "release-probe",
            str(self.search.config_path),
            str(self.search.state_path),
            str(rung),
        )
        logs = self.search.search_dir / "logs"
        working_directory = self.config.scratch_dir or self.config.output_dir
        controller_job = self._submit(
            [
                "sbatch",
                "--parsable",
                f"--job-name={self.search.run_id}-release-r{rung}",
                f"--chdir={working_directory}",
                f"--dependency=afterany:{job_id}",
                f"--account={self.config.account}",
                f"--partition={self.config.controller_partition}",
                "--cpus-per-task=1",
                "--mem=2G",
                "--time=00:10:00",
                "--export=ALL",
                f"--output={logs}/release-r{rung}-%j.out",
                f"--error={logs}/release-r{rung}-%j.err",
                f"--wrap={command}",
            ]
        )
        state = self.search.read_state()
        state["rungs"][rung]["probe"] = {
            "candidate": candidate,
            "job_id": job_id,
            "controller_job_id": controller_job,
            "status": "submitted",
        }
        state["status"] = "validating"
        self.search.write_state(state)

    def submit_validated_rung(self, state: dict[str, Any], rung: int) -> None:
        logs = self.search.search_dir / "logs"
        logs.mkdir(exist_ok=True)
        job_id = self._array(state, rung, anchor=False)
        dependency = f"afterany:{job_id}"
        anchor_job = state["anchors"].get("job_id")
        if rung == len(self.search.rungs) - 1 and anchor_job:
            dependency += f":{anchor_job}"
        command = self._controller_command(
            "-m",
            "samudra.search.worker",
            "advance",
            str(self.search.config_path),
            str(self.search.state_path),
            str(rung),
        )
        working_directory = self.config.scratch_dir or self.config.output_dir
        controller_job = self._submit(
            [
                "sbatch",
                "--parsable",
                f"--job-name={self.search.run_id}-advance-r{rung}",
                f"--chdir={working_directory}",
                f"--dependency={dependency}",
                f"--account={self.config.account}",
                f"--partition={self.config.controller_partition}",
                "--cpus-per-task=1",
                "--mem=2G",
                "--time=00:10:00",
                "--export=ALL",
                f"--output={logs}/advance-r{rung}-%j.out",
                f"--error={logs}/advance-r{rung}-%j.err",
                f"--wrap={command}",
            ]
        )
        state = self.search.read_state()
        state["rungs"][rung]["job_id"] = job_id
        state["rungs"][rung]["controller_job_id"] = controller_job
        state["status"] = "running"
        self.search.write_state(state)
