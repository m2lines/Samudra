# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-quarter baseline runner, with separate reference/qualification/train modes.

Launch qualification and training with torchrun --nproc_per_node=4. Reference
construction needs one GPU. No mode submits work. Production requires measured
four-rank qualification from this exact producer/runtime/data/science contract.
"""

import argparse
import datetime
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from samudra.experiments.missingness import model_options
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_pilot import (
    Pilot,
    atomic_json,
    atomic_torch,
    build_parser,
    digest,
)
from samudra.experiments.quarter_distributed import (
    accumulate_task_gradients,
    distributed_task_loss,
    gather_rank_rng,
    rank_samples,
    restore_rank_rng,
)
from samudra.experiments.task_schedule import TaskSchedule

ARMS = {"u-global": (3000, 3000, "mixed"), "observation-only": (0, 6000, "scratch")}
RUNTIME_MODULES = frozenset(
    {
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "zarr",
        "numcodecs",
        "cftime",
        "einops",
        "pydantic",
        "pydantic_core",
        "wandb",
    }
)


def verify_runtime(path, *, require_om4=False):
    """Bind imported package origins and loaded extensions to the pinned receipt."""
    if digest(path) != os.environ.get("SAMUDRA_RUNTIME_SHA256"):
        raise ValueError("Runtime contract hash differs")
    record = json.loads(Path(path).read_text())
    modules = record["modules"]
    required = RUNTIME_MODULES | ({"samudra_rust_loader"} if require_om4 else set())
    if not required.issubset(modules):
        raise ValueError(
            f"Runtime missing required modules: {sorted(required - modules.keys())}"
        )
    imported = {name: importlib.import_module(name) for name in modules}
    for name, module in imported.items():
        expected = modules[name]
        if module.__file__ is None:
            raise ValueError(f"Runtime module has no file origin: {name}")
        origin = Path(module.__file__).resolve()
        if getattr(module, "__version__", None) != expected[
            "version"
        ] or origin != Path(expected["origin"]):
            raise ValueError(f"Loaded runtime version/origin differs: {name}")
        spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if spec_origin and Path(spec_origin).resolve() != origin:
            raise ValueError(f"Loaded module spec/file disagree: {name}")
        binaries = expected["binaries"]
        if str(origin) not in binaries:
            raise ValueError(f"Runtime must hash its imported origin: {name}")
        for filename, checksum in binaries.items():
            artifact = Path(filename)
            if (
                not artifact.is_absolute()
                or artifact.resolve() != artifact
                or not artifact.is_relative_to(origin.parent)
            ):
                raise ValueError(f"Runtime artifact outside loaded package: {filename}")
            if digest(artifact) != checksum:
                raise ValueError(f"Runtime artifact hash differs: {filename}")
        for loaded_name, loaded in list(sys.modules.items()):
            if loaded_name != name and not loaded_name.startswith(name + "."):
                continue
            filename = getattr(loaded, "__file__", None)
            if filename and any(
                filename.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            ):
                if str(Path(filename).resolve()) not in binaries:
                    raise ValueError(
                        f"Loaded extension missing from runtime receipt: {loaded_name}"
                    )
    return record


def verify_compact_audit(path, root):
    import samudra_rust_loader as native  # type: ignore[import-not-found]

    # Maturin may expose a Python package wrapper around the actual extension.
    # Hash its loaded compiled module, never just that wrapper's __init__.py.
    binaries: set[Path] = set()
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            name == native.__name__ or name.startswith(native.__name__ + ".")
        ) and filename is not None:
            if any(
                filename.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            ):
                binaries.add(Path(filename).resolve())
    binary_hashes = {path.name: digest(path) for path in binaries}
    if not binaries or len(binary_hashes) != len(binaries):
        raise ValueError("Expected unambiguous loaded compact decoder binaries")
    record = json.loads(Path(path).read_text())
    if (
        record.get("status") != "passed"
        or record["source_metadata_sha256"]
        != digest(Path(root) / "OM4.zarr/.zmetadata")
        or record.get("module_binary_sha256") != binary_hashes
        or record.get("module_file_sha256") != digest(native.__file__)
        or not record.get("cases")
        or not all(case.get("equal_including_nan") is True for case in record["cases"])
    ):
        raise ValueError(
            "Compact native/xarray decoder audit does not match this source/runtime"
        )
    return digest(path)


class StopController:
    """Signal handlers only record intent; ranks stop at completed-update boundaries."""

    def __init__(self, hours, rank, device, *, started=None):
        self.rank, self.device = rank, device
        self.deadline = (
            time.monotonic() if started is None else started
        ) + hours * 3600
        self.reason: str | None = None
        self.handlers = {
            number: signal.getsignal(number)
            for number in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)
        }
        for number in self.handlers:
            signal.signal(number, self.handle_signal)

    def handle_signal(self, number, frame):
        self.reason = signal.Signals(number).name

    def check(self):
        requested = torch.tensor(int(self.reason is not None), device=self.device)
        dist.all_reduce(requested, op=dist.ReduceOp.MAX)
        decision: list[Any] = [None]
        if self.rank == 0:
            if requested.item():
                decision[0] = self.reason or "signal_on_worker"
            elif time.monotonic() >= self.deadline:
                decision[0] = "train_hours"
        dist.broadcast_object_list(decision, src=0)
        return decision[0]

    def close(self):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)


def configure_determinism():
    """One strict policy for reference, qualification and both production arms."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Set deterministic CUDA policy before CUDA initialization")
    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured != ":4096:8":
        raise ValueError("Quarter runner requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def pilot_arguments(args):
    """Freeze matched science; qualification flags never leak into production."""
    parsed = build_parser().parse_args(
        [
            "--data",
            args.data,
            "--iap-grid",
            str(Path(args.data) / "iap-grid.npz"),
            "--checkpoint",
            "unused-fresh",
            "--source-contract",
            "unused-fresh",
            "--output",
            args.output,
            "--name",
            args.name,
            "--from-scratch",
            "--normalization",
            "instance",
            "--initializer-architecture",
            "unet",
            "--evolution-architecture",
            "extent-unet",
            "--surface-policy",
            "observed-only",
            "--surface-fill",
            "zero",
            "--task-conditioning",
            "input-adapters",
            "--completion-weight",
            "0.1",
            "--reconstruction-weight",
            "0.1",
            "--observation-normalization",
            "--global-observations",
            "--strict-velocity-support",
            "--selection-metric",
            "rmse-only",
            "--core-lr",
            "0.0001",
            "--accumulate",
            "8",
            "--seed",
            "1729",
            "--wandb-mode",
            "online" if args.mode == "train" else "disabled",
        ]
    )
    parsed.fit_probe = args.mode != "train"
    parsed.qualification = args.qualification
    parsed.selection_reference = args.selection_reference
    return parsed


def science_contract(args, pilot_args):
    """Small immutable contract, also checked before allocating models/readers."""
    root = Path(args.data)
    code = os.environ.get("SAMUDRA_CODE_COMMIT", "")
    runtime = os.environ.get("SAMUDRA_RUNTIME_SHA256", "")
    if (
        len(code) != 40
        or len(runtime) != 64
        or digest(args.runtime_contract) != runtime
    ):
        raise ValueError(
            "Require explicit producer commit and matching runtime contract hash"
        )
    files = [
        "DATA_READY.json",
        "SHA256SUMS",
        "grid.npz",
        "iap-grid.npz",
        "iap-operator.json",
        "statistics.npz",
    ]
    hashes = {name: digest(root / name) for name in files}
    operator = json.loads((root / "iap-operator.json").read_text())
    if (
        operator["model_grid_sha256"] != hashes["grid.npz"]
        or operator["iap_grid_sha256"] != hashes["iap-grid.npz"]
    ):
        raise ValueError("Native observation operator/grid hash mismatch")
    om4 = None
    if args.arm == "u-global":
        if not args.om4_data or not args.compact_audit:
            raise ValueError("U-global requires compact OM4 source and decoder audit")
        source = Path(args.om4_data)
        om4 = {
            "path": str(source.resolve()),
            "hashes": {
                name: digest(source / name)
                for name in [
                    "COMPLETE.json",
                    "OM4.zarr/.zmetadata",
                    "OM4_means.zarr/.zmetadata",
                    "OM4_stds.zarr/.zmetadata",
                ]
            },
            "train_time": ["1975-01-03", "2013-10-04"],
            "retention_context_time": ["2013-08-01", "2014-10-05"],
            "retention_origins": 12,
            "decoder_audit_sha256": verify_compact_audit(args.compact_audit, source),
        }
    elif args.om4_data or args.compact_audit:
        raise ValueError("Observation-only must not configure an OM4 source/audit")
    return {
        "version": 1,
        "arm": args.arm,
        "schedule": list(ARMS[args.arm]),
        "code_commit": code,
        "runner_sha256": digest(__file__),
        "pilot_sha256": digest(Path(__file__).with_name("observation_pilot.py")),
        "distributed_helper_sha256": digest(
            Path(__file__).with_name("quarter_distributed.py")
        ),
        "runtime_extension_sha256": runtime,
        "data": hashes,
        "om4": om4,
        "selection_reference_sha256": digest(args.selection_reference),
        "normalization": pilot_args.normalization,
        "model_options": model_options(vars(pilot_args)),
        "evolution_architecture": pilot_args.evolution_architecture,
        "surface_fill": pilot_args.surface_fill,
        "completion_weight": 0.1,
        "reconstruction_weight": 0.1,
        "global_observations": True,
        "seed": 1729,
        "learning_rate": 1e-4,
        "global_batch": 8,
        "world_size": 4,
        "validate_every": args.validate_every,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_type": torch.cuda.get_device_name(),
        "deterministic_policy": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
        },
    }


def verify_data(root):
    """Rank zero performs full local readback, not just a readiness-label check."""
    root = Path(root)
    import pandas as pd

    from samudra.experiments.observation_data import SPLITS

    expected = {"grid.npz", "iap-grid.npz", "statistics.npz"}
    for split, bounds in SPLITS.items():
        expected.update(
            f"{split}/{month}.npz" for month in pd.period_range(*bounds, freq="M")
        )
    found = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() not in expected
            or path.as_posix() in found
        ):
            raise ValueError("Unexpected or duplicate data checksum path")
        if digest(root / path) != checksum:
            raise ValueError(f"Data checksum differs: {name}")
        found.add(path.as_posix())
    if found != expected:
        raise ValueError("Native data checksum manifest is incomplete")
    with np.load(root / "grid.npz") as grid:
        if grid["mask"].shape != (77, 720, 1440):
            raise ValueError("Expected native 77-channel 720x1440 grid")
    with np.load(root / "iap-grid.npz") as grid:
        if (len(grid["lat"]), len(grid["lon"])) != (360, 720):
            raise ValueError("Expected native IAP observation grid")


def check_qualification(record, contract):
    if record.get("quarter_contract") != contract:
        raise ValueError("Quarter qualification contract differs")
    evidence = record.get("quarter_evidence", {})
    for name in ["fit", "ddp", "resume", "io", "throughput"]:
        if evidence.get(name, {}).get("passed") is not True:
            raise ValueError(f"Missing measured quarter qualification: {name}")
    if evidence["ddp"].get("world_size") != 4:
        raise ValueError("Qualification requires four real ranks")
    measured_seconds = evidence["throughput"].get("seconds_per_update", 0)
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
        raise ValueError("Missing measured update throughput")
    tasks = ["observation"] + (["om4"] if contract["arm"] == "u-global" else [])
    hashes = evidence["ddp"].get("model_hashes", [])
    if len(hashes) != 4 or len(set(hashes)) != 1 or len(hashes[0]) != 64:
        raise ValueError("Missing matching measured DDP replica hashes")
    if set(evidence["resume"].get("tasks", [])) != set(tasks):
        raise ValueError("Resume replay must cover every trained task")
    for task in tasks:
        seconds = evidence["throughput"].get("task_seconds_per_update", {}).get(task, 0)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Missing finite positive {task} throughput")
        task_hashes = evidence["ddp"].get("task_model_hashes", {}).get(task, [])
        if len(task_hashes) != 4 or len(set(task_hashes)) != 1:
            raise ValueError(f"Missing matching {task} DDP replicas")
        losses = evidence["fit"].get("losses", {}).get(task, [])
        if (
            len(losses) != 11
            or not np.isfinite(losses).all()
            or losses[-1] >= losses[0]
        ):
            raise ValueError(f"Missing successful measured {task} fitting curve")
        components = ["initializer", "evolution", "completion_head"] + (
            ["adapter"] if task == "observation" else []
        )
        for component in components:
            if (
                evidence["fit"].get("gradients", {}).get(f"{task}/{component}")
                is not True
            ):
                raise ValueError(f"Missing measured {task}/{component} gradients")
    if evidence["io"].get("observation_samples", 0) < 1:
        raise ValueError("Missing measured observation reads")
    if "om4" in tasks and evidence["io"].get("om4_windows", 0) < 1:
        raise ValueError("Missing measured native OM4 window reads")


class Om4Adapter:
    """Bounded six-tensor reader adapted to the existing objective/retention API."""

    names: tuple[str, ...]
    lat: torch.Tensor
    mask: torch.Tensor

    def __init__(self, root, data, model):
        import xarray as xr

        from samudra.config import Om4TimeConfig
        from samudra.experiments.initializer_wave import Pair
        from samudra.experiments.missingness import TaskView
        from samudra.experiments.quarter_data import QuarterOm4Windows
        from samudra.experiments.surface_wave import data_config
        from samudra.utils.data import CanonicalSource, ChannelStatistics

        config = data_config(SimpleNamespace(readers=4)).sources[0]
        self.datasets = [
            xr.open_zarr(Path(root) / name, chunks=None)
            for name in ["OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr"]
        ]
        raw, means, stds, layout = config.canonicalize_datasets(*self.datasets)
        source = CanonicalSource.from_datasets(
            raw,
            means,
            stds,
            data_layout=layout,
            prognostic_var_names=layout.prognostic_var_names,
            boundary_var_names=layout.boundary_var_names,
        )
        statistics = ChannelStatistics(data.grid["mean"], data.grid["std"])

        def windows(bounds):
            return QuarterOm4Windows.from_compact(
                source.slice_time(Om4TimeConfig.model_validate(bounds)),
                str(Path(root) / "OM4.zarr"),
                device=data.device,
                observation_statistics=statistics,
            )

        self.trainset = windows({"start": "1975-01-03", "end": "2013-10-04"})
        self.valset = windows({"start": "2013-08-01", "end": "2014-10-05"})
        for name in ["names", "mask", "lat", "weights", "mean", "std", "device"]:
            setattr(self, name, getattr(self.trainset, name))
        if tuple(self.names) != tuple(data.grid["names"]):
            raise ValueError("OM4 and observation channel order differs")
        np.testing.assert_allclose(
            self.lat.cpu().numpy(), data.grid["lat"], rtol=0, atol=1e-5
        )
        np.testing.assert_allclose(
            self.trainset.lon.cpu().numpy(), data.grid["lon"], rtol=0, atol=1e-5
        )
        torch.testing.assert_close(self.mask, data.mask.bool(), rtol=0, atol=0)
        self.val_ids = list(range(0, len(self.valset), 6))[:12]
        if len(self.val_ids) != 12:
            raise ValueError("Incomplete 12-origin OM4 retention cohort")
        self.initializer = TaskView(model.initializer, "om4")
        self.model = Pair(
            self.initializer, TaskView(model.evolution, "om4"), joint=True
        )
        self.rank, self.world, self.args = 0, 1, SimpleNamespace(batch_size=1)
        self.reads, self.read_seconds = 0, 0.0

    def model_sample(self, dataset, ids):
        started = time.monotonic()
        result = dataset.model_sample(ids)
        self.reads += 1
        self.read_seconds += time.monotonic() - started
        return result

    def prepare(self, dataset):
        # Each model_sample is bounded; no resident period cache exists.
        pass

    def sync_buffers(self, model):
        # Retention runs on rank zero; fixed instance normalization has no running buffers.
        pass

    def reduce(self, totals):
        pass

    def retention(self):
        from samudra.experiments.initializer_wave import InitializerWave

        return InitializerWave.score(self, self.valset, self.val_ids, forecast=True)  # type: ignore[arg-type]


def tensor_hash(state):
    hasher = hashlib.sha256()
    for name, tensor in state.items():
        hasher.update(name.encode())
        hasher.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return hasher.hexdigest()


class QuarterRunner:
    def __init__(self, args, contract):
        self.args, self.contract = args, contract
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.pilot = Pilot(pilot_arguments(args), write_artifacts=self.rank == 0)
        if self.rank == 0 and args.mode == "train":
            if self.pilot.wandb is None or not self.pilot.wandb.url:
                raise ValueError("Production requires a real online W&B run")
        random.seed(1729)
        self.pilot.model.set_phase("joint")
        self.om4 = (
            Om4Adapter(args.om4_data, self.pilot.data, self.pilot.model)
            if args.arm == "u-global"
            else None
        )
        self.schedule = TaskSchedule(*ARMS[args.arm])
        self.completed = 0
        self.last_validated = -1
        self.best = math.inf
        self.optimizer = torch.optim.AdamW(
            self.pilot.model.parameters(), lr=1e-4, weight_decay=0.01
        )
        objectives = {"observation": self.observation_objective}
        if self.om4 is not None:
            objectives["om4"] = self.om4_objective
        self.ddp = distributed_task_loss(
            self.pilot.model, objectives, device_ids=[self.device.index]
        )
        self.out = Path(args.output)
        if self.rank == 0:
            reference = json.loads(Path(args.selection_reference).read_text())
            if (
                reference["protocol"] != self.pilot.protocol
                or reference["data_manifest_sha256"] != contract["data"]["SHA256SUMS"]
                or reference["control"]["origins"]
                != [p.stem for p in self.pilot.validation]
                or reference["spectral_keys"]
            ):
                raise ValueError("Fixed native RMSE reference differs")
            self.pilot.control, self.pilot.spectral_keys = reference["control"], []
        self.io_seconds = 0.0
        self.io_samples = 0

    def observation_objective(self, model, batch):
        if model is not self.pilot.model:
            raise ValueError("Objective must use the DDP-owned core")
        return self.pilot.objective(batch, "joint")

    def om4_objective(self, model, batch):
        from samudra.experiments.observation_joint import om4_objective

        sample, coverage = batch
        return om4_objective(
            model, self.om4, [sample.sample_index], 0.1, sample.mask_seed, coverage, 0.1
        )

    def load(self, task, micro):
        start = time.monotonic()
        if task == "observation":
            result = self.pilot.data.load(self.pilot.training[micro.sample_index])
            result["mask_seed"] = micro.mask_seed
        else:
            result = (
                micro,
                self.pilot.data.load(self.pilot.training[micro.coverage_index])[
                    "validity"
                ][:, :19],
            )
        self.io_seconds += time.monotonic() - start
        self.io_samples += 1
        return result

    def update(self, task=None, fixed=False):
        task = task or self.schedule.task(self.completed)
        if task == "om4" and self.om4 is None:
            raise ValueError("Observation-only cannot request OM4")
        # Rank-zero evaluation leaves the core in eval mode. Restore every rank
        # explicitly before any sample loading or task forward.
        self.ddp.train()
        count = 0 if fixed else self.schedule.counts(self.completed)[task]
        samples = rank_samples(
            task=task,
            dataset_size=len(self.om4.trainset)
            if task == "om4" and self.om4 is not None
            else len(self.pilot.training),
            observation_size=len(self.pilot.training),
            seed=1729,
            completed_task_updates=count,
            rank=self.rank,
            world_size=dist.get_world_size(),
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = accumulate_task_gradients(
            self.ddp,
            task,
            samples,
            lambda micro: self.load(task, micro),
            autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
        )
        reached = {}
        for name in ["initializer", "evolution", "adapter"]:
            grads = [
                p.grad
                for p in getattr(self.pilot.model, name).parameters()
                if p.grad is not None
            ]
            reached[name] = any(bool(gradient.count_nonzero()) for gradient in grads)
            if task == "om4" and name == "adapter" and grads:
                raise ValueError("Inactive ERA5 adapter received OM4 gradients")
        required = ["initializer", "evolution"] + (
            ["adapter"] if task == "observation" else []
        )
        if not all(reached[name] for name in required):
            raise ValueError(f"Missing task gradients: {reached}")
        initializer = self.pilot.model.initializer
        head = initializer.net[-1]
        rows = initializer.surface + [
            index + initializer.channels for index in initializer.surface
        ]
        reached["completion_head"] = head.weight.grad is not None and bool(
            head.weight.grad[rows].count_nonzero()
        )
        for name in ["initializer", "evolution"]:
            adapters = getattr(getattr(self.pilot.model, name), "input_adapters", None)
            if adapters is not None:
                other = "observation" if task == "om4" else "om4"
                if any(
                    parameter.grad is not None
                    for parameter in adapters[other].parameters()
                ):
                    raise ValueError("Inactive task adapter received gradients")
                if not any(
                    parameter.grad is not None and bool(parameter.grad.count_nonzero())
                    for parameter in adapters[task].parameters()
                ):
                    raise ValueError("Active task adapter received no gradients")
        norm = torch.nn.utils.clip_grad_norm_(self.pilot.model.parameters(), 1.0)
        finite = torch.isfinite(norm).int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite gradient on at least one rank")
        self.optimizer.step()
        if not fixed:
            self.completed += 1
        return loss, reached

    def checkpoint(self, path):
        rng = gather_rank_rng(cuda_device=self.device)
        if self.rank == 0:
            atomic_torch(
                {
                    "model": self.pilot.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "completed": self.completed,
                    "last_validated": self.last_validated,
                    "task_counts": self.schedule.counts(self.completed),
                    "rng": rng,
                    "quarter_contract": self.contract,
                    "best": self.best,
                },
                path,
            )
        dist.barrier()

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["quarter_contract"] != self.contract or state[
            "task_counts"
        ] != self.schedule.counts(state["completed"]):
            raise ValueError("Resume contract or counters differ")
        self.pilot.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.completed, self.best = state["completed"], state["best"]
        self.last_validated = state.get("last_validated", -1)
        restore_rank_rng(state["rng"], cuda_device=self.device)

    def qualify(self):
        from samudra.experiments.observation_joint import assert_exact_state, cpu_copy

        initial = self.out / "qualification-initial.pt"
        self.checkpoint(initial)
        task_losses, reached_all = {}, {}
        task_hashes = {}
        started = time.monotonic()
        tasks = ["observation"] + (["om4"] if self.om4 is not None else [])
        for task in tasks:
            self.restore(initial)
            losses = []
            task_reached: dict[str, bool] = {}
            for _ in range(11):
                loss, reached = self.update(task, fixed=True)
                losses.append(loss)
                for name, value in reached.items():
                    task_reached[name] = task_reached.get(name, False) or value
            if losses[-1] >= losses[0]:
                raise ValueError(
                    f"Real-data {task} fixed-batch fitting failed: {losses}"
                )
            task_losses[task] = losses
            reached_all.update(
                {
                    f"{task}/{name}": value
                    for name, value in task_reached.items()
                    if name != "adapter" or task == "observation"
                }
            )
            hashes = [None] * dist.get_world_size()
            dist.all_gather_object(hashes, tensor_hash(self.pilot.model.state_dict()))
            if len(set(hashes)) != 1:
                raise ValueError(f"DDP replicas diverged after {task} fitting")
            task_hashes[task] = hashes
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        self.restore(initial)
        replay_tasks = (
            ["om4", "observation"] if self.om4 is not None else ["observation"]
        )
        scheduled_update_seconds = []
        task_update_seconds: dict[str, list[float]] = {task: [] for task in tasks}

        def scheduled_update():
            torch.cuda.synchronize(self.device)
            started = time.monotonic()
            task = self.schedule.task(self.completed)
            result = self.update()
            torch.cuda.synchronize(self.device)
            seconds = time.monotonic() - started
            scheduled_update_seconds.append(seconds)
            task_update_seconds[task].append(seconds)
            return result

        for replay_task in replay_tasks:
            # Reach each task through the real schedule, never invented counters.
            while self.schedule.task(self.completed) != replay_task:
                scheduled_update()
            before = self.out / f"qualification-resume-{replay_task}.pt"
            self.checkpoint(before)
            scheduled_update()
            expected = {
                "model": cpu_copy(self.pilot.model.state_dict()),
                "optimizer": cpu_copy(self.optimizer.state_dict()),
                "completed": self.completed,
                "counts": self.schedule.counts(self.completed),
                "rng": gather_rank_rng(cuda_device=self.device),
            }
            self.restore(before)
            self.update()
            actual = {
                "model": cpu_copy(self.pilot.model.state_dict()),
                "optimizer": cpu_copy(self.optimizer.state_dict()),
                "completed": self.completed,
                "counts": self.schedule.counts(self.completed),
                "rng": gather_rank_rng(cuda_device=self.device),
            }
            assert_exact_state(actual, expected)
            del actual, expected
        rank_measurements: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(
            rank_measurements,
            {
                "rank": self.rank,
                "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "seconds_per_update": float(np.mean(scheduled_update_seconds)),
                "task_update_seconds": task_update_seconds,
                "observation_samples": self.io_samples,
                "observation_load_seconds": self.io_seconds,
                "om4_windows": self.om4.reads if self.om4 is not None else 0,
                "om4_read_seconds": self.om4.read_seconds
                if self.om4 is not None
                else 0,
            },
        )
        if self.rank == 0:
            # Native validation must execute through the same production operator.
            torch.cuda.synchronize(self.device)
            validation_started = time.monotonic()
            metrics = self.pilot.evaluate(self.pilot.validation)
            selection_score(metrics, self.pilot.control, [], metric="rmse-only")
            torch.cuda.synchronize(self.device)
            observation_validation_seconds = time.monotonic() - validation_started
            rank_measurements[0]["max_gpu_memory_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            args = self.pilot.args
            evidence = {
                "fit": {
                    "passed": True,
                    "losses": task_losses,
                    "gradients": reached_all,
                },
                "ddp": {
                    "passed": True,
                    "world_size": 4,
                    "model_hashes": hashes,
                    "task_model_hashes": task_hashes,
                },
                "resume": {
                    "passed": True,
                    "scope": "exact next model/optimizer/counters/all-rank RNG",
                    "tasks": replay_tasks,
                },
                "io": {
                    "passed": self.io_samples > 0,
                    "observation_samples": self.io_samples,
                    "observation_load_seconds": self.io_seconds,
                    "om4_bounded_windows": self.om4 is not None,
                    "om4_windows": self.om4.reads if self.om4 is not None else 0,
                    "om4_read_seconds": self.om4.read_seconds
                    if self.om4 is not None
                    else 0,
                },
                "throughput": {
                    "observation_validation_seconds": observation_validation_seconds,
                    "passed": elapsed > 0,
                    "seconds_per_update": max(
                        item["seconds_per_update"] for item in rank_measurements
                    ),
                    "scheduled_update_seconds": scheduled_update_seconds,
                    "task_seconds_per_update": {
                        task: max(
                            float(np.mean(item["task_update_seconds"][task]))
                            for item in rank_measurements
                        )
                        for task in tasks
                    },
                    "fixed_batch_seconds_per_update": elapsed / (11 * len(tasks)),
                    "max_gpu_memory_bytes": max(
                        item["max_gpu_memory_bytes"] for item in rank_measurements
                    ),
                    "rank_measurements": rank_measurements,
                },
            }
            qualification = {
                "quarter_contract": self.contract,
                "quarter_evidence": evidence,
                "code_commit": self.contract["code_commit"],
                "runtime_extension_sha256": self.contract["runtime_extension_sha256"],
                "data_manifest_sha256": self.contract["data"]["SHA256SUMS"],
                "selection_reference_sha256": self.contract[
                    "selection_reference_sha256"
                ],
                "normalization": args.normalization,
                "evolution_architecture": args.evolution_architecture,
                "model_options": model_options(vars(args)),
                "surface_fill": args.surface_fill,
                "completion_weight": args.completion_weight,
                "global_observations": True,
                "gradient_reached": reached_all,
                "losses": task_losses["observation"],
            }
            check_qualification(qualification, self.contract)
            atomic_json(qualification, self.out / "QUALIFIED.json")
        dist.barrier()

    def validate(self):
        dist.barrier()
        if self.rank == 0:
            metrics = self.pilot.evaluate(self.pilot.validation)
            score = selection_score(metrics, self.pilot.control, [], metric="rmse-only")
            retention = self.om4.retention() if self.om4 is not None else None
            if score < self.best:
                self.best = score
                metadata = {
                    "score": score,
                    "phase": "joint",
                    "step": self.schedule.counts(self.completed)["observation"],
                    "global_step": self.completed,
                    "task_counts": self.schedule.counts(self.completed),
                    "quarter_contract": self.contract,
                }
                atomic_torch(
                    {"model": self.pilot.model.state_dict(), **metadata},
                    self.out / "best.pt",
                )
                atomic_json(
                    {
                        **metadata,
                        "metrics": metrics,
                        "checkpoint_sha256": digest(self.out / "best.pt"),
                    },
                    self.out / "best.json",
                )
            atomic_json(
                {
                    "global_step": self.completed,
                    "task_counts": self.schedule.counts(self.completed),
                    "metrics": metrics,
                    "score": score,
                    "om4_retention": retention,
                },
                self.out / f"validation-{self.completed:05d}.json",
            )
            self.pilot.emit(
                {
                    "event": "quarter_validation",
                    "global_step": self.completed,
                    "validation/obs_score": score,
                }
            )
        shared = [self.best]
        dist.broadcast_object_list(shared, src=0)
        self.best = shared[0]
        self.last_validated = self.completed
        dist.barrier()

    def stop_partial(self, reason):
        last = self.out / "quarter-last.pt"
        self.checkpoint(last)
        if self.rank == 0:
            atomic_json(
                {
                    "reason": reason,
                    "updates": self.completed,
                    "last_validated": self.last_validated,
                    "task_counts": self.schedule.counts(self.completed),
                    "quarter_contract": self.contract,
                    "checkpoint_sha256": digest(last),
                },
                self.out / "PARTIAL.json",
            )
            self.pilot.emit(
                {
                    "event": "quarter_partial",
                    "reason": reason,
                    "global_step": self.completed,
                }
            )

    def train(self, stop):
        last = self.out / "quarter-last.pt"
        if last.exists():
            self.restore(last)
        # Resume a boundary checkpoint saved before its scheduled validation.
        if self.last_validated != self.completed and (
            self.completed % self.args.validate_every == 0
            or self.completed == self.schedule.total
        ):
            reason = stop.check()
            if reason:
                self.stop_partial(reason)
                return
            self.validate()
            self.checkpoint(last)
        while self.completed < self.schedule.total:
            reason = stop.check()
            if reason:
                self.stop_partial(reason)
                return
            started = time.monotonic()
            task = self.schedule.task(self.completed)
            loss, _ = self.update()
            if self.rank == 0:
                self.pilot.emit(
                    {
                        "event": "quarter_train",
                        "global_step": self.completed,
                        "task": task,
                        "loss": loss,
                        "seconds": time.monotonic() - started,
                        **self.schedule.counts(self.completed),
                    }
                )
            reason = stop.check()
            if reason:
                self.stop_partial(reason)
                return
            if (
                self.completed % self.args.validate_every == 0
                or self.completed == self.schedule.total
            ):
                # Preserve completed updates if a hard limit interrupts evaluation.
                self.checkpoint(last)
                self.validate()
                self.checkpoint(last)
        # A wall-time stop may have saved the final update before validation.
        if self.last_validated != self.completed:
            reason = stop.check()
            if reason:
                self.stop_partial(reason)
                return
            self.validate()
            self.checkpoint(last)
        if self.rank == 0:
            best = json.loads((self.out / "best.json").read_text())
            best_hash = digest(self.out / "best.pt")
            if best["checkpoint_sha256"] != best_hash:
                raise ValueError("Selected checkpoint hash differs at completion")
            atomic_json(
                {
                    "updates": self.completed,
                    "global_step": self.completed,
                    "task_counts": self.schedule.counts(self.completed),
                    "quarter_contract": self.contract,
                    "checkpoint_sha256": digest(last),
                    "best": best,
                    "best_score": best["score"],
                    "best_checkpoint_sha256": best_hash,
                    "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                },
                self.out / "TRAIN_COMPLETE.json",
            )
            (self.out / "PARTIAL.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "qualify", "train"])
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    for name in ["data", "output", "name", "runtime-contract", "selection-reference"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--om4-data")
    parser.add_argument("--compact-audit")
    parser.add_argument("--qualification")
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--train-hours", type=float, default=22.0)
    args = parser.parse_args()
    invocation_started = time.monotonic()
    if args.validate_every < 1:
        parser.error("validate-every must be positive")
    if not math.isfinite(args.train_hours) or args.train_hours <= 0:
        parser.error("train-hours must be finite and positive")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != (1 if args.mode == "reference" else 4):
        parser.error("Reference needs one GPU; qualification/train require four ranks")
    if args.mode == "train" and not args.qualification:
        parser.error("Production requires measured native-quarter qualification")
    configure_determinism()
    verify_runtime(args.runtime_contract, require_om4=args.arm == "u-global")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    runner = None
    stop = (
        StopController(
            args.train_hours,
            dist.get_rank(),
            torch.device("cuda"),
            started=invocation_started,
        )
        if args.mode == "train"
        else None
    )
    try:
        if dist.get_rank() == 0:
            verify_data(args.data)
        dist.barrier()
        if args.mode == "reference":
            pilot = Pilot(pilot_arguments(args))
            target = Path(args.selection_reference)
            if target.exists():
                raise ValueError("Refusing to overwrite frozen selection reference")
            control = pilot.evaluate(pilot.validation, climatology=True)
            atomic_json(
                {
                    "control": control,
                    "spectral_keys": [],
                    "protocol": pilot.protocol,
                    "data_manifest_sha256": pilot.manifest["data_manifest_sha256"],
                },
                target,
            )
            return
        contract = science_contract(args, pilot_arguments(args))
        if args.mode == "train":
            check_qualification(
                json.loads(Path(args.qualification).read_text()), contract
            )
        runner = QuarterRunner(args, contract)
        if args.mode == "qualify":
            runner.qualify()
        else:
            runner.train(stop)
    finally:
        if stop is not None:
            stop.close()
        if runner is not None and runner.pilot.wandb is not None:
            runner.pilot.wandb.finish()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
