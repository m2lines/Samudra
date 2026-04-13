from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import torch

from ocean_emulators import samudrax, tide_jax
from ocean_emulators.config import TrainConfig
from ocean_emulators.constants import LoaderVersion
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope


@dataclass(frozen=True)
class Case:
    name: str
    loader_version: str
    workers: int
    use_jax_frontend: bool = False
    jax_blob_placement: tide_jax.JaxBlobPlacementPolicy = "cpu"
    output_placement: tide_jax.JaxBlobPlacementPolicy = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="~/data/om4_onedeg")
    parser.add_argument("--config", default="configs/test/train_default_2step.yaml")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-start", default="1975-01-01")
    parser.add_argument("--train-end", default="1977-01-01")
    parser.add_argument("--val-start", default="1977-01-01")
    parser.add_argument("--val-end", default="1977-03-01")
    parser.add_argument(
        "--jax-crop",
        default=None,
        help="Optional plain-JAX crop as lat_start,lat_end,lon_start,lon_end.",
    )
    return parser.parse_args()


def validate_data_root(data_root: Path) -> None:
    missing = [
        name
        for name in ["OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr"]
        if not (data_root / name).exists()
    ]
    if missing:
        raise SystemExit(f"{data_root} is missing required OM4 files: {missing}")


def parse_jax_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("--jax-crop must have four comma-separated integers")
    return tuple(parts)


def apply_spatial_crop(tensor: Any, jax_crop: tuple[int, int, int, int] | None) -> Any:
    if jax_crop is None:
        return tensor
    lat_start, lat_end, lon_start, lon_end = jax_crop
    return tensor[..., lat_start:lat_end, lon_start:lon_end]


def crop_data_source(
    source: samudrax.DataSource, jax_crop: tuple[int, int, int, int] | None
) -> samudrax.DataSource:
    if jax_crop is None:
        return source
    lat_start, lat_end, lon_start, lon_end = jax_crop
    return source.crop_spatial(lat_start, lat_end, lon_start, lon_end)


def shape_after_spatial_crop(
    shape: tuple[int, ...], jax_crop: tuple[int, int, int, int] | None
) -> tuple[int, ...]:
    if jax_crop is None:
        return shape
    lat_start, lat_end, lon_start, lon_end = jax_crop
    return (*shape[:-2], lat_end - lat_start, lon_end - lon_start)


def build_config(args: argparse.Namespace, case: Case) -> TrainConfig:
    sources = json.dumps(
        [
            {
                "data_location": "OM4.zarr",
                "data_means_location": "OM4_means.zarr",
                "data_stds_location": "OM4_stds.zarr",
            }
        ]
    )
    cfg = TrainConfig.from_yaml_and_cli(
        [
            args.config,
            "--backend",
            args.backend,
            "--experiment.data_root",
            str(Path(args.data_root).expanduser()),
            "--experiment.base_output_dir",
            "/tmp/ocean_emulator_bench",
            "--experiment.name",
            f"bench_tide_jax_{case.name.replace(' ', '_').replace('=', '')}",
            "--data.sources",
            sources,
            "--data.loader_version",
            case.loader_version,
            "--data.num_workers",
            str(case.workers),
            "--batch_size",
            str(args.batch_size),
            "--train_time.start",
            args.train_start,
            "--train_time.end",
            args.train_end,
            "--val_time.start",
            args.val_start,
            "--val_time.end",
            args.val_end,
        ]
    )
    return cfg


def sync_torch(backend: str) -> None:
    if backend == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def block_jax(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def consume_torch_batch(batch: Any) -> None:
    _ = batch.get_input(0)
    _ = batch.get_label(0)
    for step in range(1, len(batch)):
        prev_prediction = torch.zeros_like(batch.get_label(step - 1))
        _ = batch.merge_prognostic_and_boundary(prev_prediction, step)
        _ = batch.get_label(step)


def single_samudra_data_source(loader: Any) -> samudrax.DataSource:
    datasets = loader.datasets
    if len(datasets) != 1:
        raise NotImplementedError(
            "Tide JAX benchmark v0 expects one data source; pass the matching "
            "train dataset explicitly for multi-source schedules."
        )
    return samudrax.DataSource.from_train_dataset(datasets[0])


def build_tide_jax_programs(
    batch: Any,
    *,
    source: samudrax.DataSource,
    jax_crop: tuple[int, int, int, int] | None,
) -> tuple[float, Any, dict[int, Any]]:
    start = time.perf_counter()
    spec = tide_jax.shape_spec_from_batch(batch)
    source = crop_data_source(source, jax_crop)

    def step0():
        prognostic = samudrax.normalize_and_mask(
            apply_spatial_crop(tide_jax.raw_step0_prognostic(spec), jax_crop),
            source,
            "prognostic",
        )
        boundary = samudrax.normalize_and_mask(
            apply_spatial_crop(tide_jax.raw_step0_boundary(spec), jax_crop),
            source,
            "boundary",
        )
        label = samudrax.normalize_and_mask(
            apply_spatial_crop(tide_jax.raw_step0_label(spec), jax_crop),
            source,
            "prognostic",
        )
        return jnp.concatenate((prognostic, boundary), axis=1), label

    step0_program = tide_jax.trace_tide_jax(step0)

    later_programs = {}
    prev = jax.ShapeDtypeStruct(
        shape_after_spatial_crop(spec.step0_label_shape, jax_crop),
        jnp.float32,
    )
    for step in range(1, len(batch)):

        def later_step(prev_prediction, *, step=step):
            boundary = samudrax.normalize_and_mask(
                apply_spatial_crop(tide_jax.raw_boundary(spec, step), jax_crop),
                source,
                "boundary",
            )
            label = samudrax.normalize_and_mask(
                apply_spatial_crop(tide_jax.raw_label(spec, step), jax_crop),
                source,
                "prognostic",
            )
            return (
                jnp.concatenate((prev_prediction, boundary), axis=1),
                label,
            )

        later_programs[step] = tide_jax.trace_tide_jax(later_step, prev)

    return time.perf_counter() - start, step0_program, later_programs


def consume_tide_jax_batch(
    batch: Any,
    step0_program: Any,
    later_programs: dict[int, Any],
    *,
    jax_blob_placement: tide_jax.JaxBlobPlacementPolicy,
    output_placement: tide_jax.JaxBlobPlacementPolicy,
) -> None:
    materializer = tide_jax.RustTrainBatchMaterializer(
        batch,
        jax_blob_placement=jax_blob_placement,
        output_placement=output_placement,
    )
    outputs = step0_program.eval(materializer)
    block_jax(outputs)
    _, label = outputs

    for step in range(1, len(batch)):
        prev_prediction = jnp.zeros_like(label)
        outputs = later_programs[step].eval(materializer, prev_prediction)
        block_jax(outputs)
        _, label = outputs


def run_case(args: argparse.Namespace, case: Case) -> dict[str, float]:
    with MultitonScope():
        cfg = build_config(args, case)
        init_start = time.perf_counter()
        trainer = Trainer(cfg)
        trainer.init_data_loaders(cur_step=cfg.steps[0])
        init_s = time.perf_counter() - init_start

        loader_iter = iter(trainer.train_loader)
        setup_start = time.perf_counter()
        step0_program: Any = None
        later_programs: dict[int, Any] = {}
        if case.use_jax_frontend:
            warmup_batch = next(loader_iter)
            _, step0_program, later_programs = build_tide_jax_programs(
                warmup_batch,
                source=single_samudra_data_source(trainer.train_loader),
                jax_crop=args.jax_crop,
            )
            consume_tide_jax_batch(
                warmup_batch,
                step0_program,
                later_programs,
                jax_blob_placement=case.jax_blob_placement,
                output_placement=case.output_placement,
            )
        else:
            consume_torch_batch(next(loader_iter))
        sync_torch(args.backend)
        setup_s = time.perf_counter() - setup_start

        elapsed = []
        for _ in range(min(args.batches, max(len(trainer.train_loader) - 1, 0))):
            sync_torch(args.backend)
            start = time.perf_counter()
            batch = next(loader_iter)
            if case.use_jax_frontend:
                consume_tide_jax_batch(
                    batch,
                    step0_program,
                    later_programs,
                    jax_blob_placement=case.jax_blob_placement,
                    output_placement=case.output_placement,
                )
            else:
                consume_torch_batch(batch)
            sync_torch(args.backend)
            elapsed.append(time.perf_counter() - start)

        del loader_iter
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not elapsed:
        raise SystemExit(f"No batches were available for {case.name}")

    return {
        "init_s": init_s,
        "setup_s": setup_s,
        "first_batch_s": elapsed[0],
        "mean_batch_s": statistics.fmean(elapsed),
        "mean_excl_first_s": statistics.fmean(elapsed[1:])
        if len(elapsed) > 1
        else elapsed[0],
        "median_batch_s": statistics.median(elapsed),
        "min_s": min(elapsed),
        "max_s": max(elapsed),
        "batch_per_s": len(elapsed) / sum(elapsed),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    validate_data_root(data_root)
    args.data_root = str(data_root)
    args.jax_crop = parse_jax_crop(args.jax_crop)

    if args.backend == "cuda":
        print(
            "Tide JAX cases compare opaque JAX blob placement. The cpu-blob case "
            "loads raw Tide leaves on CPU, runs the JAX blob on CPU, then moves "
            "outputs to the selected JAX device. The device-blob case moves raw "
            "leaves to the torch/JAX device before running the blob."
        )

    cases = [
        Case("torch workers=0", LoaderVersion.OM4_TORCH.value, 0),
        Case(
            "tide jax cpu-blob->device",
            LoaderVersion.OM4_RUST_V0.value,
            0,
            True,
            "cpu",
            "device",
        ),
        Case(
            "tide jax device-blob",
            LoaderVersion.OM4_RUST_V0.value,
            0,
            True,
            "device",
            "device",
        ),
    ]

    print(
        "case, init_s, setup_s, first_batch_s, mean_batch_s, "
        "mean_excl_first_s, median_batch_s, min_s, max_s, batch_per_s"
    )
    for case in cases:
        result = run_case(args, case)
        print(
            f"{case.name}, "
            f"{result['init_s']:.3f}, "
            f"{result['setup_s']:.4f}, "
            f"{result['first_batch_s']:.4f}, "
            f"{result['mean_batch_s']:.4f}, "
            f"{result['mean_excl_first_s']:.4f}, "
            f"{result['median_batch_s']:.4f}, "
            f"{result['min_s']:.4f}, "
            f"{result['max_s']:.4f}, "
            f"{result['batch_per_s']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
