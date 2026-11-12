# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Resumable native wet-cell inference, followed by paired all-channel scoring."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from numcodecs import blosc  # type: ignore[import-untyped]

from samudra.config import SamudraConfig
from samudra.skyline.baseline import file_sha256
from samudra.skyline.data import CHANNELS, DEPTHS
from samudra.skyline.multitask import ComparisonSource
from samudra.skyline.run import to_device
from samudra.skyline.soft_prior_maps import load_selection
from samudra.utils.ctx import BatchGrid
from samudra.utils.device import set_device


def atomic_npz(path, **arrays):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def samudra(source, frame, config, output, device):
    path = output / "samudra.npz"
    if path.exists():
        return np.load(path)["prediction"], json.loads(
            (output / "samudra.json").read_text()
        )
    before = time.perf_counter()
    baseline = yaml.safe_load(Path(config["samudra_config"]).read_text())
    assert baseline["data"]["input_steps"] == 2
    assert baseline["data"]["output_steps"] == 1
    assert baseline["data"]["sources"][0]["prognostic_vars_key"] == "thermo_dynamic_all"
    assert baseline["data"]["sources"][0]["boundary_vars_key"] == "tau_hfds"
    assert not baseline["model"]["pred_residuals"]
    digest = file_sha256(config["samudra_checkpoint"])
    assert digest == config["samudra_sha256"]
    model = SamudraConfig(**baseline["model"]).build(
        prog_channels=154,
        boundary_channels=6,
        out_channels=77,
        input_steps=2,
        grid_sizes=[(source.height, source.width)],
    )
    state = torch.load(
        config["samudra_checkpoint"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        {k.removeprefix("module."): v for k, v in state["model"].items()}, strict=True
    )
    epoch = state["epoch"]
    del state
    model = model.to(device).eval()
    coordinates = (torch.from_numpy(source.lat), torch.from_numpy(source.lon))
    context = BatchGrid(
        torch.from_numpy(source.channel_wet[:77]).to(device), coordinates, coordinates
    )
    prog = torch.from_numpy(
        frame["input"][:, :77].reshape(1, 154, source.height, source.width)
    ).to(device)
    forcing = torch.from_numpy(
        frame["input"][:, 77:].reshape(1, 6, source.height, source.width)
    ).to(device)
    torch.cuda.synchronize()
    forward_start = time.perf_counter()
    prediction = model.forward_once(prog, forcing, context)[0].float().cpu().numpy()
    forward_seconds = time.perf_counter() - forward_start
    assert np.isfinite(prediction[source.channel_wet[:77]]).all()
    receipt = dict(
        checkpoint_sha256=digest,
        epoch=epoch,
        weights="model",
        precision="BF16 internal autocast, float32 returned array",
        setup_and_forward_seconds=time.perf_counter() - before,
        forward_and_cpu_transfer_seconds=forward_seconds,
    )
    cache = Path(config["samudra_temperature_cache"])
    if cache.exists():
        old = np.load(cache)
        diff = prediction[38:57][source.wet] - old[source.wet]
        receipt["cached_temperature_max_abs_normalized"] = float(np.abs(diff).max())
        receipt["cached_temperature_rmse_normalized"] = float(
            np.sqrt(np.mean(diff.astype(float) ** 2))
        )
        receipt["cached_temperature_exact_fraction"] = float(np.mean(diff == 0))
        receipt["cached_temperature_rmse_c"] = [
            float(
                np.sqrt(
                    np.mean(
                        (
                            (
                                prediction[38 + z][source.wet[z]].astype(float)
                                - old[z][source.wet[z]]
                            )
                            * source.std[38 + z]
                        )
                        ** 2
                    )
                )
            )
            for z in range(19)
        ]
    else:
        raise FileNotFoundError(cache)
    atomic_npz(path, prediction=prediction)
    atomic_json(output / "samudra.json", receipt)
    del model, prog, forcing
    torch.cuda.empty_cache()
    return prediction, receipt


def finalize(source, frame, config, output, tasks, receipt):
    shape = (77, source.height, source.width)
    prediction = np.full(shape, np.nan, np.float32)
    coverage = np.zeros((19, source.height * source.width), bool)
    for depth, chunk, indices in tasks:
        data = np.load(output / "chunks" / f"z{depth:02d}-{chunk:05d}.npz")
        assert np.array_equal(data["indices"], indices)
        assert not coverage[depth, indices].any()
        coverage[depth, indices] = True
        values = data["prediction"]
        for field in range(4):
            prediction[field * 19 + depth].ravel()[indices] = values[:, field]
        if depth == 0:
            prediction[76].ravel()[indices] = values[:, 4]
    assert np.array_equal(coverage.reshape(source.wet.shape), source.wet)
    baseline = np.load(output / "samudra.npz")["prediction"]
    wet = source.channel_wet[:77]
    arrays = {}
    metrics = []  # type: list[dict]
    for name, normalized in [
        ("classifier", prediction),
        ("samudra", baseline),
        ("persistence", frame["input"][1, :77]),
        ("gold", frame["target"]),
    ]:
        assert np.isfinite(normalized[wet]).all()
        physical = (
            normalized * source.std[:77, None, None] + source.mean[:77, None, None]
        )
        arrays[name] = np.where(wet, physical, np.nan).astype(np.float32)
        if name == "gold":
            continue
        for channel in range(77):
            mask = wet[channel]
            error = (
                normalized[channel][mask].astype(float) - frame["target"][channel][mask]
            ) * source.std[channel]
            weights = source.area[mask].astype(float)
            weights /= weights.sum()
            nerror = error / source.std[channel]
            metrics.append(
                dict(
                    model=name,
                    channel=channel,
                    name=CHANNELS[channel],
                    field=CHANNELS[channel].split("_")[0],
                    depth_index=channel % 19 if channel < 76 else 0,
                    depth_m=float(DEPTHS[channel % 19]) if channel < 76 else 0,
                    wet_count=int(mask.sum()),
                    area_rmse=float(np.sqrt(np.sum(weights * error**2))),
                    area_mae=float(np.sum(weights * np.abs(error))),
                    area_bias=float(np.sum(weights * error)),
                    rmse=float(np.sqrt(np.mean(error**2))),
                    normalized_full_grid_mse=float(np.sum(nerror**2) / mask.size),
                    normalized_wet_mse=float(np.mean(nerror**2)),
                )
            )
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    atomic_npz(
        output / "arrays.npz",
        **arrays,
        wet=wet,
        lat=source.lat,
        lon=source.lon,
        area=source.area,
        mean=source.mean[:77],
        std=source.std[:77],
        depths=DEPTHS,
        channels=np.array(CHANNELS[:77]),
    )
    summary = {
        name: {
            "full_grid_nmse": float(
                np.mean(
                    [
                        x["normalized_full_grid_mse"]
                        for x in metrics
                        if x["model"] == name
                    ]
                )
            )
        }
        for name in ("classifier", "samudra", "persistence")
    }
    receipt["aggregate"] = summary
    receipt["rank_timings"] = [
        json.loads(p.read_text()) for p in sorted(output.glob("rank-*.json"))
    ]
    receipt["samudra"] = json.loads((output / "samudra.json").read_text())
    receipt["exactly_once_wet_queries"] = int(coverage.sum())
    atomic_json(output / "provenance.json", receipt)
    atomic_json(
        output / "COMPLETE.json",
        dict(queries=int(coverage.sum()), metrics=len(metrics)),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    set_device(device)
    torch.set_num_threads(4)
    blosc.set_nthreads(1)
    dist.init_process_group("gloo")
    output = Path(config["output"]) / ("qualification" if args.qualify else "full")
    (output / "chunks").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = ComparisonSource(
        config["source"], 4, config.get("reference_root"), arm="all_fields"
    )
    frame = source.read(config["origin"])
    model, vocab, cfg, receipt = load_selection(config["classifier"], source, device)
    receipt.update(
        origin=config["origin"],
        target_date=str(source.date_strings[config["origin"] + 1]),
        world_size=world,
        gpu=torch.cuda.get_device_name(),
        source_commit=config["source_commit"],
        harness_sha256=file_sha256(__file__),
    )
    rng = np.random.default_rng(0)

    def predict(depth, indices):
        points = np.column_stack(
            np.unravel_index(indices, (source.height, source.width))
        )
        batch = to_device(
            source.sample(
                frame,
                rng,
                len(indices),
                cfg["crop"],
                np.full(len(indices), depth),
                points,
            ),
            device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model({k: v for k, v in batch.items() if k != "target"})  # noqa: F821
        return vocab.centers[batch["query_channel"], logits.argmax(-1)].cpu().numpy()  # noqa: F821

    if args.qualify:
        trial = predict(rank, np.flatnonzero(source.wet[rank])[:64])
        assert np.isfinite(trial).all()
        comparisons = []
        for path in sorted(
            Path(config["classifier_map_cache"]).glob("new-all-fields-*-z*.npz")
        ):
            old = np.load(path)
            depth = int(old["depth"])
            wet = old["wet"]
            rows = old["query_rows"][wet]
            cols = old["query_cols"][wet]
            if rank != depth % world:
                continue
            indices = np.ravel_multi_index(
                (rows[:64], cols[:64]), (source.height, source.width)
            )
            actual = (
                predict(depth, indices)[:, 2] * source.std[38 + depth]
                + source.mean[38 + depth]
            )
            expected = old["thetao_prediction"][wet][:64]
            comparisons.append(
                dict(
                    cache=path.name,
                    max_abs_c=float(np.max(np.abs(actual - expected))),
                    rmse_c=float(np.sqrt(np.mean((actual - expected) ** 2))),
                    exact_fraction=float(np.mean(actual == expected)),
                )
            )
            if not np.isfinite(actual).all():
                raise RuntimeError("Nonfinite qualification")
        atomic_json(output / f"comparison-{rank}.json", comparisons)
        if rank == 0:
            samudra(source, frame, config, output, device)
        dist.barrier()
        atomic_json(
            output / f"rank-{rank}.json", dict(seconds=time.perf_counter() - started)
        )
        if rank == 0:
            atomic_json(output / "COMPLETE.json", dict(world_size=world))
        source.close()
        dist.destroy_process_group()
        return
    tasks = []  # type: list[tuple[int, int, np.ndarray]]
    for depth in range(19):
        wet = np.flatnonzero(source.wet[depth])
        tasks.extend(
            (depth, start // 4096, wet[start : start + 4096])
            for start in range(0, len(wet), 4096)
        )
    done = 0
    compute_start = time.perf_counter()
    for task_index, (depth, chunk, indices) in enumerate(tasks):
        if task_index % world != rank:
            continue
        path = output / "chunks" / f"z{depth:02d}-{chunk:05d}.npz"
        if path.exists():
            continue
        parts = [
            predict(depth, indices[start : start + 64])
            for start in range(0, len(indices), 64)
        ]
        values = np.concatenate(parts)
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite classifier")
        atomic_npz(path, indices=indices, prediction=values)
        done += len(indices)
        if done % 131072 == 0:
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        queries=done,
                        seconds=time.perf_counter() - compute_start,
                    )
                ),
                flush=True,
            )
    atomic_json(
        output / f"rank-{rank}.json",
        dict(
            queries_this_allocation=done,
            pipeline_seconds=time.perf_counter() - compute_start,
            total_seconds=time.perf_counter() - started,
        ),
    )
    del model, vocab
    torch.cuda.empty_cache()
    if rank == 0:
        samudra(source, frame, config, output, device)
    dist.barrier()
    if rank == 0:
        finalize(source, frame, config, output, tasks, receipt)
    source.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
