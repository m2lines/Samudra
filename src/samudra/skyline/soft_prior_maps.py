# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded native-cell maps from independently configured skyline checkpoints."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from .baseline import file_sha256
from .channel_vocabulary import ChannelVocabulary
from .ema import restore_ema
from .model import ModelConfig, PointTransformer
from .multitask import ComparisonSource
from .run import to_device

FIELD_INDEX = {"uo": 0, "vo": 1, "thetao": 2, "so": 3, "zos": 4}


def channel_index(field, depth):
    if field not in FIELD_INDEX or not 0 <= depth < 19:
        raise ValueError("Unknown field or depth")
    if field == "zos":
        if depth != 0:
            raise ValueError("SSH is surface-only")
        return 76
    return FIELD_INDEX[field] * 19 + depth


def query_column(config, field):
    if field not in FIELD_INDEX:
        raise ValueError("Unknown field")
    if config["arm"] == "temperature":
        if field != "thetao":
            raise ValueError("Temperature model has no supervised auxiliary forecast")
        return 0
    if config["arm"] != "all_fields":
        raise ValueError("Map runner requires sampled-depth field queries")
    if config.get("temperature_loss_only", False) and field != "thetao":
        raise ValueError("Temperature-only loss leaves auxiliary outputs unsupervised")
    return FIELD_INDEX[field]


def task_fields(task, model_name):
    """Select explicitly supervised fields while sharing a native task rectangle."""
    overrides = task.get("fields_by_model", {})
    if not set(overrides).issubset(task["models"]):
        raise ValueError("Field overrides must reference participating task models")
    fields = overrides.get(model_name, task["fields"])
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Task fields must be nonempty and unique")
    if any(field not in FIELD_INDEX for field in fields):
        raise ValueError("Unknown task field")
    return fields


def task_grid(source, task):
    """Select native rectangular cells; bounds use half-open latitude/longitude."""
    if "rows" in task or "cols" in task:
        if "rows" not in task or "cols" not in task:
            raise ValueError("Explicit task needs both row and column bounds")
        rows = np.arange(*task["rows"])
        cols = np.arange(*task["cols"])
    else:
        rows = np.flatnonzero(
            (source.lat >= task["lat"][0]) & (source.lat < task["lat"][1])
        )
        cols = np.flatnonzero(
            (source.lon >= task["lon"][0]) & (source.lon < task["lon"][1])
        )
    if (
        not len(rows)
        or not len(cols)
        or rows.min() < 0
        or cols.min() < 0
        or rows.max() >= source.height
        or cols.max() >= source.width
    ):
        raise ValueError("Task rectangle is empty or outside the native grid")
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    return rows, cols, yy, xx


def point_readout(output, batch, field, config, vocab=None):
    """Return normalized point predictions and optional categorical diagnostics."""
    col = query_column(config, field)
    if config["model"]["objective"] == "regression":
        return output[:, col].float(), {}
    if vocab is None or config["model"]["objective"] != "absolute_ce":
        raise ValueError("Expected regression or absolute categorical model")
    logits = output[:, col].float()
    channels = batch["query_channel"][:, col]
    ids = logits.argmax(-1)
    prediction = vocab.centers[channels, ids]
    true_ids = vocab.encode(batch["target"][:, col], channels)
    logp = logits.log_softmax(-1)
    details = dict(
        predicted_bin=ids,
        true_bin=true_ids,
        predicted_probability=logp.gather(1, ids[:, None]).squeeze(1).exp(),
        true_probability=logp.gather(1, true_ids[:, None]).squeeze(1).exp(),
        quantized_target=vocab.centers[channels, true_ids],
    )
    return prediction, details


def load_selection(selection, source, device):
    path = Path(selection["checkpoint"])
    digest = file_sha256(path)
    if selection.get("sha256", digest) != digest:
        raise ValueError("Checkpoint checksum changed")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = state["config"]
    if state["step"] != selection["step"]:
        raise ValueError("Checkpoint update changed")
    if not cfg["model"].get("field_queries", False):
        raise ValueError("This runner requires field-query checkpoints")
    provenance_path = Path(
        selection.get("provenance", path.parent / "provenance-restart-0.json")
    )
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    if provenance is not None and provenance["config"] != cfg:
        raise ValueError("Snapshot/provenance configuration mismatch")
    training_source = state.get("source")
    if training_source is None:
        if provenance is None:
            raise ValueError("Snapshot needs its training source provenance")
        training_source = provenance["source"]
    for key in ("metadata_sha256", "mean", "std"):
        if training_source[key] != source.receipt()[key]:
            raise ValueError(f"Checkpoint/source mismatch: {key}")
    if (
        cfg["source"] != source.root
        or cfg.get("reference_root") != source.reference_root
    ):
        raise ValueError("Checkpoint source paths do not match map source")
    model = PointTransformer(ModelConfig(**cfg["model"]))
    model.load_state_dict(state["model"])
    if selection["from_latest"]:
        ema = restore_ema(model, cfg["ema"], state["ema"], state["step"])
        ema.copy_to(model)
        del ema
    elif state.get("weights_kind") != "ema":
        raise ValueError("Snapshot must explicitly contain EMA weights")
    elif int(state["ema"]["num_updates"]) != state["step"]:
        raise ValueError("Snapshot EMA update count differs from checkpoint step")
    vocab = (
        ChannelVocabulary(cfg["vocabulary_bounds"], cfg["model"]["bins"]).to(device)
        if cfg["model"]["objective"] == "absolute_ce"
        else None
    )
    receipt = dict(
        selection,
        checkpoint_sha256=digest,
        config=cfg,
        source=source.receipt(),
        parameters=sum(p.numel() for p in model.parameters()),
        weights="ema",
        weights_kind="ema",
        checkpoint_step=state["step"],
        architecture=cfg["model"],
        crop=cfg["crop"],
        training_source_commit=provenance.get("source_commit")
        if provenance
        else state.get("source_commit"),
        inference_commit=os.environ.get("SKYLINE_COMMIT"),
    )
    return model.to(device).eval(), vocab, cfg, receipt


@torch.inference_mode()
def run(config, output):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Bounded map runner uses one GPU")
    started = time.monotonic()

    def check_budget():
        if "SKYLINE_ALLOCATION_START" in os.environ:
            elapsed = (
                float(os.environ.get("SKYLINE_PRIOR_SECONDS", 0))
                + time.time()
                - float(os.environ["SKYLINE_ALLOCATION_START"])
            )
        else:
            elapsed = time.monotonic() - started
        if elapsed >= config.get("max_allocated_seconds", 7200):
            raise RuntimeError("Map allocation budget reached; complete tasks retained")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_num_threads(config.get("torch_threads", 4))
    blosc.set_nthreads(1)
    if (
        config.get("microbatch", 32) < 1
        or config.get("max_allocated_seconds", 7200) <= 0
    ):
        raise ValueError("Map batch and budget must be positive")
    output.mkdir(parents=True, exist_ok=True)
    source = ComparisonSource(
        config["source"],
        config.get("readers", 4),
        config.get("reference_root"),
        arm="all_fields",
    )
    try:
        task_ids = [t["name"] for t in config["tasks"]]
        model_ids = [m["name"] for m in config["models"]]
        if len(set(task_ids)) != len(task_ids) or len(set(model_ids)) != len(model_ids):
            raise ValueError("Model and task names must be unique")
        if not task_ids or not model_ids:
            raise ValueError("Map configuration must contain models and tasks")
        for task in config["tasks"]:
            if not task["models"] or not set(task["models"]).issubset(model_ids):
                raise ValueError("Task references an unknown model")
            for model_name in task["models"]:
                task_fields(task, model_name)
        for selection in config["models"]:
            check_budget()
            model, vocab, cfg, receipt = load_selection(selection, source, device)
            source.arm = cfg["arm"]
            receipt["tasks"] = config["tasks"]
            receipt["map_config"] = config
            receipt_path = output / f"{selection['name']}-receipt.json"
            if (
                receipt_path.exists()
                and json.loads(receipt_path.read_text()) != receipt
            ):
                raise ValueError("Map provenance changed across restart")
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            current_origin = None
            frame: Any = None
            for task in config["tasks"]:
                if selection["name"] not in task["models"]:
                    continue
                fields = task_fields(task, selection["name"])
                depth = task["depth"]
                for field in fields:
                    channel_index(field, depth)
                    query_column(cfg, field)
                dest = output / f"{selection['name']}-{task['name']}.npz"
                if dest.exists():
                    with np.load(dest) as saved:
                        if (
                            str(saved["checkpoint_sha256"])
                            != receipt["checkpoint_sha256"]
                            or int(saved["origin"]) != task["origin"]
                        ):
                            raise ValueError("Completed task provenance mismatch")
                    continue
                check_budget()
                origin = task["origin"]
                split = task.get("split", "test")
                if (
                    split not in {"test", "validation"}
                    or origin not in source.origins[split]
                ):
                    raise ValueError(
                        "Task date does not belong to its declared evaluation split"
                    )
                if origin != current_origin:
                    frame = source.read(origin)
                    current_origin = origin
                rows, cols, yy, xx = task_grid(source, task)
                wet = source.wet[depth][yy, xx]
                points = np.column_stack((yy[wet], xx[wet]))
                if not len(points):
                    raise ValueError("Task has no wet cells at this depth")
                if len(points) > config.get("max_points_per_task", len(points)):
                    raise ValueError(
                        "Task exceeds point cap; select a smaller native rectangle"
                    )
                values: dict[str, list[np.ndarray]] = {field: [] for field in fields}
                extras: dict[str, dict[str, list[np.ndarray]]] = {
                    field: {} for field in fields
                }
                for start in range(0, len(points), config.get("microbatch", 32)):
                    check_budget()
                    pp = points[start : start + config.get("microbatch", 32)]
                    batch = to_device(
                        source.sample(
                            frame,
                            np.random.default_rng(0),
                            len(pp),
                            cfg["crop"],
                            np.full(len(pp), depth),
                            pp,
                        ),
                        device,
                    )
                    if not np.array_equal(
                        batch["row"].cpu().numpy(), pp[:, 0]
                    ) or not np.array_equal(batch["col"].cpu().numpy(), pp[:, 1]):
                        raise ValueError("Query coordinates changed")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        raw = model({k: v for k, v in batch.items() if k != "target"})
                    for field in fields:
                        pred, extra = point_readout(raw, batch, field, cfg, vocab)
                        values[field].append(pred.cpu().numpy())
                        for key, value in extra.items():
                            extras[field].setdefault(key, []).append(
                                value.cpu().numpy()
                            )
                payload = dict(
                    wet=wet,
                    lat=source.lat[rows],
                    lon=source.lon[cols],
                    area=source.area[yy, xx],
                    query_rows=yy,
                    query_cols=xx,
                    depth=depth,
                    origin=origin,
                    input_date=str(source.date_strings[origin]),
                    target_date=str(source.date_strings[origin + 1]),
                    checkpoint_sha256=receipt["checkpoint_sha256"],
                    crop=cfg["crop"],
                    split=split,
                )
                for field in fields:
                    channel = channel_index(field, depth)
                    scale, mean = source.std[channel], source.mean[channel]
                    pred = np.full(wet.shape, np.nan, dtype=np.float32)
                    pred[wet] = np.concatenate(values[field]) * scale + mean
                    gold = frame["target"][channel][yy, xx] * scale + mean
                    persistence = frame["input"][1, channel][yy, xx] * scale + mean
                    if not all(
                        np.isfinite(x[wet]).all() for x in (pred, gold, persistence)
                    ):
                        raise FloatingPointError("Nonfinite wet-cell values")
                    payload.update(
                        {
                            f"{field}_prediction": pred,
                            f"{field}_gold": gold,
                            f"{field}_persistence": persistence,
                            f"{field}_units": {
                                "thetao": "degC",
                                "so": "PSU",
                                "uo": "m/s",
                                "vo": "m/s",
                                "zos": "m",
                            }[field],
                        }
                    )
                    for key, arrays in extras[field].items():
                        grid = np.full(wet.shape, np.nan, dtype=np.float64)
                        grid[wet] = np.concatenate(arrays)
                        if key == "quantized_target":
                            grid = grid * scale + mean
                        payload[f"{field}_{key}"] = grid
                partial = dest.with_suffix(".partial.npz")
                np.savez_compressed(partial, **payload)
                partial.replace(dest)
                print(
                    json.dumps(
                        dict(
                            model=selection["name"],
                            task=task["name"],
                            points=len(points),
                            fields=fields,
                        )
                    ),
                    flush=True,
                )
            del model, vocab
            torch.cuda.empty_cache()
        (output / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    models=model_ids,
                    tasks=task_ids,
                    elapsed_seconds=time.monotonic() - started,
                )
            )
            + "\n"
        )
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
