# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Editable U-global observation/model-data baseline. Scientific choices live here.

    python -m samudra.observations.baseline train --observations OBS --output RUN

Use --mix om4-obs --om4 OM4_ROOT, or --mix om4-cm4-obs --om4 OM4_ROOT
--cm4 CM4.zarr. Paths and runtime controls are CLI arguments; architecture,
losses, source ratios, replay policy and training phases are ordinary Python.
"""

import argparse
import hashlib
import json
import math
import os
import random
import signal
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations.archive import SPLITS
from samudra.observations.evaluate import AnnualObservationEvaluator
from samudra.observations.intervals import replay_month
from samudra.observations.losses import balanced_loss, completion_loss, corrupt_sample
from samudra.observations.model_data import ModelMonths
from samudra.observations.replay import ReplayEntry, states_at
from samudra.observations.samples import Samples
from samudra.observations.state import TrainingState, write_json

SEED = 1729
MAIN_UPDATES, COOLDOWN_UPDATES = 2000, 1000
BATCH_SIZE, FRESH_EXAMPLES = 8, 4
VALIDATE_EVERY, SAVE_EVERY = 250, 25
LEARNING_RATE, FINAL_LEARNING_RATE = 1e-4, 1e-6
VALIDATION_ORIGIN = "2013-11-01"
SELECTION_METRIC = "rollout_val/365d/normalized_rmse/channel_mean"
MIXES = {
    "obs": ("obs",),
    "om4-obs": ("obs", "om4"),
    "om4-cm4-obs": ("obs", "om4", "obs", "cm4"),
}


def source_at(mix, update):
    """Whole optimizer updates have one source; cooldown is exclusively obs."""
    if update < 0 or update >= MAIN_UPDATES + COOLDOWN_UPDATES:
        raise ValueError("Update outside the training plan")
    cycle = MIXES[mix]
    return "obs" if update >= MAIN_UPDATES else cycle[update % len(cycle)]


def learning_rate(update):
    if update < MAIN_UPDATES:
        return LEARNING_RATE
    fraction = (update - MAIN_UPDATES + 1) / COOLDOWN_UPDATES
    return LEARNING_RATE + fraction * (FINAL_LEARNING_RATE - LEARNING_RATE)


class Sources:
    """One observation normalization shared by the explicitly opened sources."""

    def __init__(self, observations, models):
        self.obs, self.models = observations, models
        self.paths = observations.paths("train")
        expected = [str(m) for m in pd.period_range(*SPLITS["train"], freq="M")]
        if [p.stem for p in self.paths] != expected:
            raise ValueError("Observation training cohort is incomplete")
        self.names = ("obs", *models)

    def fresh_origin(self, source, rng):
        if source == "obs":
            return self.paths[int(rng.integers(len(self.paths)))].stem
        return int(rng.integers(len(self.models[source].windows)))

    def layout(self, source, origin, month):
        if month > 11:
            return None
        if source != "obs":
            return self.models[source].continuation_layout(int(origin), month)
        target = pd.Timestamp(origin) + pd.DateOffset(months=month)
        if not SPLITS["train"][0] <= str(target)[:7] <= SPLITS["train"][1]:
            return None
        start, stop, next_start = replay_month(origin, target)
        if stop > 73:
            return None
        return dict(start_lead=start, end_lead=stop, next_start_lead=next_start)

    def load(self, source, origin, month=0):
        if source == "obs":
            target = pd.Timestamp(origin) + pd.DateOffset(months=month)
            if month:
                sample = self.obs.trajectory(origin, target_month=target, split="train")
            else:
                sample = self.obs.load(
                    self.obs.root / "train" / (str(origin)[:7] + ".npz")
                )
                sample.update(self.layout(source, origin, month))
            sample["mask"] = self.obs.mask
            return sample
        raw = self.models[source].continuation(int(origin), month, self.obs.device)
        fields = raw["fields"]
        return {
            **raw,
            "surface": fields[:, :, [38, 76]],
            "atmosphere": raw["forcing"],
            "month_weights": raw["weights"],
            "validity": raw["mask"][[38, 76]].bool().expand_as(fields[:, :, [38, 76]]),
        }

    def auxiliary(self, source, origin, month, sample):
        if source != "obs" or month == 0:
            return sample
        target = pd.Timestamp(origin) + pd.DateOffset(months=month)
        result = self.obs.load(self.obs.root / "train" / (str(target)[:7] + ".npz"))
        result["mask"] = self.obs.mask
        return result


def objective(
    model, data, source, sample, auxiliary, seed, *, cached=None, coverage=None
):
    """One forecast window plus separate initializer reconstruction/completion.

    Replay never sees the auxiliary branch's contemporaneous ocean observations.
    Only its own detached physical history, future forcing and calendar advance it.
    """
    task = "observation" if source == "obs" else "om4"
    auxiliary = corrupt_sample(auxiliary, seed)
    if source != "obs":
        if coverage is None:
            raise ValueError("Model initialization needs observation coverage")
        visible = auxiliary["validity"].clone()
        visible[:, :19] &= coverage[:, :19].bool()
        auxiliary["validity"] = visible
        auxiliary["surface"] = torch.where(visible, auxiliary["surface"], 0)
        auxiliary["completion_valid"] = (
            sample["mask"][[38, 76]].bool().expand_as(visible[:, 17:19])
            & ~visible[:, 17:19]
        )
    args = (
        auxiliary["surface"],
        auxiliary["atmosphere"],
        auxiliary["contexts"],
        auxiliary["mask"],
        auxiliary["validity"],
    )
    if cached is None:
        prediction, initial = model.forecast(*args, task=task)
        replay_initial = initial
    else:
        replay_initial = cached.detach().to(data.device)
        prediction, _ = model.continue_forecast(
            replay_initial,
            sample["atmosphere"][:, 19:],
            sample["contexts"][:, 19:],
            sample["mask"],
            start_lead=sample["start_lead"] + 1,
            task=task,
        )
        initial = model.initialize(
            args[0][:, :19],
            args[1][:, :19],
            args[2][:, 18],
            args[3],
            args[4][:, :19],
            task,
        )
    parts = {
        "completion": completion_loss(
            initial[:, :, [38, 76]],
            auxiliary["completion_target"],
            auxiliary["completion_valid"],
            data.grid["lat"],
        )
    }
    if source == "obs":
        parts.update(data.forecast_losses(prediction, sample))
        reconstructed = model.reconstruct_month(*args, auxiliary["month_weights"])
        parts["reconstruction"] = data.interior_loss(reconstructed, auxiliary)
        forecast = 0.8 * parts["monthly_ts"] + 0.1 * parts["sst"] + 0.1 * parts["ssh"]
    else:
        weights = sample["mask"] * data.area
        parts["forecast"] = balanced_loss(
            prediction, sample["fields"][:, 19:], weights, model.names
        )
        parts["reconstruction"] = balanced_loss(
            initial,
            auxiliary["fields"][:, 17:19],
            weights,
            model.names,
            interior_only=True,
        )
        forecast = parts["forecast"]
    loss = forecast + 0.1 * parts["reconstruction"] + 0.1 * parts["completion"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite {source} objective")
    boundary = sample["next_start_lead"] - sample["start_lead"]
    next_states = states_at(replay_initial, prediction, boundary).detach()
    return loss, parts, next_states


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def annual_metrics(model, data, trajectory):
    model.eval()
    evaluator = AnnualObservationEvaluator(
        trajectory,
        data.std,
        data.area.expand(data.mask.shape[-2:]),
        surface_mask=data.mask[[38, 76]].bool(),
        profile_mask=data.ts_mask.bool(),
    )
    with autocast(data.device):
        states = model.initialize(
            trajectory["surface"][:, :19],
            trajectory["atmosphere"][:, :19],
            trajectory["contexts"][:, 18],
            data.mask,
            trajectory["validity"][:, :19],
        )
        for step in range(73):
            prediction, states = model.continue_forecast(
                states,
                trajectory["atmosphere"][:, 19 + step : 20 + step],
                trajectory["contexts"][:, 19 + step : 20 + step],
                data.mask,
                start_lead=step + 1,
            )
            evaluator.record(data.physical(prediction)[:, 0])
    return evaluator.result()


def train(
    state,
    sources,
    output,
    manifest,
    validate,
    emit,
    *,
    stop_after=None,
    stopping=lambda: False,
):
    """The complete scientific sequence is visible and editable here."""
    total = MAIN_UPDATES + COOLDOWN_UPDATES
    limit = total if stop_after is None else min(stop_after, total)
    if limit < state.updates:
        raise ValueError("Stop limit precedes the restored checkpoint")
    for update in range(state.updates, limit):
        # Specialize to observations; keep Adam moments, reset EMA exactly once.
        if update == MAIN_UPDATES and not state.cooled:
            state.ema = state.new_ema()
            state.cooled = True
        source = source_at(manifest["mix"], update)
        pool = state.pools[source]
        for group in state.optimizer.param_groups:
            group["lr"] = learning_rate(update)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        record = {
            "source": source,
            "lr": learning_rate(update),
            "max_start_lead_days": 0,
        }
        for example in range(BATCH_SIZE):
            entry = pool.draw(update) if example >= FRESH_EXAMPLES else None
            origin = (
                sources.fresh_origin(source, state.rng)
                if entry is None
                else entry.origin
            )
            month = 0 if entry is None else entry.month
            sample = sources.load(source, origin, month)
            if entry is not None and entry.start_lead != sample["start_lead"]:
                raise ValueError("Replay state time differs from target window")
            auxiliary = sources.auxiliary(source, origin, month, sample)
            coverage = None
            if source != "obs":
                coverage_origin = sources.fresh_origin("obs", state.rng)
                coverage = sources.load("obs", coverage_origin)["validity"]
            seed = int(state.rng.integers(2**31))
            with autocast(sources.obs.device):
                loss, parts, next_states = objective(
                    state.model,
                    sources.obs,
                    source,
                    sample,
                    auxiliary,
                    seed,
                    cached=None if entry is None else entry.states,
                    coverage=coverage,
                )
            (loss / BATCH_SIZE).backward()
            exposure = "fresh" if entry is None else "replay"
            state.exposure[source][exposure] += 1
            record[exposure] = record.get(exposure, 0) + 1
            record["max_start_lead_days"] = max(
                record["max_start_lead_days"], sample["start_lead"] * 5
            )
            for name, value in {"loss": loss, **parts}.items():
                record[name] = (
                    record.get(name, 0.0) + float(value.detach()) / BATCH_SIZE
                )
            following = sources.layout(source, origin, month + 1)
            if following is not None:
                if following["start_lead"] != sample["next_start_lead"]:
                    raise ValueError(
                        "Replay boundary differs from next month's input time"
                    )
                pool.add(
                    ReplayEntry(
                        source,
                        origin,
                        month + 1,
                        following["start_lead"],
                        update if entry is None else entry.origin_update,
                        next_states,
                    )
                )
        norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), 1.0, error_if_nonfinite=True
        )
        state.optimizer.step()
        state.ema(state.model)
        state.updates = update + 1
        state.exposure[source]["updates"] += 1
        emit(
            {
                "updates": state.updates,
                "train/gradient_norm": float(norm),
                **{
                    f"train/{source}/{key}": value
                    for key, value in record.items()
                    if key != "source"
                },
                **{
                    f"exposure/{name}/{key}": value
                    for name, counts in state.exposure.items()
                    for key, value in counts.items()
                },
            }
        )
        if state.updates % VALIDATE_EVERY == 0 or state.updates == total:
            with state.averaged_model():
                metrics = validate(state.model)
            score = metrics[SELECTION_METRIC]
            if not math.isfinite(score):
                raise FloatingPointError("Annual selection score is nonfinite")
            if score < state.best_score:
                state.best_score = score
                state.save(output / "best.pt", manifest)
            emit({"updates": state.updates, **metrics})
        if state.updates == MAIN_UPDATES:
            state.save(output / "main.pt", manifest)
        if state.updates % SAVE_EVERY == 0 or state.updates == limit or stopping():
            state.save(output / "latest.pt", manifest)
        if stopping():
            break
    if state.updates == total:
        state.save(output / "endpoint.pt", manifest)
        write_json(
            {"updates": total, "exposure": state.exposure}, output / "COMPLETE.json"
        )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_normalization(directory):
    """Read back fitted samples; a saved manifest alone is not proof."""
    proof = json.loads((directory / "normalization.json").read_text())
    if proof["training_split"] != list(SPLITS["train"]):
        raise ValueError("Normalization uses a different training split")
    for path, expected in proof["sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Normalization input changed: {path}")


def archive_identity(root):
    """Change detection, not a full content read-back of the daily archive.

    Record names, sizes, mtimes and preparation manifests. Content hashes where
    provided by preprocessing remain in the included manifests.
    """
    identity = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".npz", ".json"):
            stat = path.stat()
            signature = (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
            identity.update(json.dumps(signature).encode())
            if path.suffix == ".json":
                identity.update(path.read_bytes())
            count += 1
    if count == 0:
        raise ValueError("Daily observation archive is empty")
    return {"inventory_sha256": identity.hexdigest(), "files": count}


def run_manifest(args, data, sources):
    package = Path(__file__).resolve().parents[1]
    normalization = Path(args.normalization or args.observations)
    verify_normalization(normalization)
    files = [Path(args.observations) / "grid.npz"] + [
        normalization / n
        for n in ("normalization.npz", "normalization.json", "statistics.npz")
    ]
    for kind in sources.models:
        root = Path(args.om4) / "OM4.zarr" if kind == "om4" else Path(args.cm4)
        files.append(root / ".zmetadata")
    if sources.models:
        files.extend(
            Path(args.om4) / name / ".zmetadata"
            for name in ("OM4_means.zarr", "OM4_stds.zarr")
        )
    return {
        "version": 1,
        "mix": args.mix,
        "seed": SEED,
        "main_updates": MAIN_UPDATES,
        "cooldown_updates": COOLDOWN_UPDATES,
        "batch_size": BATCH_SIZE,
        "fresh_examples": FRESH_EXAMPLES,
        "validation_origin": VALIDATION_ORIGIN,
        "selection_metric": SELECTION_METRIC,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "channels": data.grid["names"].tolist(),
        "shape": list(data.mask.shape),
        "observations": str(Path(args.observations).resolve()),
        "daily": str(Path(args.daily or Path(args.observations) / "daily").resolve()),
        "daily_identity": archive_identity(
            Path(args.daily or Path(args.observations) / "daily")
        ),
        "model_identity_scope": "Zarr metadata; field chunks checked for support and finiteness when read",
        "data_sha256": {str(path.resolve()): digest(path) for path in files},
        "code_sha256": {
            str(path.relative_to(package)): digest(path)
            for path in sorted(package.rglob("*.py"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--daily", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--om4", type=Path)
    parser.add_argument("--cm4", type=Path)
    parser.add_argument("--mix", choices=tuple(MIXES), default="obs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--project", default="observational-transfer")
    parser.add_argument("--entity")
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if "om4" in MIXES[args.mix] and args.om4 is None:
        parser.error("Model mixes require --om4 (including native forcing statistics)")
    if "cm4" in MIXES[args.mix] and args.cm4 is None:
        parser.error("OM4/CM4 mix requires --cm4")
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    data = Samples(
        args.observations,
        args.device,
        daily=args.daily,
        normalization=args.normalization,
    )
    models = {}
    if "om4" in MIXES[args.mix]:
        models["om4"] = ModelMonths(args.om4 / "OM4.zarr", "om4", data.grid, args.om4)
    if "cm4" in MIXES[args.mix]:
        models["cm4"] = ModelMonths(args.cm4, "cm4", data.grid, args.om4)
    sources = Sources(data, models)
    model = SurfaceInitializedModel(data.grid["names"].tolist()).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    state = TrainingState(model, optimizer, sources.names, SEED)
    manifest = run_manifest(args, data, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Output directory belongs to another code/data/recipe manifest"
        )
    checkpoint = args.checkpoint
    if checkpoint is None and (args.output / "latest.pt").exists():
        checkpoint = args.output / "latest.pt"
    if checkpoint is not None:
        state.restore(checkpoint, manifest)
    write_json(manifest, manifest_path)
    trajectory = data.trajectory(VALIDATION_ORIGIN, steps=73, split="validation")

    def validate(current_model):
        return annual_metrics(current_model, data, trajectory)

    if args.command == "evaluate":
        with state.averaged_model():
            metrics = validate(model)
        write_json(
            {"checkpoint_sha256": digest(checkpoint), **metrics},
            args.output / "evaluation.json",
        )
        print(json.dumps(metrics, allow_nan=False), flush=True)
        return
    import wandb

    id_path = args.output / "wandb-id.txt"
    if not id_path.exists():
        id_path.write_text(wandb.util.generate_id())
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        mode=args.wandb_mode,
        id=id_path.read_text().strip(),
        resume="allow",
        config=manifest,
        name=args.output.name,
    )
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    def emit(record):
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        run.log(record, step=record["updates"])
        print(json.dumps(record, allow_nan=False), flush=True)

    try:
        train(
            state,
            sources,
            args.output,
            manifest,
            validate,
            emit,
            stop_after=args.stop_after,
            stopping=lambda: stopped,
        )
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
