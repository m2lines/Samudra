# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched temperature maps and held-out candidate likelihoods, restartable by chunk."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from samudra.metrics.spectra import SPATIAL_REGIONS

from .baseline import file_sha256
from .data import OceanSource
from .model import ModelConfig, PointTransformer, Vocabulary
from .run import to_device


def evaluation_dates(source, config):
    """Pin calendar dates before scoring; use first valid origin on/after each date."""
    result = []
    available = source.origins["test"]
    for role, dates in config["dates"].items():
        for date in dates:
            eligible = available[source.date_strings[available] >= date]
            if not len(eligible):
                raise ValueError(f"No test origin on/after {date}")
            origin = int(eligible[0])
            if source.date_strings[origin][:4] != date[:4]:
                raise ValueError("Requested evaluation year is unavailable")
            result.append(
                dict(
                    role=role,
                    requested=date,
                    origin=origin,
                    date=str(source.date_strings[origin]),
                    target_date=str(source.date_strings[origin + 1]),
                    dense=date == config["map_date"],
                )
            )
    if len({r["origin"] for r in result}) != len(result):
        raise ValueError("Calibration and assessment dates must be disjoint")
    if not any(r["dense"] and r["role"] == "assessment" for r in result):
        raise ValueError("Map date must be in the assessment split")
    return result


def region_masks(source):
    masks = {}
    for name, lon, lat in SPATIAL_REGIONS:
        masks[name] = (
            (source.lon[None] >= lon.start)
            & (source.lon[None] < lon.stop)
            & (source.lat[:, None] >= lat.start)
            & (source.lat[:, None] < lat.stop)
        )
    return masks


def query_points(source, depth, origin, dense, config):
    if dense:
        rows, cols = np.indices((source.height, source.width))
        selected = (rows % config["stride"] == 0) & (cols % config["stride"] == 0)
        for mask in region_masks(source).values():
            selected |= mask
        flat = np.flatnonzero(selected & source.wet[depth])
        limit = config.get("qualification_dense_limit")
        if limit:
            flat = flat[np.linspace(0, len(flat) - 1, min(limit, len(flat)), dtype=int)]
    else:
        rng = np.random.default_rng(
            np.random.SeedSequence([config["seed"], origin, depth])
        )
        flat = source.locations[depth][
            np.searchsorted(
                source.location_cdfs[depth], rng.random(config["points_per_depth"])
            )
        ]
    return np.column_stack(np.unravel_index(flat, (source.height, source.width)))


def distribution_scores(logits, vocab, depth, values, std):
    """Score supplied candidates; no future target is needed for Samudra scores."""
    logp = logits.float().log_softmax(-1)
    p = logp.exp()
    cdf = p.cumsum(-1)
    edges = vocab.edges[depth]
    out = {}
    for name, value in values.items():
        ids = vocab.encode(value, depth)
        outside = vocab.outside(value, depth)
        mass = p.gather(1, ids[:, None]).squeeze(1)
        lower = edges.gather(1, ids[:, None]).squeeze(1)
        upper = edges.gather(1, (ids + 1)[:, None]).squeeze(1)
        before = cdf.gather(1, ids[:, None]).squeeze(1) - mass
        pit = before + mass * ((value - lower) / (upper - lower)).clamp(0, 1)
        pit = torch.where(value < edges[:, 0], 0, pit)
        pit = torch.where(value > edges[:, -1], 1, pit).clamp(0, 1)
        out[name + "_bin_nll"] = (
            -logp.gather(1, ids[:, None]).squeeze(1).masked_fill(outside, -torch.inf)
        )
        out[name + "_density_nll"] = -vocab.log_density(logits, value, depth, std)
        out[name + "_cdf"] = pit
        out[name + "_outside"] = outside
    centers = vocab.centers[depth]
    mean = (p * centers).sum(-1)
    out["predictive_std_c"] = (
        (p * (centers - mean[:, None]).square()).sum(-1)
    ).sqrt() * std[depth]
    out["entropy_nats"] = -(p * logp).sum(-1)
    return out


def load_model(path, device):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = state["config"]
    model = PointTransformer(ModelConfig(**config["model"]))
    model.load_state_dict(state["model"], strict=True)
    vocab = Vocabulary(state["vocabulary"]["centers"][:, -1], model.config.bins)
    vocab.load_state_dict(state["vocabulary"], strict=True)
    return model.to(device).eval(), vocab.to(device), config, state["step"]


@torch.inference_mode()
def run(config, output):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_num_threads(config["torch_threads"])
    blosc.set_nthreads(1)
    source = OceanSource(
        config["source"], config["readers"], config.get("reference_root")
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        classifier, vocab, cc, cs = load_model(config["classifier"], device)
        regression, _, rc, rs = load_model(config["regression"], device)
        if (
            classifier.config.objective != "absolute_ce"
            or regression.config.objective != "regression"
        ):
            raise ValueError("Expected absolute classifier and scalar regression")
        if cc["crop"] != rc["crop"] or cc["crop"] != config["crop"]:
            raise ValueError("Input crops differ")
        baseline = Path(config["baseline_cache"])
        manifest = json.loads((baseline / "manifest.json").read_text())
        support = json.loads(Path(config["support"]).read_text())
        for key in ("metadata_sha256", "mean", "std"):
            if (
                source.receipt()[key] != manifest["source"][key]
                or source.receipt()[key] != support["source"][key]
            ):
                raise ValueError(f"Source normalization/provenance mismatch: {key}")
        dates = evaluation_dates(source, config)
        receipt = dict(
            config=config,
            dates=dates,
            source=source.receipt(),
            classifier_sha256=file_sha256(config["classifier"]),
            regression_sha256=file_sha256(config["regression"]),
            classifier_step=cs,
            regression_step=rs,
            baseline_sha256=manifest["checkpoint_sha256"],
            source_commit=os.environ.get("SKYLINE_COMMIT"),
            world_size=world,
        )
        receipt_path = output / f"receipt-rank-{rank}.json"
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Refusing to mix map runs with different provenance")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        means = torch.tensor(source.mean[38:57], device=device)
        stds = torch.tensor(source.std[38:57], device=device)
        rng = np.random.default_rng(0)
        task = 0
        begin = time.monotonic()
        evaluated = 0
        for date in dates:
            origin = date["origin"]
            frame = None
            base = np.load(baseline / f"temperature-{origin:04d}.npy", mmap_mode="r")
            for z in range(19):
                points = query_points(source, z, origin, date["dense"], config)
                for start in range(0, len(points), config["chunk_size"]):
                    task += 1
                    if (task - 1) % world != rank:
                        continue
                    dest = output / f"points-{origin:04d}-{z:02d}-{start:07d}.npz"
                    if dest.exists():
                        continue
                    if frame is None:
                        frame = source.read(origin)
                    chosen = points[start : start + config["chunk_size"]]
                    arrays: dict[str, list] = {}
                    for offset in range(0, len(chosen), config["microbatch"]):
                        pp = chosen[offset : offset + config["microbatch"]]
                        depths = np.full(len(pp), z)
                        batch = to_device(
                            source.sample(
                                frame, rng, len(pp), config["crop"], depths, pp
                            ),
                            device,
                        )
                        # Explicitly exclude targets from both forward calls.
                        inputs = {k: v for k, v in batch.items() if k != "target"}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = classifier(inputs)
                            prediction = regression(inputs).squeeze(-1)
                        sam = torch.tensor(
                            np.array(base[z, pp[:, 0], pp[:, 1]]), device=device
                        )
                        result = distribution_scores(
                            logits,
                            vocab,
                            batch["depth"],
                            {"gold": batch["target"], "samudra": sam},
                            stds,
                        )
                        for name, value in dict(
                            gold=batch["target"],
                            samudra=sam,
                            regression=prediction,
                            persistence=batch["current"],
                        ).items():
                            result[name + "_c"] = value * stds[z] + means[z]
                        result["classifier_mode_c"] = (
                            vocab.decode(logits, batch["depth"]) * stds[z] + means[z]
                        )
                        for key, value in result.items():
                            arrays.setdefault(key, []).append(value.cpu().numpy())
                    payload = {k: np.concatenate(v) for k, v in arrays.items()}
                    payload.update(
                        row=chosen[:, 0],
                        col=chosen[:, 1],
                        depth=np.full(len(chosen), z),
                        origin=np.full(len(chosen), origin),
                        area=source.area[chosen[:, 0], chosen[:, 1]],
                        lat=source.lat[chosen[:, 0]],
                        lon=source.lon[chosen[:, 1]],
                        global_lattice=(chosen[:, 0] % config["stride"] == 0)
                        & (chosen[:, 1] % config["stride"] == 0),
                    )
                    if any(
                        not np.isfinite(payload[k]).all()
                        for k in (
                            "gold_c",
                            "samudra_c",
                            "regression_c",
                            "persistence_c",
                        )
                    ):
                        raise FloatingPointError("Nonfinite physical prediction")
                    tmp = dest.with_suffix(".tmp")
                    with tmp.open("wb") as f:
                        np.savez_compressed(f, **payload)
                    tmp.replace(dest)
                    evaluated += len(chosen)
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                origin=origin,
                                depth=z,
                                chunk=start,
                                points=evaluated,
                                seconds=time.monotonic() - begin,
                            )
                        ),
                        flush=True,
                    )
        (output / f"COMPLETE-rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    points_evaluated=evaluated,
                    seconds=time.monotonic() - begin,
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
