# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact accepted-draw replay and discrete validation outliers for the pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from .channel_vocabulary import ChannelVocabulary
from .coverage import sample_coverage
from .model import ModelConfig, PointTransformer
from .multitask import ComparisonSource, full_grid_weights, query_channels
from .run import FrameStream, to_device

DECODERS = ("flat2048", "flat8192", "coarse128-fine64")
ARMS = tuple(
    f"{task}-{decoder}"
    for decoder in DECODERS
    for task in ("temperature", "all-fields")
)
DEPTH = 13
CHANNEL = 38 + DEPTH


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def point_batch(source, frame, depths, points):
    """Same accepted center targets/weights as ComparisonSource, without crops."""
    depths = np.asarray(depths)
    points = np.asarray(points)
    rows, cols = points[:, 0], points[:, 1]
    channels = query_channels("all_fields", depths)
    weights, valid = full_grid_weights(
        "all_fields",
        depths,
        rows,
        cols,
        channels,
        source.wet,
        source.area,
        source.wet_area,
    )
    return {
        "depth": torch.from_numpy(depths.copy()),
        "row": torch.from_numpy(rows.copy()),
        "col": torch.from_numpy(cols.copy()),
        "query_channel": torch.from_numpy(channels),
        "query_valid": torch.from_numpy(valid),
        "target": torch.from_numpy(
            frame["target"][channels, rows[:, None], cols[:, None]].copy()
        ),
        "query_current": torch.from_numpy(
            frame["input"][1, channels, rows[:, None], cols[:, None]].copy()
        ),
        "loss_weight": torch.from_numpy(weights),
    }


class ReplaySource(ComparisonSource):
    def sample(self, frame, rng, size, crop, depth_sequence=None, points=None):
        if points is None or depth_sequence is None or len(points) != size:
            raise ValueError("Replay requires explicit accepted coverage locations")
        return point_batch(self, frame, depth_sequence, points)


def replay(training_root, output):
    output.mkdir(parents=True, exist_ok=True)
    base = None
    expected = None
    source_receipt = None
    checks = {}
    sampler_keys = (
        "source",
        "reference_root",
        "seed",
        "world_size",
        "microbatch",
        "accumulation",
        "samples_per_frame",
        "coverage",
        "crop",
        "max_updates",
        "vocabulary_bounds",
    )
    for arm in ARMS:
        state = torch.load(
            training_root / arm / "latest.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        config = state["config"]
        assert state["step"] == 2000 and config["world_size"] == 1
        rank = state["rank_states"][0]
        sampler = {k: config[k] for k in sampler_keys}
        if base is None:
            base, expected, source_receipt = (
                config,
                dict(rng=rank["rng"], stream=rank["stream"]),
                state["source"],
            )
            reference_sampler = sampler
        assert sampler == reference_sampler, arm
        assert dict(rng=rank["rng"], stream=rank["stream"]) == expected, arm
        assert state["source"] == source_receipt, arm
        checks[arm] = dict(step=state["step"], sampling_state_matches=True)
        del state
    assert base is not None and expected is not None
    source = ReplaySource(base["source"], 16, base["reference_root"], arm="all_fields")
    assert source.receipt() == source_receipt
    rng = np.random.default_rng(base["seed"])
    stream = FrameStream(source, base["seed"] + 100)
    n = base["max_updates"] * base["microbatch"] * base["accumulation"]
    metadata = np.empty((n, 5), dtype=np.int32)
    targets = np.empty((n, 5), dtype=np.float32)
    current = np.empty_like(targets)
    weights = np.empty_like(targets)
    importance = np.empty(n, dtype=np.float32)
    position = 0
    try:
        for step in range(1, base["max_updates"] + 1):
            for _ in range(base["accumulation"]):
                if stream.used >= base["samples_per_frame"]:
                    stream.advance()
                batch = sample_coverage(
                    source,
                    stream.frame,
                    rng,
                    base["microbatch"],
                    base["crop"],
                    base["coverage"],
                )
                end = position + base["microbatch"]
                metadata[position:end] = np.column_stack(
                    (
                        np.full(base["microbatch"], step),
                        np.full(base["microbatch"], stream.current_origin),
                        batch["depth"].numpy(),
                        batch["row"].numpy(),
                        batch["col"].numpy(),
                    )
                )
                targets[position:end] = batch["target"].numpy()
                current[position:end] = batch["query_current"].numpy()
                weights[position:end] = batch["loss_weight"].numpy()
                importance[position:end] = batch["importance"].numpy()
                position = end
                stream.used += base["microbatch"]
            if step % 100 == 0:
                print(
                    json.dumps(dict(event="replay", step=step, accepted=position)),
                    flush=True,
                )
        assert position == n
        assert rng.bit_generator.state == expected["rng"], (
            "Coverage RNG differs from saved state"
        )
        assert stream.state() == expected["stream"], (
            "FrameStream differs from saved state"
        )
    finally:
        stream.close()
        source.close()
    depths = metadata[:, 2]
    channels = query_channels("all_fields", depths)
    valid = weights > 0
    mass = importance[:, None].astype(np.float64) * weights
    counts = {}
    for bins in (2048, 8192):
        vocab = ChannelVocabulary(base["vocabulary_bounds"], bins)
        ids = np.empty_like(channels)
        for c in range(77):
            selected = channels == c
            ids[selected] = np.searchsorted(
                vocab.boundaries[c].numpy(), targets[selected], side="left"
            )
        flat = channels[valid] * bins + ids[valid]
        shape = (77, bins)
        counts[f"count_{bins}"] = np.bincount(flat, minlength=77 * bins).reshape(shape)
        counts[f"mass_{bins}"] = np.bincount(
            flat, weights=mass[valid], minlength=77 * bins
        ).reshape(shape)
        counts[f"centers_{bins}"] = vocab.centers.numpy()
        counts[f"ids_{bins}"] = ids.astype(np.int32)
    np.savez_compressed(
        output / "training-draws.npz",
        metadata=metadata,
        targets=targets,
        current=current,
        weights=weights,
        importance=importance,
        **counts,
    )
    deep = depths == DEPTH
    physical = targets[deep, 2] * source.std[CHANNEL] + source.mean[CHANNEL]
    write_json(
        output / "replay.json",
        dict(
            arms=checks,
            accepted=n,
            accepted_depth13=int(deep.sum()),
            unique_depth13_date_cells=len(
                np.unique(metadata[deep][:, [1, 3, 4]], axis=0)
            ),
            point_rng_matches=True,
            stream_matches=True,
            frame_state=expected["stream"],
            source=source_receipt,
            configuration=base,
            depth13_target_quantiles=np.quantile(
                physical, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
            ).tolist(),
            depth13_warm_counts={
                str(t): int((physical > t).sum()) for t in (5, 8, 10, 15, 20)
            },
            metadata_columns=["update", "origin", "depth", "row", "col"],
            weighted_mass="sum(coverage importance * original equal-channel full-grid loss weight), before division by effective batch; valid active channels only; temperature-only arm zeros all other channels",
        ),
    )
    print("REPLAY VERIFIED", flush=True)


@torch.no_grad()
def infer(training_root, output):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    models, vocabs, configs, receipts = {}, {}, {}, {}
    for arm in ARMS:
        saved = torch.load(
            training_root / arm / "weights-ema-0002000.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        config = saved["config"]
        model = PointTransformer(ModelConfig(**config["model"])).to(device).eval()
        model.load_state_dict(saved["model"])
        models[arm], configs[arm] = model, config
        vocabs[arm] = ChannelVocabulary(
            config["vocabulary_bounds"], config["model"]["bins"]
        ).to(device)
        receipts[arm] = saved["validation"]
        del saved
    config = configs[ARMS[0]]
    source = ComparisonSource(
        config["source"], 8, config["reference_root"], arm="all_fields"
    )
    origins = source.origins["validation"]
    origins = origins[
        np.linspace(0, len(origins) - 1, config["validation_dates"], dtype=int)
    ]
    rng = np.random.default_rng(config["validation_seed"])
    records = []
    probability_arrays: dict[str, list] = {arm: [] for arm in ARMS}
    try:
        for origin in origins:
            frame = source.read(int(origin))
            depths = np.tile(np.arange(19), config["validation_points_per_depth"])
            for start in range(0, len(depths), config["microbatch"]):
                selected = depths[start : start + config["microbatch"]]
                batch = to_device(
                    source.sample(frame, rng, len(selected), config["crop"], selected),
                    device,
                )
                deep = np.flatnonzero(selected == DEPTH)
                if not len(deep):
                    continue
                for arm in ARMS:
                    model, vocab = models[arm], vocabs[arm]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(batch)
                    # Preserve original mixed-depth batch layout; select only afterward.
                    logits = logits[deep, 2].float()
                    probability = logits.softmax(-1)
                    ids = logits.argmax(-1)
                    target = batch["target"][deep, 2]
                    true_ids = torch.searchsorted(
                        vocab.boundaries[CHANNEL].contiguous(), target.contiguous()
                    )
                    prediction = vocab.centers[CHANNEL, ids]
                    squared = (prediction - target).square().cpu().numpy()
                    greedy_ids = ids.clone()
                    coarse = configs[arm]["model"].get("coarse_bins", 0)
                    extra: list[dict] = [{} for _ in deep]
                    if coarse:
                        fine = logits.shape[-1] // coarse
                        joint = logits.unflatten(-1, (coarse, fine))
                        marginal_logits = joint.logsumexp(-1)
                        coarse_ids = marginal_logits.argmax(-1)
                        conditional = joint - marginal_logits.unsqueeze(-1)
                        greedy_fine = conditional[
                            torch.arange(len(deep), device=device), coarse_ids
                        ].argmax(-1)
                        greedy_ids = coarse_ids * fine + greedy_fine
                        for j in range(len(deep)):
                            true_coarse, true_fine = (
                                int(true_ids[j]) // fine,
                                int(true_ids[j]) % fine,
                            )
                            map_coarse, map_fine = (
                                int(ids[j]) // fine,
                                int(ids[j]) % fine,
                            )
                            extra[j] = dict(
                                true_coarse=true_coarse,
                                true_fine=true_fine,
                                map_coarse=map_coarse,
                                map_fine=map_fine,
                                greedy_coarse=int(coarse_ids[j]),
                                greedy_fine=int(greedy_fine[j]),
                                true_coarse_probability=float(
                                    marginal_logits[j].softmax(-1)[true_coarse]
                                ),
                                map_coarse_probability=float(
                                    marginal_logits[j].softmax(-1)[map_coarse]
                                ),
                                greedy_coarse_probability=float(
                                    marginal_logits[j].softmax(-1)[coarse_ids[j]]
                                ),
                                true_fine_conditional_probability=float(
                                    conditional[j, true_coarse, true_fine].exp()
                                ),
                                map_fine_conditional_probability=float(
                                    conditional[j, map_coarse, map_fine].exp()
                                ),
                            )
                    probs = probability.cpu().numpy()
                    probability_arrays[arm].append(probs)
                    for j, i in enumerate(deep):
                        row, col = int(batch["row"][i]), int(batch["col"][i])
                        weight = float(
                            source.wet_area[DEPTH]
                            / (source.area.size * source.area[row, col])
                        )
                        true_id, pred_id, greedy_id = (
                            int(true_ids[j]),
                            int(ids[j]),
                            int(greedy_ids[j]),
                        )
                        scale, mean = (
                            float(source.std[CHANNEL]),
                            float(source.mean[CHANNEL]),
                        )
                        records.append(
                            dict(
                                arm=arm,
                                origin=int(origin),
                                origin_date=source.date_strings[origin],
                                target_date=source.date_strings[origin + 1],
                                row=row,
                                col=col,
                                lat=float(source.lat[row]),
                                lon=float(source.lon[col]),
                                previous_c=float(frame["input"][0, CHANNEL, row, col])
                                * scale
                                + mean,
                                current_c=float(batch["query_current"][i, 2]) * scale
                                + mean,
                                target_c=float(target[j]) * scale + mean,
                                prediction_c=float(prediction[j]) * scale + mean,
                                greedy_c=float(vocab.centers[CHANNEL, greedy_id])
                                * scale
                                + mean,
                                true_bin=true_id,
                                predicted_bin=pred_id,
                                greedy_bin=greedy_id,
                                true_bin_probability=float(probs[j, true_id]),
                                predicted_bin_probability=float(probs[j, pred_id]),
                                true_bin_rank=int(
                                    (probs[j] > probs[j, true_id]).sum() + 1
                                ),
                                full_grid_weight=weight,
                                squared_normalized_error=float(squared[j]),
                                weighted_squared_error=weight * float(squared[j]),
                                **extra[j],
                            )
                        )
            print(
                json.dumps(
                    dict(event="inference", origin=int(origin), cases=len(records))
                ),
                flush=True,
            )
    finally:
        source.close()
    checks = {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        assert len(rows) == 64
        actual = float(np.mean([r["weighted_squared_error"] for r in rows]))
        expected = next(
            r for r in receipts[arm]["per_channel"] if r["channel_index"] == CHANNEL
        )["full_grid_nmse"]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)
        checks[arm] = dict(
            cases=len(rows),
            reproduced_full_grid_nmse=actual,
            saved_full_grid_nmse=expected,
        )
    write_json(output / "validation-cases.json", records)
    write_json(
        output / "inference.json",
        dict(
            checks=checks,
            configuration=config,
            source=source.receipt(),
            device=torch.cuda.get_device_name(),
            batch_shape_preserved=True,
            precision="original BF16 autocast; logits and scoring FP32",
        ),
    )
    np.savez_compressed(
        output / "validation-probabilities.npz",
        **{arm: np.concatenate(arrays) for arm, arrays in probability_arrays.items()},
    )
    print("INFERENCE VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "infer"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    blosc.set_nthreads(1)
    (replay if args.mode == "replay" else infer)(args.training_root, args.output)
    write_json(args.output / "COMPLETE.json", dict(mode=args.mode, verified=True))


if __name__ == "__main__":
    main()
