# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Trace validation outliers to actual training draws and local sensitivities."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS, OceanSource
from .model import ModelConfig, PointTransformer, Vocabulary
from .run import to_device

FEATURES = ["thetao_13", "so_13", "uo_13", "vo_13", "zos", "hfds"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay_samples(source, config, saved):
    """Replay accepted updates, excluding discarded updates before preemption."""
    rng = np.random.default_rng(config["seed"])
    order_rng = np.random.default_rng(config["seed"] + 100)
    order, cursor = order_rng.permutation(source.origins["train"]).tolist(), 0

    def pop():
        nonlocal order, cursor
        if cursor == len(order):
            order, cursor = order_rng.permutation(source.origins["train"]).tolist(), 0
        value = order[cursor]
        cursor += 1
        return value

    current, following, used = pop(), pop(), 0
    records = []
    for step in range(1, saved["step"] + 1):
        for _ in range(config["accumulation"]):
            if used >= config["samples_per_frame"]:
                current, following, used = following, pop(), 0
            depths = rng.integers(0, 19, config["microbatch"])
            uniforms = rng.random(config["microbatch"])
            for depth, u in zip(depths, uniforms, strict=True):
                if depth == 13:
                    index = np.searchsorted(
                        source.location_cdfs[depth], u, side="right"
                    )
                    row, col = divmod(int(source.locations[depth][index]), source.width)
                    records.append((step, current, row, col))
            used += config["microbatch"]
    assert rng.bit_generator.state == saved["numpy_rng"], "Point RNG replay differs"
    actual = dict(
        rng=order_rng.bit_generator.state,
        order=order,
        cursor=cursor,
        current_origin=current,
        next_origin=following,
        used=used,
    )
    assert actual == saved["stream"], "Frame-stream replay differs"
    return np.asarray(records, dtype=np.int64)


def inspect_training(config, output):
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(
        config["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    training = state["config"]
    source = OceanSource(
        training["source"], config.get("readers", 8), training["reference_root"]
    )
    cases = json.loads(Path(config["cases"]).read_text())
    records = replay_samples(source, training, state)
    vocab = Vocabulary(np.ones(19), training["model"]["bins"])
    vocab.load_state_dict(state["vocabulary"])
    del state
    n = len(records)
    features = np.full((n, 2, len(FEATURES)), np.nan, dtype=np.float32)
    labels = np.full(n, np.nan, dtype=np.float32)
    query = np.full((len(cases), 2, len(FEATURES)), np.nan, dtype=np.float32)
    # Same-cell histories are separate analogues, not necessarily training draws.
    origins = source.origins["train"]
    local_features = np.full(
        (len(origins), len(cases), 2, len(FEATURES)), np.nan, dtype=np.float32
    )
    local_labels = np.full((len(origins), len(cases)), np.nan, dtype=np.float32)
    plans: dict[int, list[tuple]] = defaultdict(list)
    for i, (_, origin, row, col) in enumerate(records):
        for history in range(2):
            plans[int(origin - 1 + history)].append(("sample", i, history, row, col))
        plans[int(origin + 1)].append(("label", i, 0, row, col))
    for i, origin in enumerate(origins):
        for j, case in enumerate(cases):
            for history in range(2):
                plans[int(origin - 1 + history)].append(
                    ("local", (i, j), history, case["row"], case["col"])
                )
            plans[int(origin + 1)].append(
                ("local_label", (i, j), 0, case["row"], case["col"])
            )
    for i, case in enumerate(cases):
        for history in range(2):
            plans[case["origin"] - 1 + history].append(
                ("query", i, history, case["row"], case["col"])
            )

    # Each compressed full-grid plane is decoded once. Only point features survive.
    def extract(item):
        date, requests = item
        rows = np.asarray([r[3] for r in requests])
        cols = np.asarray([r[4] for r in requests])
        values = np.stack(
            [np.asarray(source.zarr[name][date])[rows, cols] for name in FEATURES], -1
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite wet feature at date {date}")
        return requests, values

    ordered = sorted(plans.items())
    # Bound in-flight results; Executor.map eagerly submits all items but results
    # here contain only selected points, never global planes.
    for done, (requests, values) in enumerate(source.pool.map(extract, ordered), 1):
        for (kind, index, history, _, _), value in zip(requests, values, strict=True):
            if kind == "sample":
                features[index, history] = value
            elif kind == "label":
                labels[index] = value[0]
            elif kind == "local":
                local_features[index[0], index[1], history] = value
            elif kind == "local_label":
                local_labels[index] = value[0]
            else:
                query[index, history] = value
        if done % 250 == 0:
            print(
                json.dumps(dict(planes_dates_done=done, total=len(ordered))), flush=True
            )
    for array in (features, labels, local_features, local_labels, query):
        assert np.isfinite(array).all()
    means = source.mean[[CHANNELS.index(name) for name in FEATURES]]
    stds = source.std[[CHANNELS.index(name) for name in FEATURES]]
    normalized = ((features - means) / stds).reshape(n, -1)
    normalized_query = ((query - means) / stds).reshape(len(cases), -1)
    centers = vocab.centers[13].numpy() * source.std[51] + source.mean[51]
    ids = vocab.encode(
        torch.from_numpy((labels - source.mean[51]) / source.std[51]),
        torch.full((n,), 13),
    ).numpy()
    counts = np.bincount(ids, minlength=len(centers))
    results = []
    for j, case in enumerate(cases):
        distance = np.sqrt(np.mean((normalized - normalized_query[j]) ** 2, axis=1))
        predicted, truth = case["prediction_c"], case["truth_c"]
        matching = np.abs(labels - predicted) <= 0.25
        truthful = np.abs(labels - truth) <= 0.25

        def neighbors(eligible):
            indices = np.flatnonzero(eligible)
            selected = indices[np.argsort(distance[indices], kind="stable")[:10]]
            return [
                dict(
                    step=int(records[i, 0]),
                    origin=int(records[i, 1]),
                    date=source.date_strings[records[i, 1]],
                    row=int(records[i, 2]),
                    col=int(records[i, 3]),
                    lat=float(source.lat[records[i, 2]]),
                    lon=float(source.lon[records[i, 3]]),
                    distance=float(distance[i]),
                    target_c=float(labels[i]),
                    current_c=float(features[i, 1, 0]),
                    features=features[i].tolist(),
                )
                for i in selected
            ]

        local_distance = np.sqrt(
            np.mean(
                (
                    ((local_features[:, j] - means) / stds).reshape(len(origins), -1)
                    - normalized_query[j]
                )
                ** 2,
                axis=1,
            )
        )
        nearest_local = np.argsort(local_distance, kind="stable")[:5]
        true_bin = int(
            vocab.encode(
                torch.tensor([(truth - source.mean[51]) / source.std[51]]),
                torch.tensor([13]),
            )[0]
        )
        pred_bin = int(np.argmin(abs(centers - predicted)))
        results.append(
            dict(
                **case,
                lat=float(source.lat[case["row"]]),
                lon=float(source.lon[case["col"]]),
                origin_date=source.date_strings[case["origin"]],
                target_date=source.date_strings[case["origin"] + 1],
                current_c=float(query[j, 1, 0]),
                previous_c=float(query[j, 0, 0]),
                true_bin=true_bin,
                predicted_bin=pred_bin,
                true_bin_training_count=int(counts[true_bin]),
                predicted_bin_training_count=int(counts[pred_bin]),
                near_truth_training_count=int(truthful.sum()),
                near_prediction_training_count=int(matching.sum()),
                neighbors=neighbors(np.ones(n, dtype=bool)),
                confusable_neighbors=neighbors(matching),
                true_temperature_neighbors=neighbors(truthful),
                same_cell_training_range=[
                    float(local_labels[:, j].min()),
                    float(local_labels[:, j].max()),
                ],
                same_cell_analogues=[
                    dict(
                        origin=int(origins[i]),
                        date=source.date_strings[origins[i]],
                        distance=float(local_distance[i]),
                        target_c=float(local_labels[i, j]),
                    )
                    for i in nearest_local
                ],
            )
        )
    np.savez_compressed(
        output / "training-points.npz",
        records=records,
        features=features,
        labels=labels,
        query_features=query,
        feature_mean=means,
        feature_std=stds,
        bin_centers_c=centers,
        bin_counts=counts,
        local_origins=origins,
        local_features=local_features,
        local_labels=local_labels,
    )
    write_json(
        output / "audit.json",
        dict(
            checkpoint=config["checkpoint"],
            step=training["steps"],
            sampler_replay_matches_checkpoint=True,
            training_draws_at_depth=n,
            unique_training_date_cells=len(set(map(tuple, records[:, 1:]))),
            features=FEATURES,
            metric="RMS difference of 12 training-standardized center values, two history times; no geography or target in distance",
            confusable_filter="training target within 0.25 C of classifier prediction; diagnostic label filter only",
            target_temperature_quantiles=np.quantile(
                labels, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
            ).tolist(),
            warm_counts={str(t): int((labels > t).sum()) for t in (5, 8, 10, 15, 20)},
            cases=results,
        ),
    )
    source.close()
    print(
        json.dumps(dict(event="audit_complete", samples=n, output=str(output))),
        flush=True,
    )


def explain(config, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(config["audit"]).read_text())
    state = torch.load(
        config["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    training = state["config"]
    source = OceanSource(
        training["source"], config.get("readers", 8), training["reference_root"]
    )
    device = torch.device("cuda")
    model = PointTransformer(ModelConfig(**training["model"])).to(device).eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    vocab = Vocabulary(np.ones(19), training["model"]["bins"]).to(device)
    vocab.load_state_dict(state["vocabulary"])
    del state
    centers = vocab.centers[13].cpu().numpy() * source.std[51] + source.mean[51]
    results, gradients = [], {}
    with np.load(Path(config["audit"]).parent / "training-points.npz") as points:
        bin_counts = points["bin_counts"]
    for j, case in enumerate(audit["cases"]):
        frame = source.read(case["origin"])
        batch = to_device(
            source.sample(
                frame,
                np.random.default_rng(0),
                1,
                training["crop"],
                [13],
                [(case["row"], case["col"])],
            ),
            device,
        )
        for key in ("values", "current", "geometry"):
            batch[key].requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        probabilities = logits.softmax(-1)[0].detach().cpu().numpy()
        pred_bin, true_bin = case["predicted_bin"], case["true_bin"]
        margin = logits[0, pred_bin] - logits[0, true_bin]
        values_tensor, current_grad, geometry_tensor = torch.autograd.grad(
            margin, [batch["values"], batch["current"], batch["geometry"]]
        )
        channel_wet = torch.cat([batch["wet"]] * 4 + [batch["wet"][..., :1]] * 4, -1)
        values_grad = (
            (values_tensor * channel_wet)[0]
            .float()
            .cpu()
            .numpy()
            .reshape(2, training["crop"], training["crop"], 80)
        )
        geometry_grad = (
            geometry_tensor[0]
            .float()
            .cpu()
            .numpy()
            .reshape(2, training["crop"], training["crop"], 6)
        )
        gradients[f"case_{j}_values"] = values_grad
        gradients[f"case_{j}_geometry"] = geometry_grad
        del frame
        top = np.argsort(probabilities)[-10:][::-1]
        field_gradient = np.abs(values_grad).sum(axis=(0, 1, 2))
        top_fields = np.argsort(field_gradient)[-10:][::-1]
        neighbor_scores = []
        # Actual nearest draws, nearest misleading-label draws, and same-cell
        # analogues; forward their full original inputs, not the 12-D summaries.
        choices = [
            (kind, r)
            for kind in ("neighbors", "confusable_neighbors")
            for r in case[kind][:2]
        ]
        choices += [
            ("same_cell_analogue", dict(case, **r))
            for r in case["same_cell_analogues"][:1]
        ]
        for kind, r in choices:
            neighbor_frame = source.read(r["origin"])
            nb = to_device(
                source.sample(
                    neighbor_frame,
                    np.random.default_rng(0),
                    1,
                    training["crop"],
                    [13],
                    [(r["row"], r["col"])],
                ),
                device,
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                nl = model(nb)
            npred = int(nl.argmax(-1)[0])
            neighbor_scores.append(
                dict(
                    kind=kind,
                    origin=r["origin"],
                    row=r["row"],
                    col=r["col"],
                    target_c=r["target_c"],
                    prediction_c=float(centers[npred]),
                    probability=float(nl.softmax(-1)[0, npred]),
                )
            )
            del neighbor_frame, nb, nl
        results.append(
            dict(
                origin=case["origin"],
                row=case["row"],
                col=case["col"],
                truth_c=case["truth_c"],
                saved_prediction_c=case["prediction_c"],
                reproduced_prediction_c=float(centers[probabilities.argmax()]),
                mean_c=float(probabilities @ centers),
                median_c=float(centers[np.searchsorted(probabilities.cumsum(), 0.5)]),
                truth_bin_rank=int((probabilities > probabilities[true_bin]).sum() + 1),
                truth_probability=float(probabilities[true_bin]),
                predicted_probability=float(probabilities[pred_bin]),
                mass_within_half_degree_of_truth=float(
                    probabilities[abs(centers - case["truth_c"]) <= 0.5].sum()
                ),
                top_bins=[
                    dict(
                        bin=int(i),
                        temperature_c=float(centers[i]),
                        probability=float(probabilities[i]),
                        training_count=int(bin_counts[i]),
                    )
                    for i in top
                ],
                logit_margin=float(margin),
                current_query_gradient=float(current_grad[0]),
                top_input_gradient_channels=[
                    dict(
                        channel=CHANNELS[i], absolute_gradient=float(field_gradient[i])
                    )
                    for i in top_fields
                ],
                neighbors=neighbor_scores,
            )
        )
        gradients[f"case_{j}_probabilities"] = probabilities
        write_json(
            output / "explanations.json",
            dict(
                gradient_definition="Local derivative of saved-predicted-bin logit minus true-bin logit; values in training-normalized units. Sensitivity, not causal training-example influence.",
                cases=results,
            ),
        )
        print(
            json.dumps(
                dict(
                    case=j,
                    prediction=results[-1]["reproduced_prediction_c"],
                    top_probability=float(probabilities.max()),
                )
            ),
            flush=True,
        )
    np.savez_compressed(
        output / "gradients-and-probabilities.npz", **gradients, centers_c=centers
    )
    write_json(
        output / "COMPLETE.json",
        dict(cases=len(results), checkpoint=config["checkpoint"]),
    )
    source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "explain"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config.get("torch_threads", 4))
    (inspect_training if args.mode == "audit" else explain)(config, args.output)


if __name__ == "__main__":
    main()
