#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Cross input source with decoder support on fixed pre-cooldown latent states."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_annual import history_fields
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_annual import model_batch
from samudra.experiments.observation_annual import inputs, read_origin
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def summary(prediction, reference, mask):
    valid = mask & np.isfinite(prediction) & np.isfinite(reference)
    a, b = prediction[valid].astype(float), reference[valid].astype(float)
    neighbors = valid & np.roll(valid, 1, -1)
    adjacent = prediction[neighbors] - np.roll(prediction, 1, -1)[neighbors]
    return dict(
        rmse=float(np.mean((a - b) ** 2) ** 0.5),
        correlation=float(np.corrcoef(a, b)[0, 1]),
        spatial_std=float(a.std()),
        truth_spatial_std=float(b.std()),
        adjacent_difference_rms_over_sqrt2=float(np.mean(adjacent**2) ** 0.5 / 2**0.5),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.export_root
    config = json.loads((root / "paths.json").read_text())
    audit = json.loads((root / "AUDITED.json").read_text())
    assert audit["config_sha256"] == digest(root / "paths.json")
    for path, expected in audit["contracts"].items():
        assert digest(path) == expected, path
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"], "cuda", surface_fill="zero", global_observations=True
    )
    saved = torch.load(
        config["model"]["checkpoint"], map_location="cpu", weights_only=False
    )
    assert saved["step"] == 2000
    assert saved["counts"] == dict(obs=1000, om4=1000, cm4=0)
    model = MonthlyLatentOcean().cuda().eval()
    model.load_state_dict(saved["ema"], strict=True)
    supports = {
        "full77": data.mask.bool(),
        "surface2": torch.zeros_like(data.mask, dtype=torch.bool),
    }
    supports["surface2"][[38, 76]] = data.mask[[38, 76]].bool()
    arrays, rows = {}, []
    source_inputs = {}
    for source in ("obs", "om4"):
        origin = "2015-01-01"
        if source == "obs":
            description, raw = read_origin(Path(config["annual"]) / "test" / origin)
            surface, atmosphere, contexts, _, valid = inputs(data, raw)
            fields, available = history_fields(surface, valid)
            prefix = data.load(data.root / "test" / "2015-01.npz")["atmosphere"].shape[
                1
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forcing = torch.cat(
                    [
                        model.adapt(atmosphere[:, :prefix]),
                        model.adapt(atmosphere[:, prefix:]),
                    ],
                    1,
                )
            gold = [raw["surface"][18], raw["surface"][19]]
            case_index = 0
        else:
            with np.load(root / "prepared" / (origin + ".npz")) as z:
                raw = dict(z)
            batch = model_batch(data, raw)
            fields, available, forcing, contexts = (
                batch[k] for k in ("fields", "available", "forcing", "contexts")
            )
            initial = data.physical(fields[:, 18:19])[0, 0, [38, 76]].cpu().numpy()
            gold = [initial, raw["reference_surface"][0, :2]]
            case_index = 3
        source_inputs[source] = dict(
            visible_normalized_rms=float(
                fields[:, :19][available[:, :19].expand_as(fields[:, :19])]
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            forcing_channel_rms=forcing[:, :19]
            .float()
            .square()
            .mean((0, 1, 3, 4))
            .sqrt()
            .cpu()
            .tolist(),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = model.initialize(
                fields[:, :19], available[:, :19], forcing[:, :19], contexts[:, 18]
            )
            future = model.processor(state, forcing[:, 19], contexts[:, 19])
        for label, latent, reference, position, seed in [
            ("initial", state, gold[0], -2.5, 905001),
            ("day5", future, gold[1], 2.5, 904001),
        ]:
            for support_name, support in supports.items():
                for steps in (
                    [32, 128] if support_name == "full77" and label == "day5" else [32]
                ):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        draws = model.decoder.sample(
                            latent.expand(8, -1, -1, -1, -1),
                            support,
                            torch.Generator(device="cuda").manual_seed(seed),
                            steps=steps,
                            position_days=position,
                        )
                    physical = (
                        data.physical(draws[:, None])[:, 0, [38, 76]].cpu().numpy()
                    )
                    mean_normalized = draws.mean(0, keepdim=True)
                    mean = (
                        data.physical(mean_normalized[:, None])[0, 0, [38, 76]]
                        .cpu()
                        .numpy()
                    )
                    key = f"{source}-{label}-{support_name}-{steps}"
                    arrays[key + "-mean"] = mean
                    arrays[key + "-member0"] = physical[0]
                    arrays[key + "-member_std"] = physical.std(0, ddof=1)
                    # Repeat the actual saved path before interpreting interventions.
                    repeat = steps == 32 and (
                        (label == "initial" and support_name == "full77")
                        or (
                            label == "day5"
                            and support_name
                            == ("full77" if source == "om4" else "surface2")
                        )
                    )
                    if repeat:
                        with np.load(
                            root / f"case-{case_index}" / "forecast.npz"
                        ) as original:
                            expected = (
                                original["initial_surface"]
                                if label == "initial"
                                else original["surface"][0]
                            )
                            np.testing.assert_array_equal(mean, expected)
                    for readout, value in [("mean8", mean), ("member0", physical[0])]:
                        for channel, field in enumerate(["SST", "SSH"]):
                            rows.append(
                                dict(
                                    source=source,
                                    state=label,
                                    support=support_name,
                                    sampler_steps=steps,
                                    readout=readout,
                                    field=field,
                                    latent_rms=float(
                                        latent.float().square().mean().sqrt()
                                    ),
                                    **summary(
                                        value[channel],
                                        reference[channel],
                                        data.grid["mask"][[38, 76][channel]],
                                    ),
                                )
                            )
                    print(json.dumps(dict(finished=key)), flush=True)
    path = args.output / "readouts.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path) as check:
        for key, value in arrays.items():
            np.testing.assert_array_equal(check[key], value)
    atomic_json(
        dict(
            producer=os.environ["SAMUDRA_CODE_COMMIT"],
            export_producer=config["producer"],
            checkpoint_sha256=audit["contracts"][config["model"]["checkpoint"]],
            script_sha256=digest(Path(__file__)),
            source_audit_sha256=digest(root / "AUDITED.json"),
            source_inputs=source_inputs,
            same_latents_across_decoder_supports=True,
            same_initial_noise_seed=True,
            original_readouts_exact=True,
            summaries=rows,
            files={path.name: digest(path)},
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
