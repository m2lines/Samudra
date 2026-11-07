# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native regional temperature fields around the all-fields validation spike."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from samudra.metrics.spectra import SPATIAL_REGIONS

from .baseline import file_sha256
from .ema import restore_ema
from .model import ModelConfig, PointTransformer
from .multitask import ComparisonSource
from .run import to_device


@torch.inference_mode()
def run(config, output):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    blosc.set_nthreads(1)
    output.mkdir(parents=True, exist_ok=True)
    source = ComparisonSource(
        config["source"], 4, config["reference_root"], arm="all_fields"
    )
    try:
        baseline = Path(config["baseline_cache"])
        manifest = json.loads((baseline / "manifest.json").read_text())
        for key in ("metadata_sha256", "mean", "std"):
            if manifest["source"][key] != source.receipt()[key]:
                raise ValueError(f"Baseline/source mismatch: {key}")
        origin = config["origin"]
        if origin not in source.origins["test"]:
            raise ValueError("Diagnostic origin must be held out")
        frame = source.read(origin)
        base = np.load(baseline / f"temperature-{origin:04d}.npy", mmap_mode="r")
        for selection in config["models"]:
            state = torch.load(
                selection["checkpoint"],
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            if state["step"] != selection["step"]:
                raise ValueError("Checkpoint step changed")
            model = PointTransformer(ModelConfig(**state["config"]["model"]))
            model.load_state_dict(state["model"])
            if selection["from_latest"]:
                if int(state["ema"]["num_updates"]) != state["step"]:
                    raise ValueError("EMA update mismatch")
                ema = restore_ema(
                    model, state["config"]["ema"], state["ema"], state["step"]
                )
                ema.copy_to(model)
                del ema
            elif state.get("weights_kind") != "ema":
                raise ValueError("Expected selected EMA checkpoint")
            source.arm = state["config"]["arm"]
            model.to(device).eval()
            del state
            if rank == 0:
                receipt = dict(
                    selection,
                    sha256=file_sha256(selection["checkpoint"]),
                    source=source.receipt(),
                    origin=origin,
                    input_date=str(source.date_strings[origin]),
                    target_date=str(source.date_strings[origin + 1]),
                    baseline_checkpoint_sha256=manifest["checkpoint_sha256"],
                    inference_commit=os.environ.get("SKYLINE_COMMIT"),
                )
                (output / f"{selection['name']}-receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
            for region_index, (name, lon, lat) in enumerate(SPATIAL_REGIONS):
                rows = np.flatnonzero(
                    (source.lat >= lat.start) & (source.lat < lat.stop)
                )
                cols = np.flatnonzero(
                    (source.lon >= lon.start) & (source.lon < lon.stop)
                )
                yy, xx = np.meshgrid(rows, cols, indexing="ij")
                for depth_index, z in enumerate(config["depths"]):
                    if (
                        region_index * len(config["depths"]) + depth_index
                    ) % world != rank:
                        continue
                    dest = output / f"{selection['name']}-r{region_index}-z{z:02d}.npz"
                    if dest.exists():
                        continue
                    mask = source.wet[z][yy, xx]
                    points = np.column_stack((yy[mask], xx[mask]))
                    result = np.full(mask.shape, np.nan, dtype=np.float32)
                    predictions = []
                    for start in range(0, len(points), config["microbatch"]):
                        pp = points[start : start + config["microbatch"]]
                        batch = to_device(
                            source.sample(
                                frame,
                                np.random.default_rng(0),
                                len(pp),
                                33,
                                np.full(len(pp), z),
                                pp,
                            ),
                            device,
                        )
                        inputs = {k: v for k, v in batch.items() if k != "target"}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = model(inputs)
                        column = 0 if source.arm == "temperature" else 2
                        predictions.append(pred[:, column].cpu().numpy())
                    result[mask] = (
                        np.concatenate(predictions) * source.std[38 + z]
                        + source.mean[38 + z]
                    )
                    if not np.isfinite(result[mask]).all():
                        raise FloatingPointError("Nonfinite wet predictions")
                    gold = (
                        frame["target"][38 + z][yy, xx] * source.std[38 + z]
                        + source.mean[38 + z]
                    )
                    samudra = base[z][yy, xx] * source.std[38 + z] + source.mean[38 + z]
                    partial = dest.with_suffix(".partial.npz")
                    np.savez_compressed(
                        partial,
                        prediction=result,
                        gold=gold,
                        samudra=samudra,
                        wet=mask,
                        lat=source.lat[rows],
                        lon=source.lon[cols],
                        area=source.area[yy, xx],
                        region=name,
                        depth=z,
                        origin=origin,
                    )
                    partial.replace(dest)
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                model=selection["name"],
                                region=name,
                                depth=z,
                                points=len(points),
                            )
                        ),
                        flush=True,
                    )
            del model
            torch.cuda.empty_cache()
        (output / f"rank-{rank}-COMPLETE.json").write_text(
            json.dumps(dict(rank=rank, world=world)) + "\n"
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
