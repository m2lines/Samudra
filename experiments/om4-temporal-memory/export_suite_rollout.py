"""Export physical rollouts from experiment checkpoints for the standard suite."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import OceanModel, Prepared, mse, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--selection", choices=("selected", "epoch70"), required=True)
    ap.add_argument("--origin", type=int, default=0)
    ap.add_argument("--max-outputs", type=int)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "EXPORT_COMPLETE").exists():
        return
    data = Prepared(args.root / "training-data-archive")
    start = data.meta["diagnostic_origins"][args.origin]
    length = len(data.times) - start - 2
    if args.max_outputs is not None:
        length = min(length, args.max_outputs)
    indices = data.meta["source_indices"][start : start + 2 + length]
    assert np.all(np.diff(indices) == 1)
    checkpoint = (
        args.root
        / "training"
        / args.arm
        / ("latest.pt" if args.selection == "epoch70" else "best-ema.pt")
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.selection == "epoch70":
        assert saved["epoch"] == 70 and saved["next_group"] == 0
    model = (
        OceanModel(torch.tensor(data.mask, device="cuda"), arm=args.arm).cuda().eval()
    )
    state_key = "ema" if args.selection == "epoch70" else "model"
    model.load_state_dict(saved[state_key], strict=True)
    meta = dict(
        arm=args.arm,
        selection=args.selection,
        epoch=saved["epoch"],
        checkpoint_sha256=sha256(checkpoint),
        state_key=state_key,
        origin=args.origin,
        start=start,
        times=data.times[start + 2 : start + 2 + length],
        input_times=data.times[start : start + 2],
        source_data=data.meta["source_attrs"],
        normalization=data.meta["normalization"],
    )
    del saved
    output = np.lib.format.open_memmap(
        args.output / "normalized-prediction.npy",
        mode="w+",
        dtype=np.float32,
        shape=(length, 77, *data.mask.shape[-2:]),
    )
    x = (
        torch.from_numpy(np.array(data.prog[start : start + 2]))
        .flatten(0, 1)[None]
        .cuda()
    )
    h = None
    scores = []
    with torch.no_grad():
        for offset in range(0, length, 2):
            boundary = (
                torch.from_numpy(
                    np.array(data.bound[start + offset : start + offset + 2])
                )
                .flatten(0, 1)[None]
                .cuda()
            )
            x, h = model(x, boundary, h)
            n = min(2, length - offset)
            output[offset : offset + n] = (
                x.reshape(2, 77, *x.shape[-2:])[:n].cpu().numpy()
            )
            if offset // 2 in (0, 2, 8, 35, 71) and n == 2:
                truth = (
                    torch.from_numpy(
                        np.array(data.prog[start + offset + 2 : start + offset + 4])
                    )
                    .flatten(0, 1)[None]
                    .cuda()
                )
                scores.append(
                    dict(call=offset // 2 + 1, mse=float(mse(x, truth, model.wet)))
                )
            if offset % 40 == 0:
                output.flush()
                print(
                    args.arm,
                    args.selection,
                    "exported",
                    offset + n,
                    "of",
                    length,
                    flush=True,
                )
    output.flush()
    previous = json.loads(
        (
            args.root
            / "checkpoint-comparison"
            / args.arm
            / args.selection
            / "heldout.json"
        ).read_text()
    )
    expected = {
        r["call"]: r["mse"]
        for r in previous["records"]
        if r["mode"] == "persistent" and r["origin"] == args.origin
    }
    for row in scores:
        assert row["mse"] == expected[row["call"]], (row, expected[row["call"]])
    meta["matched_existing_mse"] = scores
    meta["prediction_sha256"] = sha256(args.output / "normalized-prediction.npy")
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    (args.output / "EXPORT_COMPLETE").write_text(
        "Physical predictions verified against existing same-checkpoint scores.\n"
    )


if __name__ == "__main__":
    main()
