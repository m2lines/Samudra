"""Common held-out evaluation of a completed training arm's selected EMA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import OceanModel, Prepared, metrics, mse, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--selection", choices=("selected", "epoch70"), default="selected")
    ap.add_argument(
        "--output",
        help="Separate evaluation directory; leaves training artifacts unchanged",
    )
    args = ap.parse_args()
    root = Path(args.run)
    if not (root / "COMPLETE").exists():
        raise RuntimeError("Refuse final evaluation of an incomplete training run")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Prepared(args.data)
    mask = torch.tensor(data.mask, device="cuda")
    model = OceanModel(mask, arm=args.arm).cuda().eval()
    output = Path(args.output) if args.output else root
    if args.selection == "epoch70" and args.output is None:
        raise ValueError("Epoch-70 evaluation requires a separate --output directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = root / ("best-ema.pt" if args.selection == "selected" else "latest.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.selection == "epoch70":
        assert saved["epoch"] == 70 and saved["next_group"] == 0
    model.load_state_dict(
        saved["model" if args.selection == "selected" else "ema"], strict=True
    )
    result = {
        "arm": args.arm,
        "selected_epoch": saved["epoch"],
        "checkpoint_sha256": sha256(checkpoint),
        "selection": args.selection,
        "state_key": "model" if args.selection == "selected" else "ema",
        "checkpoint_path": str(checkpoint),
        "records": [],
    }
    del saved  # latest.pt also contains optimizer and raw weights; don't retain them.
    std = torch.tensor(data.std, device="cuda")
    map_channels = [38, 51, 54, 57, 70, 76, 13, 32]
    from common import PROG

    map_meta = {
        "channels": [PROG[i] for i in map_channels],
        "channel_indices": map_channels,
        "latitude": data.meta["latitude"],
        "longitude": data.meta["longitude"],
        "mean": data.mean[map_channels].tolist(),
        "std": data.std[map_channels].tolist(),
        "origin": 0,
        "selection": args.selection,
    }
    (output / "map-metadata.json").write_text(json.dumps(map_meta, indent=2))
    modes = (
        ["persistent", "reset_every_call"] if "latent" in args.arm else ["persistent"]
    )
    for mode in modes:
        for oi, start in enumerate(data.meta["diagnostic_origins"]):
            x, bs, truth = data.window(start, 72, "cuda")
            h = None
            with torch.no_grad():
                for k in range(72):
                    if mode == "reset_every_call":
                        h = None
                    x, h = model(x, bs[k], h)
                    if k in (0, 2, 8, 35, 71):
                        result["records"].append(
                            {
                                "mode": mode,
                                "origin": oi,
                                "call": k + 1,
                                "mse": float(mse(x, truth[k], model.wet)),
                                **metrics(x, truth[k], mask, std),
                            }
                        )
                    if oi == 0 and mode == "persistent" and k in (0, 35, 71):
                        np.savez_compressed(
                            output / f"maps-call-{k + 1:02d}.npz",
                            prediction=x.reshape(2, 77, *x.shape[-2:])[:, map_channels]
                            .cpu()
                            .numpy(),
                            truth=truth[k]
                            .reshape(2, 77, *x.shape[-2:])[:, map_channels]
                            .cpu()
                            .numpy(),
                            mask=data.mask[map_channels],
                            times=np.array(
                                data.times[start + 2 * k + 2 : start + 2 * k + 4]
                            ),
                            initial_times=np.array(data.times[start : start + 2]),
                        )
                    if k == 0 and mode == "persistent":
                        np.savez_compressed(
                            output / f"heldout-first-{oi}.npz",
                            prediction=x.cpu().numpy(),
                            truth=truth[0].cpu().numpy(),
                        )
            print("Evaluated", args.arm, mode, oi, flush=True)
    (output / "heldout.json").write_text(json.dumps(result))
    (output / "EVAL_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
