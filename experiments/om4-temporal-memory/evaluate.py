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
    saved = torch.load(root / "best-ema.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    result = {
        "arm": args.arm,
        "selected_epoch": saved["epoch"],
        "checkpoint_sha256": sha256(root / "best-ema.pt"),
        "records": [],
    }
    std = torch.tensor(data.std, device="cuda")
    modes = (
        ["persistent", "reset_every_call"]
        if args.arm.startswith("latent")
        else ["persistent"]
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
                    if k == 0 and mode == "persistent":
                        np.savez_compressed(
                            root / f"heldout-first-{oi}.npz",
                            prediction=x.cpu().numpy(),
                            truth=truth[0].cpu().numpy(),
                        )
            print("Evaluated", args.arm, mode, oi, flush=True)
    (root / "heldout.json").write_text(json.dumps(result))
    (root / "EVAL_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
