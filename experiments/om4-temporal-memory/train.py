"""Four exposure-matched 1-degree arms with fixed normalized MSE."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from common import ARMS, OceanModel, Prepared, metrics, mse
from torch.utils.data import DataLoader, Dataset


class Windows(Dataset):
    def __init__(self, root, indices):
        self.data = Prepared(root)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        start = int(self.indices[i])
        return (
            np.array(self.data.prog[start : start + 10]),
            np.array(self.data.bound[start : start + 10]),
        )


def save_atomic(state, path):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-seconds", type=int, default=19000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    assert args.batch_size % args.microbatch == 0
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists():
        return
    data = Prepared(args.data)
    if data.meta["diagnostic_only"] and not args.smoke:
        raise ValueError("Full training requires the complete chronological source")
    device = torch.device("cuda")
    mask = torch.tensor(data.mask, device=device)
    model = OceanModel(mask, arm=args.arm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epoch = 0
    next_group = 0
    updates = 0
    best = float("inf")
    ckpt_path = output / "latest.pt"
    run_id = f"om4mem-{args.arm}-{args.seed}-20261016"
    if ckpt_path.exists():
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        optimizer.load_state_dict(c["optimizer"])
        ema = c["ema"]
        epoch = c["epoch"]
        next_group = c["next_group"]
        updates = c["updates"]
        best = c["best"]
        del c
    config = vars(args) | {
        "lr": 6e-4,
        "loss": "fixed normalized wet-cell MSE, equal channel weight",
        "calls": 4,
        "input_steps": 2,
        "output_steps": 2,
        "precision": "fp32",
        "normalization": "batch",
        "latent_channels": 16 if args.arm.startswith("latent") else 0,
        "checkpoint_batchnorm_replay": "running stats restored after replay",
        "train_dates": ["1975-01-03", "2013-10-05"],
        "validation_dates": ["2013-10-05", "2014-10-05"],
        "initialization": "paired seed; identical physical parameters",
        "ema_decay": 0.999,
        "ema_faster_start": True,
        "selection": "minimum normalized 360-day validation MSE from 2013-10-05 origin",
        "source_data": data.meta["source_attrs"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    run = wandb.init(
        entity="ocean_emulators",
        project="default",
        group="om4-temporal-memory-20261016",
        id=run_id,
        name=run_id,
        config=config,
        resume="allow",
        mode="disabled" if args.smoke else "online",
    )
    print("WANDB_URL", run.url, flush=True)
    if not args.smoke and not run.url:
        raise RuntimeError("Online W&B initialization required")
    starts = np.array(
        [
            i
            for i, t in enumerate(data.times[:-9])
            if t[:10] >= "1975-01-03" and data.times[i + 9][:10] < "2013-10-05"
        ]
    )
    if args.smoke:
        starts = np.array(
            [
                i
                for i in range(len(data.times) - 9)
                if np.all(np.diff(data.meta["source_indices"][i : i + 10]) == 1)
            ]
        )[:32]
    if not len(starts):
        raise RuntimeError("No training windows")
    terminate = [False]
    signal.signal(signal.SIGUSR1, lambda *_: terminate.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: terminate.__setitem__(0, True))
    begin = time.monotonic()

    def save(ep, group):
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema,
                "epoch": ep,
                "next_group": group,
                "updates": updates,
                "best": best,
                "config": config,
            },
            ckpt_path,
        )

    while epoch < args.epochs:
        order = np.random.default_rng(args.seed + epoch).permutation(starts)
        groups = len(order) // args.batch_size
        # All arms see the same complete effective batches, in the same order.
        selected = order[next_group * args.batch_size : groups * args.batch_size]
        loader = DataLoader(
            Windows(args.data, selected),
            batch_size=args.microbatch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        model.train()
        lr = 6e-4 * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        accum = args.batch_size // args.microbatch
        group = next_group
        for mi, (p, b) in enumerate(loader):
            p = p.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            x = p[:, :2].flatten(1, 2)
            h = None
            loss = 0.0
            for k in range(4):
                pred, h = model(x, b[:, 2 * k : 2 * k + 2].flatten(1, 2), h)
                loss = (
                    loss
                    + mse(pred, p[:, 2 * k + 2 : 2 * k + 4].flatten(1, 2), model.wet)
                    / 4
                )
                x, h = model.feedback(pred, h)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / accum).backward()
            total += float(loss.detach()) / accum
            if (mi + 1) % accum:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            group += 1
            decay = min(0.999, (1 + updates) / (10 + updates))
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if value.is_floating_point():
                        ema[key].lerp_(value, 1 - decay)
                    else:
                        ema[key].copy_(value)
            run.log(
                {"train/mse": total, "epoch": epoch + 1, "updates": updates, "lr": lr},
                step=updates,
            )
            print(
                f"Training Epoch: [{epoch + 1}] [{group}/{groups}] mse={total:.7g}",
                flush=True,
            )
            total = 0.0
            if updates % 20 == 0:
                save(epoch, group)
            if (
                terminate[0]
                or time.monotonic() - begin > args.max_seconds
                or args.smoke
            ):
                save(epoch, group)
                (output / "NEEDS_RESUME").write_text(
                    f"epoch={epoch} next_group={group}\n"
                )
                run.finish()
                return
        del loader
        # Deterministic validation starts stay entirely within the held-out year.
        model.eval()
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        validation = []
        origins = [next(i for i, t in enumerate(data.times) if t[:10] >= "2013-10-05")]
        assert data.times[origins[0] + 73][:10] <= "2014-10-05"
        with torch.no_grad():
            for start in origins:
                x, bs, targets = data.window(start, 36, device)
                h = None
                for k in range(36):
                    x, h = model(x, bs[k], h)
                    if k in (0, 2, 8, 35):
                        validation.append(
                            {
                                "origin": start,
                                "call": k + 1,
                                "mse": float(mse(x, targets[k], model.wet)),
                                **metrics(
                                    x,
                                    targets[k],
                                    mask,
                                    torch.tensor(data.std, device=device),
                                ),
                            }
                        )
        score = float(np.mean([r["mse"] for r in validation if r["call"] == 36]))
        (output / f"validation-{epoch + 1:03d}.json").write_text(json.dumps(validation))
        run.log({"validation/360day_mse": score, "epoch": epoch + 1}, step=updates)
        if score < best:
            best = score
            save_atomic(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "score": score,
                    "config": config,
                },
                output / "best-ema.pt",
            )
        model.load_state_dict(raw)
        del raw
        epoch += 1
        next_group = 0
        save(epoch, 0)
        (output / "NEEDS_RESUME").unlink(missing_ok=True)
    (output / "COMPLETE").write_text("complete\n")
    run.finish()


if __name__ == "__main__":
    main()
