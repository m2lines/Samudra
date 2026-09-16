"""Resolve a checkpoint to one whose ``model`` weights are the EMA weights.

Checkpoints saved for resuming training (``ckpt.pt``, ``ckpt_<epoch>.pt``,
``best_validation_ckpt.pt``) carry the EMA shadow parameters under
``checkpoint["ema"]["ema_params"]`` while ``checkpoint["model"]`` holds the raw
weights. ``ema_ckpt.pt`` already stores the EMA weights in ``model`` -- but it
is rewritten every epoch, so only the most recent epoch is available that way.

``eval.py`` only ever reads ``checkpoint["model"]``, so to evaluate the EMA
weights of any other epoch the parameters have to be swapped into that slot.

Only parameters that require gradients are tracked by ``EMATracker``; every
other entry (buffers, frozen tensors) is carried over from the raw weights,
which mirrors what ``EMATracker.copy_to`` does at validation time.

    python scripts/materialize_ema_checkpoint.py IN.pt OUT.pt
    python scripts/materialize_ema_checkpoint.py --resolve IN.pt

``--resolve`` prints (on stdout, alone) the path of a checkpoint holding EMA
weights, converting into a cached ``<name>_ema.pt`` sibling only when needed.
It is idempotent: an up-to-date cached file is reused. Everything else goes to
stderr so the output can be captured directly by a job script.
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch


def ema_name(name: str) -> str:
    """Mirror ``EMATracker._get_ema_name``."""
    return name.removeprefix("module.").replace(".", "")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def claims_ema(ckpt) -> bool:
    """True if ``ckpt`` is shaped like an EMA checkpoint.

    ``save_checkpoint(for_inference=True)`` writes the EMA weights into
    ``model`` and deliberately omits ``ema_params``, so a checkpoint carrying
    an ``ema`` block without them claims to be an EMA checkpoint.

    The claim is not proof. Before the aliasing fix, ``ema_ckpt.pt`` was written
    with exactly this shape but held the raw weights, because ``state_dict()``
    aliased live parameters that ``_ema_context`` restored before the save. Such
    a file is structurally identical to a good one, so ``verify_ema_weights``
    settles it by comparison against a sibling.
    """
    if ckpt.get("weights_are_ema"):
        return True
    return "ema" in ckpt and "ema_params" not in ckpt["ema"]


def verify_ema_weights(src: str, ckpt) -> bool:
    """Check an EMA-shaped checkpoint against the raw checkpoint beside it.

    Returns True if the weights really are the EMA weights. A pre-fix
    ``ema_ckpt.pt`` is bitwise identical to its sibling ``ckpt.pt`` at the same
    epoch; a correct one differs.
    """
    if ckpt.get("weights_are_ema"):
        return True  # written by this script, so already verified once

    sibling = os.path.join(os.path.dirname(src), "ckpt.pt")
    if not os.path.exists(sibling):
        log(
            f"WARNING: cannot verify {os.path.basename(src)} -- no ckpt.pt beside it "
            "to compare against. If this run predates the ema_ckpt aliasing fix, "
            "these are the raw weights."
        )
        return True

    raw = torch.load(sibling, map_location="cpu", weights_only=False)
    if raw.get("epoch") != ckpt.get("epoch"):
        log(
            f"WARNING: cannot verify {os.path.basename(src)} -- ckpt.pt is at epoch "
            f"{raw.get('epoch')}, not {ckpt.get('epoch')}."
        )
        return True

    model, raw_model = ckpt["model"], raw["model"]
    for key, value in model.items():
        other = raw_model.get(key)
        if other is None or not value.dtype.is_floating_point:
            continue
        if not torch.equal(value, other):
            return True  # differs from raw, so genuinely the EMA weights
    return False


def convert(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if claims_ema(ckpt) and verify_ema_weights(src, ckpt):
        raise SystemExit(
            f"{src} already holds EMA weights in ['model']; copy it instead of converting."
        )
    ema_params = ckpt.get("ema", {}).get("ema_params")
    if not ema_params:
        raise SystemExit(
            f"{src} has no ema.ema_params, so its EMA weights are unrecoverable. "
            "Checkpoints written before the ema_ckpt aliasing fix, or by a run "
            "without EMA, cannot be converted."
        )

    model = ckpt["model"]
    swapped = OrderedDict()
    n_ema = 0
    for key, value in model.items():
        shadow = ema_params.get(ema_name(key))
        if shadow is None:
            swapped[key] = value
            continue
        if tuple(shadow.shape) != tuple(value.shape):
            raise SystemExit(f"shape mismatch for {key}: {shadow.shape} vs {value.shape}")
        swapped[key] = shadow.detach().clone().to(value.dtype)
        n_ema += 1
    if hasattr(model, "_metadata"):
        swapped._metadata = model._metadata

    ckpt["model"] = swapped
    ckpt["ema"] = {k: v for k, v in ckpt["ema"].items() if k != "ema_params"}
    ckpt["weights_are_ema"] = True

    # Write via a temporary file in the destination directory so an interrupted
    # run cannot leave a half-written checkpoint that a later --resolve trusts.
    tmp = f"{dst}.tmp{os.getpid()}"
    torch.save(ckpt, tmp)
    os.replace(tmp, dst)

    log(
        f"epoch {ckpt.get('epoch')}, ema num_updates "
        f"{int(ckpt['ema']['num_updates'])}, decay {float(ckpt['ema']['decay'])}"
    )
    log(f"replaced {n_ema}/{len(model)} tensors with EMA weights -> {dst}")


def resolve(src: str) -> str:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if claims_ema(ckpt):
        if verify_ema_weights(src, ckpt):
            log(f"{os.path.basename(src)} already holds EMA weights; using it directly.")
            return src
        # Pre-fix ema_ckpt.pt: raw weights wearing an EMA label. The real EMA
        # parameters survive in the sibling checkpoint, so convert from that.
        sibling = os.path.join(os.path.dirname(src), "ckpt.pt")
        log(
            f"{os.path.basename(src)} is bitwise identical to ckpt.pt -- it holds RAW "
            "weights (written before the ema_ckpt aliasing fix). Converting from "
            "ckpt.pt instead."
        )
        src = sibling
        ckpt = torch.load(src, map_location="cpu", weights_only=False)

    base, ext = os.path.splitext(src)
    dst = f"{base}_ema{ext}"
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        log(f"reusing cached EMA checkpoint {os.path.basename(dst)}")
        return dst

    log(f"materializing EMA weights from {os.path.basename(src)}")
    convert(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", metavar="SRC", help="print a path to EMA weights")
    parser.add_argument("src", nargs="?", help="training checkpoint containing ema.ema_params")
    parser.add_argument("dst", nargs="?", help="path to write the EMA-weighted checkpoint to")
    args = parser.parse_args()

    if args.resolve:
        if args.src or args.dst:
            parser.error("--resolve takes no positional arguments")
        print(resolve(args.resolve))
    elif args.src and args.dst:
        convert(args.src, args.dst)
    else:
        parser.error("give SRC and DST, or --resolve SRC")


if __name__ == "__main__":
    main()
