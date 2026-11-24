"""Check full-size GPU pilot checkpoint and same-step resume before release."""

import argparse
from pathlib import Path

import torch
from common import OceanModel, Prepared, mse

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = Prepared(args.root / "training-data-archive")
for arm in ("fine_latent", "fine_latent_detach"):
    c = torch.load(
        args.root / "fine-pilot" / arm / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert c["updates"] == 1
    model = OceanModel(torch.tensor(data.mask, device="cuda"), arm=arm).cuda()
    model.load_state_dict(c["model"], strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=6e-4)
    opt.load_state_dict(c["optimizer"])
    x, boundaries, targets = data.window(data.meta["diagnostic_origins"][0], 4, "cuda")
    h = None
    loss = 0
    for b, target in zip(boundaries, targets, strict=True):
        pred, h = model(x, b, h)
        assert h.shape == (1, 32, 180, 360)
        loss = loss + mse(pred, target, model.wet) / 4
        x, h = model.feedback(pred, h)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fine_decoder.weight.grad.abs().sum() > 0
    opt.step()
    print(
        arm,
        "resumed update finite",
        float(loss.detach()),
        "parameters",
        sum(p.numel() for p in model.parameters()),
        "max_cuda_bytes",
        torch.cuda.max_memory_allocated(),
        flush=True,
    )
    del c, model, opt, x, h, loss, pred
    torch.cuda.empty_cache()
(args.root / "FINE_QUALIFIED").write_text(
    "Both full-size arms: optimizer update, checkpoint reload, resumed update, gradient tests passed.\n"
)
