# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Controlled isotropic-field experiment with the production diffusion backbone."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    denoising_loss,
    sample_joint,
)

torch.set_num_threads(1)
out = Path(os.environ["TOY_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
device = "cuda"
h, w = 44, 88
ky = torch.fft.fftfreq(h, device=device)[:, None]
kx = torch.fft.fftfreq(w, device=device)[None, :]
filt = torch.exp(-0.5 * (2 * torch.pi) ** 2 * (kx * kx + ky * ky) * 2**2)
filt = filt / torch.sqrt(filt.square().mean())
mask = torch.ones(2, h, w, device=device)


def target(n, g):
    x = torch.randn(n, 2, h, w, device=device, generator=g)
    return torch.fft.ifft2(torch.fft.fft2(x) * filt).real


def stats(x):
    x = x.float()[:, :, 8:-8, 8:-8].cpu().numpy()
    x = x - x.mean((-2, -1), keepdims=True)
    a = x[:, :, :-1, :-1].ravel()
    b = x[:, :, 1:, 1:].ravel()
    c = x[:, :, :-1, 1:].ravel()
    d = x[:, :, 1:, :-1].ravel()
    return dict(
        sd=float(x.std()),
        diagonal_plus=float(np.corrcoef(a, b)[0, 1]),
        diagonal_minus=float(np.corrcoef(c, d)[0, 1]),
    )


class Exact(torch.nn.Module):
    fields = 2

    def forward(self, x, sigma, latent, mask):
        transfer = filt.square()[None, None] / (
            filt.square()[None, None] + sigma[:, None, None, None].square()
        )
        return torch.fft.ifft2(torch.fft.fft2(x) * transfer).real


records = []
for seed in (1729, 1730):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    model = JointInteriorDecoder(2, 2, width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(2001):
        if step in (0, 500, 2000):
            model.eval()
            with torch.no_grad():
                for steps in (16, 64):
                    evg = torch.Generator(device=device).manual_seed(4041729)
                    x = sample_joint(
                        model,
                        torch.zeros(16, 2, h, w, device=device),
                        mask,
                        evg,
                        steps=steps,
                    )
                    r = dict(seed=seed, step=step, sampling_steps=steps, **stats(x))
                    records.append(r)
                    print(json.dumps(r), flush=True)
                    np.savez_compressed(
                        out / f"toy-{seed}-{step}-{steps}.npz",
                        samples=x[:4].cpu().numpy(),
                    )
            model.train()
        if step == 2000:
            break
        t = target(8, g)
        loss = denoising_loss(model, torch.zeros_like(t), t, mask, mask, g)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for label in ("truth", "exact16", "exact64"):
        with torch.no_grad():
            eg = torch.Generator(device=device).manual_seed(4041729)
            x = (
                target(32, eg)
                if label == "truth"
                else sample_joint(
                    Exact(),
                    torch.zeros(32, 2, h, w, device=device),
                    mask,
                    eg,
                    steps=int(label[5:]),
                )
            )
            records.append(dict(seed=seed, control=label, **stats(x)))
    (out / "metrics.json").write_text(json.dumps(records, indent=2))
