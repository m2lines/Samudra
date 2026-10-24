"""Test channel-wise correction scaling with the original fixed state-MSE loss.

Scales use the single training example's target increment RMS, not spatially
varying target information. This is a fitting/preconditioning control; a full
training experiment would estimate tendency scales on its training split only.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

import representation_sweep as trial


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scale-mode", choices=("identity", "increment_rms"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args, remainder = parser.parse_known_args()
    data = trial.Prepared(args.data)
    change = np.asarray(data.prog[2], dtype=np.float64) - data.prog[1]
    mask = data.mask
    # Per-channel RMS, with an explicit floor only for an exactly quiet channel.
    scales = np.sqrt(((change**2) * mask).sum((-2, -1)) / mask.sum((-2, -1)))
    assert np.isfinite(scales).all() and (scales >= 0).all()
    used = np.ones_like(scales) if args.scale_mode == "identity" else np.maximum(scales, 1e-6)
    scale_tensor = torch.from_numpy(np.tile(used.astype(np.float32), 2))[None, :, None, None]
    base = trial.OceanModel

    class ScaledCorrectionModel(base):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            self.register_buffer("correction_scale", scale_tensor.clone())

        def forward(self, *forward_args, **forward_kwargs):
            prediction, hidden = super().forward(*forward_args, **forward_kwargs)
            return prediction * self.correction_scale, hidden

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale-protocol.json").write_text(json.dumps({
        "mode": args.scale_mode, "input": args.data,
        "channel_order": data.meta["variables"],
        "normalized_increment_rms": scales.tolist(),
        "applied_scales": used.tolist(), "scale_floor": 1e-6,
        "loss": "Unchanged original equal-channel wet-cell state MSE",
        "source": "First target minus latest input in this one training example",
        "scope": "Single-example optimization test; not independently estimated production statistics",
    }, indent=2) + "\n")
    trial.OceanModel = ScaledCorrectionModel
    sys.argv = [sys.argv[0], "--data", args.data, "--output", args.output, *remainder]
    trial.main()


if __name__ == "__main__":
    main()
