# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe noise invisible to a trained decoder on an explicitly synthetic wet patch."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from samudra.experiments.joint_diffusion import JointInteriorDecoder, sample_joint


@torch.no_grad()
def diagnose(decoder, known_channels, *, seed=13):
    """A synthetic counterexample, not an estimate of the real masked-ocean error."""
    torch.manual_seed(seed)
    fields = decoder.fields
    unknown = [i for i in range(fields) if i not in known_channels]
    kernel = decoder.stem.weight.double().sum((-1, -2))[:, unknown]
    rank = int(torch.linalg.matrix_rank(kernel))
    if rank >= len(unknown):
        raise ValueError("No constant-channel input nullspace exists")
    _, _, vectors = torch.linalg.svd(kernel, full_matrices=True)
    vector = torch.zeros(fields)
    vector[unknown] = vectors[-1].float()
    vector = vector.reshape(1, fields, 1, 1).expand(1, fields, 8, 12).contiguous()
    noisy = torch.randn_like(vector)
    latent = torch.randn(1, decoder.condition[0].in_channels, 8, 12)
    mask = torch.ones_like(vector[0])
    known = torch.zeros_like(vector, dtype=torch.bool)
    known[:, known_channels] = True
    rows = []
    for level in (0.01, 0.1, 1.0, 10.0, 80.0):
        sigma = torch.tensor([level])
        delta = decoder(noisy + vector, sigma, latent, mask) - decoder(
            noisy, sigma, latent, mask
        )
        expected = vector / (1 + level**2)
        rows.append(
            dict(
                sigma=level,
                maximum_absolute_identity_error=float((delta - expected).abs().max()),
            )
        )
    generator = torch.Generator().manual_seed(21)
    noise = torch.randn(vector.shape, generator=generator)
    sampling = []
    for steps in (8, 16, 32):
        outputs = []
        for perturbation in (0, 1):
            # Patch only the sampler's initial draw; this decoder contains no RNG.
            with patch("torch.randn", return_value=noise + perturbation * vector):
                outputs.append(
                    sample_joint(
                        decoder,
                        latent,
                        mask,
                        generator,
                        steps=steps,
                        known_values=torch.zeros_like(vector),
                        known_mask=known,
                    )
                )
        delta = outputs[1] - outputs[0]
        gain = float((delta * vector).sum() / vector.square().sum())
        sampling.append(
            dict(
                steps=steps,
                initial_unit_noise_survival=gain,
                off_direction_relative_norm=float(
                    torch.linalg.vector_norm(delta - gain * vector)
                    / torch.linalg.vector_norm(vector)
                ),
            )
        )
    return dict(
        scope="Synthetic 8x12 all-wet patch; known surface channels anchored to zero. Not a quantitative real-ocean noise floor.",
        unknown_channels=len(unknown),
        stem_rank=rank,
        constant_channel_nullity=len(unknown) - rank,
        stem_perturbation_max=float(
            (decoder.stem(vector) - decoder.stem(torch.zeros_like(vector))).abs().max()
        ),
        denoising=rows,
        sampling=sampling,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--known-channels", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.decoder, weights_only=True, map_location="cpu")
    decoder = JointInteriorDecoder(
        state["condition.0.weight"].shape[1],
        state["stem.weight"].shape[1],
        width=state["stem.weight"].shape[0],
    ).eval()
    decoder.load_state_dict(state, strict=True)
    result = diagnose(decoder, args.known_channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
