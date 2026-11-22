# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact replica checks with diagnostics that identify divergent parameters."""

import hashlib

import torch
import torch.distributed as dist


def tensor_digest(value):
    """Hash logical tensor bytes without a floating-point reduction."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_replicas(parameters, world):
    """Require identical raw/EMA bytes and report elementwise differences."""
    local = {name: tensor_digest(value) for name, value in parameters.items()}
    hashes: list = [None] * world
    dist.all_gather_object(hashes, local)
    mismatched = [
        name
        for name in local
        if any(other[name] != hashes[0][name] for other in hashes)
    ]
    details = {}
    for name in mismatched:
        value = parameters[name].detach()
        reference = value.clone()
        dist.broadcast(reference, src=0)
        difference = (value - reference).abs()
        statistics = torch.stack(
            (
                difference.max(),
                (value != reference).sum().float(),
                reference.abs().max(),
            )
        )
        dist.all_reduce(statistics, op=dist.ReduceOp.MAX)
        details[name] = dict(
            max_absolute_difference=float(statistics[0]),
            max_different_elements=int(statistics[1]),
            reference_max_absolute=float(statistics[2]),
        )
    return dict(identical=not mismatched, per_rank_sha256=hashes, differences=details)
