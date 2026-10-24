# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Four-rank execution of one unchanged global eight-example update."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from samudra.experiments.monthly_batching import batched_loss, groups


class UpdateLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, items, source, area):
        losses, pieces, rms = [], [], []
        for group in groups(items, 2):
            loss, component, magnitude = batched_loss(self.model, group, source, area)
            losses.append(loss * len(group) / len(items))
            pieces.extend(
                [
                    {key: float(value[index]) for key, value in component.items()}
                    for index in range(len(group))
                ]
            )
            rms.extend(magnitude.tolist())
        return sum(losses), pieces, rms


class DistributedUpdate:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world not in (1, 4):
            raise ValueError(
                "Only single-GPU and four-rank global-batch-eight execution are qualified"
            )
        if self.world == 4:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group("nccl", timeout=timedelta(minutes=45))

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def share(self, value):
        if self.world == 1:
            return value
        box = [value if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def prepared(self, items):
        """Broadcast tensor payloads directly on NVLink; metadata only uses pickle."""
        if self.world == 1:
            return items
        tensors: list[torch.Tensor] = []

        def encode(value):
            if isinstance(value, torch.Tensor):
                index = len(tensors)
                tensors.append(value.contiguous())
                return {
                    "__tensor_index__": index,
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype).split(".")[-1],
                }
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [encode(v) for v in value]
            return value

        metadata = self.share(encode(items) if self.rank == 0 else None)
        received = {}

        def decode(value):
            if isinstance(value, dict) and "__tensor_index__" in value:
                index = value["__tensor_index__"]
                if index not in received:
                    tensor = (
                        tensors[index]
                        if self.rank == 0
                        else torch.empty(
                            value["shape"],
                            device="cuda",
                            dtype=getattr(torch, value["dtype"]),
                        )
                    )
                    dist.broadcast(tensor, src=0)
                    received[index] = tensor
                return received[index]
            if isinstance(value, dict):
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        return decode(metadata)

    def wrap(self, model):
        if self.world == 1:
            return None
        return nn.parallel.DistributedDataParallel(
            UpdateLoss(model),
            device_ids=[self.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    def write(self, path, text):
        if self.rank == 0:
            path.write_text(text)

    def evaluate(self, function, *args, **kwargs):
        result = function(*args, **kwargs) if self.rank == 0 else None
        return self.share(result)

    def rng_states(self):
        return self.gather(
            dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state().cpu())
        )
