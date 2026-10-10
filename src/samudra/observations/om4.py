# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""OM4 task samples from Samudra's canonical loader, with shared observation scales."""

import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from samudra.config import DataConfig
from samudra.datasets import BatchLoader, TorchTrainDataset
from samudra.observations.frame_cache import PreparedFrameCache
from samudra.utils.location import LocalLocation
from samudra.utils.train import collate_host_batches


class Om4Samples:
    def __init__(
        self, root, config: DataConfig, observations, device, cache_device="cpu"
    ):
        self.config, self.device, self.cache_device = config, device, cache_device
        self.bundle = config.build(LocalLocation(path=root))
        self.source = self.bundle.train_sources[0]
        self.names = self.bundle.data_layout.prognostic_var_names
        self.mask = self.source.masks.prognostic.to(device)
        lat, lon = self.source.resolution
        self.lat = lat.to(device).float()
        np.testing.assert_allclose(
            lat.cpu(), observations.grid["lat"], atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(
            lon.cpu(), observations.grid["lon"], atol=1e-5, rtol=0
        )
        if list(self.names) != list(observations.grid["names"]):
            raise ValueError("Task channel orders differ")
        self.weights = self.mask.float() * torch.deg2rad(self.lat).cos()[None, :, None]
        self.native_mean = torch.tensor(
            self.source.statistics(self.names).mean, device=device
        )
        self.native_std = torch.tensor(
            self.source.statistics(self.names).std, device=device
        )
        self.mean = observations.mean[0, 0, :, 0, 0]
        self.std = observations.std[0, 0, :, 0, 0]
        lat2, lon2 = torch.meshgrid(
            torch.deg2rad(self.lat),
            torch.deg2rad(lon.to(device).float()),
            indexing="ij",
        )
        self.geo = torch.stack(
            (lat2.cos() * lon2.cos(), lat2.cos() * lon2.sin(), lat2.sin())
        )
        self.surface = [self.names.index("thetao_0"), self.names.index("zos")]
        self.trainset = self.dataset(self.source)
        context = config.model_dump(mode="json")
        context["sources"][0]["train_time"]["end"] = "2013-07-31"
        context["sources"][0]["val_time"]["start"] = "2013-08-01"
        context["sources"][0]["inference_times"][0]["start"] = "2014-08-06"
        self.context_bundle = DataConfig.model_validate(context).build(
            LocalLocation(path=root)
        )
        self.valset = self.dataset(self.context_bundle.val_sources[0])
        self.val_ids = list(range(0, len(self.valset), 6))[:12]
        self.caches: dict[int, PreparedFrameCache] = {}

    def dataset(self, source, steps=6):
        return TorchTrainDataset(
            input_source=source,
            label_source=None,
            prognostic_var_names=self.names,
            boundary_var_names=self.bundle.data_layout.boundary_var_names,
            input_steps=19,
            output_steps=1,
            steps=steps,
            normalize_before_mask=True,
            masked_fill_value=0.0,
        )

    def native_loader(self, dataset, schedule):
        workers = self.config.loading.num_pytorch_workers()
        loader = DataLoader(
            dataset,
            batch_sampler=schedule,
            collate_fn=collate_host_batches,
            num_workers=workers,
            persistent_workers=False,
            pin_memory=self.device.type == "cuda",
            multiprocessing_context="spawn" if workers else None,
        )
        return BatchLoader(loader, [dataset], self.device)

    def prepare(self, dataset):
        source = dataset.sources[0]
        if id(source) in self.caches:
            return
        warm = self.dataset(source, 1)
        cache = PreparedFrameCache(
            len(source.time),
            len(self.names),
            3,
            tuple(self.mask.shape[-2:]),
            self.cache_device,
        )
        ids = list(range(0, len(warm), 19))
        if ids[-1] != len(warm) - 1:
            ids.append(len(warm) - 1)
        schedule = [[i] for i in ids]
        for loaded, (indices, batch) in enumerate(
            zip(schedule, self.native_loader(warm, schedule), strict=True), 1
        ):
            if loaded == 1 or loaded % 10 == 0 or loaded == len(schedule):
                print(
                    json.dumps(
                        {
                            "event": "om4_cache_warm",
                            "loaded": loaded,
                            "batches": len(schedule),
                        }
                    ),
                    flush=True,
                )
            # Cache stores the canonical float32 prepared values, never re-normalizes.
            offsets = np.asarray(indices)[:, None]
            history = offsets + np.arange(19)
            prog, boundary = batch.get_initial_input()
            cache._write(
                cache.prognostic,
                cache.prognostic_ready,
                history,
                prog.to(self.cache_device),
            )
            cache._write(
                cache.boundary,
                cache.boundary_ready,
                history,
                boundary.to(self.cache_device),
            )
            cache._write(
                cache.prognostic,
                cache.prognostic_ready,
                offsets + 19,
                batch.get_label(0).to(self.cache_device),
            )
        if not cache.prognostic_ready.all() or not cache.boundary_ready[:-1].all():
            raise ValueError("Incomplete OM4 cache")
        probes = [[0, len(dataset) - 1]]
        for indices, reference in zip(
            probes, self.native_loader(dataset, probes), strict=True
        ):
            offsets = np.asarray(indices)[:, None]
            for step, expected in enumerate(reference):
                history = offsets + step + np.arange(19)
                actual = [
                    cache._read(cache.prognostic, cache.prognostic_ready, history),
                    cache._read(cache.boundary, cache.boundary_ready, history),
                    cache._read(
                        cache.prognostic, cache.prognostic_ready, offsets + step + 19
                    ),
                ]
                for x, y in zip(actual, expected, strict=True):
                    torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)
        self.caches[id(source)] = cache

    def model_sample(self, dataset, ids):
        self.prepare(dataset)
        cache = self.caches[id(dataset.sources[0])]
        offsets = torch.tensor(ids, device=cache.prognostic.device)[:, None]
        history = offsets + torch.arange(19, device=offsets.device)
        future = offsets + torch.arange(18, 18 + dataset.steps, device=offsets.device)
        surface = cache.prognostic[history][:, :, self.surface].to(self.device)
        past = cache.boundary[history].to(self.device)
        truth = cache.prognostic[history[:, -2:]].to(self.device)
        forcing = cache.boundary[future].to(self.device)
        labels = cache.prognostic[future + 1].to(self.device)
        dates = dataset.sources[0].time.values[np.asarray(ids) + 18]
        phases = [2 * math.pi * (t.dayofyr - 1) / 365.25 for t in dates]
        season = surface.new_tensor([[math.sin(p), math.cos(p)] for p in phases])
        season = season[:, :, None, None].expand(-1, -1, *self.mask.shape[-2:])
        context = torch.cat((self.geo.expand(len(ids), -1, -1, -1), season), 1)

        def convert(values, channels):
            result = (
                values * self.native_std[channels][None, None, :, None, None]
                + self.native_mean[channels][None, None, :, None, None]
                - self.mean[channels][None, None, :, None, None]
            ) / self.std[channels][None, None, :, None, None]
            return result * self.mask[channels][None, None]

        return (
            convert(surface, self.surface),
            past,
            context,
            convert(truth, slice(None)),
            forcing,
            convert(labels, slice(None)),
        )
