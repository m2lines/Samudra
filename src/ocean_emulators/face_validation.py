"""Scoring a face that is split across ranks, as if it were one field.

A face-parallel rank holds a slice of the face -- nine of its 36 tiles at four
ranks -- so anything it computes on its own describes that slice, not the face.
Reporting rank 0's number, which is what the per-rank validation path did,
meant scoring the open-ocean quadrant and never the coastal one.

Both metrics here are sums before they are ratios, which is what lets the
pieces add up exactly:

* **Loss.** The loss is built with face-wide denominators (see
  `FaceParallelContext.global_loss_norms`), so each rank's value is a SHARE of
  the face mean rather than a mean of its own; the ranks' shares sum to the
  face's. The denominators carry DDP's ``1 / world_size``, so the face value is
  the MEAN over ranks -- the same convention training uses.
* **RMSE.** Pooled: every rank sums area-weighted squared error and wet area
  over the cells it owns, one all-reduce adds them, and the ratio is taken
  once. That is exactly the RMSE of the stitched face -- land tiles contribute
  nothing rather than a NaN, and a 6%-wet coastal tile weighs by its ocean
  area, not as much as an open-ocean one.

Only the cells a tile OWNS are scored. Tiles overlap, and scoring each in full
would count the seams twice.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, TypeVar

import torch

from ocean_emulators.utils.loss import LossFn

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class FaceMetrics:
    """One step's face-wide scores, identical on every rank."""

    #: ``[C]`` face loss per channel.
    loss_per_channel: torch.Tensor
    #: Channel mean of ``loss_per_channel``, as the per-rank path reports it.
    loss: torch.Tensor
    #: ``[C]`` pooled, area-weighted RMSE; NaN for a channel with no wet cell.
    rmse_per_channel: torch.Tensor
    #: Mean over the channels that have wet cells.
    rmse: torch.Tensor


class FaceScorer:
    """Face loss and pooled face RMSE from a rank's own tiles.

    Every rank must call `score` the same number of times: it all-reduces.
    """

    def __init__(
        self,
        *,
        loss_fn: LossFn,
        weight: torch.Tensor,
        wet: torch.Tensor,
        area: torch.Tensor,
        world_size: int,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Args:
            loss_fn: Built with face-wide denominators, so it returns a share.
            weight: ``[Tlocal, C, H, W]`` bool, each tile's own wet cells
                restricted to the cells it owns. Handed to ``loss_fn`` as the
                sample weight, exactly as its denominators were derived.
            wet: ``[C, H, W]`` mask the model output is masked with; ANDed in
                so the RMSE scores the same cells the loss does.
            area: ``[Tlocal, 1, H, W]`` cell area of each tile. Units cancel.
            world_size: Ranks the face is split over.
            reduce_sum: Sums a tensor across those ranks.
        """
        if weight.ndim != 4:
            raise ValueError(f"weight must be [T, C, H, W], got {tuple(weight.shape)}")
        if area.shape[0] != weight.shape[0] or area.shape[-2:] != weight.shape[-2:]:
            raise ValueError(
                f"area {tuple(area.shape)} does not match weight "
                f"{tuple(weight.shape)}"
            )
        self.loss_fn = loss_fn
        self.weight = weight.bool()
        self.mask = self.weight & wet.bool().to(weight.device).unsqueeze(0)
        self.area = area.to(device=weight.device, dtype=torch.float32)
        self.world_size = world_size
        self.reduce_sum = reduce_sum
        # Constant for the whole run: the owned wet area of the FACE, per
        # channel. One collective here instead of one per step.
        self.face_area = reduce_sum(self._owned_area())

    @property
    def num_tiles(self) -> int:
        return self.weight.shape[0]

    def _owned_area(self) -> torch.Tensor:
        total = torch.zeros(
            self.mask.shape[1], dtype=torch.float64, device=self.mask.device
        )
        for tile in range(self.num_tiles):
            total += torch.where(self.mask[tile], self.area[tile], 0.0).sum(
                dim=(-2, -1), dtype=torch.float64
            )
        return total

    @torch.no_grad()
    def local_terms(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``[2, C]`` float64: this rank's loss share and area-weighted SSE.

        One tile at a time. The loss denominators are fixed, so any split of
        the tiles sums to the same share, and a tile is the split with the
        smallest transients -- `gradient_h` alone builds about eight tensors
        the size of what it is given.
        """
        if prediction.shape != target.shape or prediction.shape[0] != self.num_tiles:
            raise ValueError(
                f"Expected prediction and target shaped [{self.num_tiles}, C, H, "
                f"W], got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        terms = torch.zeros(
            (2, self.mask.shape[1]), dtype=torch.float64, device=prediction.device
        )
        for tile in range(self.num_tiles):
            span = slice(tile, tile + 1)
            terms[0] += self.loss_fn(
                prediction[span], target[span], sample_weight=self.weight[span]
            ).to(torch.float64)
            squared = (prediction[tile] - target[tile]).square_()
            terms[1] += torch.where(
                self.mask[tile], squared.mul_(self.area[tile]), 0.0
            ).sum(dim=(-2, -1), dtype=torch.float64)
        return terms

    @torch.no_grad()
    def face_metrics(self, terms: torch.Tensor) -> FaceMetrics:
        """Reduce `local_terms` across the ranks. Collective."""
        total = self.reduce_sum(terms)
        loss_per_channel = (total[0] / self.world_size).to(torch.float32)
        face_area = self.face_area.to(total.device)
        rmse_per_channel = torch.where(
            face_area > 0,
            (total[1] / face_area.clamp_min(1e-30)).sqrt(),
            torch.nan,
        ).to(torch.float32)
        return FaceMetrics(
            loss_per_channel=loss_per_channel,
            loss=loss_per_channel.mean(),
            rmse_per_channel=rmse_per_channel,
            rmse=rmse_per_channel.nanmean(),
        )

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> FaceMetrics:
        """Face metrics for one step. Collective."""
        return self.face_metrics(self.local_terms(prediction, target))


class ReadAhead(Generic[T]):
    """Run reads in order on one background thread, ``depth`` ahead of use.

    Validation reads do not depend on the model -- a rollout's forcing and
    truth are fixed by its window -- so the next steps can be read while the
    GPU works on this one. One thread, because the reads it runs are already
    parallel inside (`GroupChunkReader` decodes on its own pool); more would
    only oversubscribe the store, whose optimum is measured, not guessed.
    """

    def __init__(self, reads: Sequence[Callable[[], T]], *, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._reads = list(reads)
        self._depth = depth

    def __iter__(self) -> Iterator[T]:
        pending: collections.deque[Future[T]] = collections.deque()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="val_read")
        next_read = 0
        try:
            while next_read < len(self._reads) and len(pending) < self._depth:
                pending.append(executor.submit(self._reads[next_read]))
                next_read += 1
            while pending:
                result = pending.popleft().result()
                if next_read < len(self._reads):
                    pending.append(executor.submit(self._reads[next_read]))
                    next_read += 1
                yield result
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
