import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

from ocean_emulators.datasets import ReplayCursor

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReplayEntry:
    state: torch.Tensor
    cursor: ReplayCursor


class ReplayBuffer:
    def __init__(
        self,
        *,
        buffer_size: int,
        storage_dtype: torch.dtype,
        generator: torch.Generator,
        pin_memory: bool,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("Replay buffer_size must be >= 1")
        self.buffer_size = buffer_size
        self.storage_dtype = storage_dtype
        self.generator = generator
        self.pin_memory = pin_memory
        self.entries: list[ReplayEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        return len(self.entries) >= self.buffer_size

    def append(self, entry: ReplayEntry) -> None:
        if self.is_full:
            raise ValueError("Replay buffer is full")
        self.entries.append(self._prepare_entry(entry))

    def replace(self, index: int, entry: ReplayEntry) -> None:
        self.entries[index] = self._prepare_entry(entry)

    def sample_indices(self, batch_size: int, max_lead_steps: int) -> list[int]:
        eligible = [
            index
            for index, entry in enumerate(self.entries)
            if entry.cursor.lead_step < max_lead_steps
        ]
        if not eligible:
            raise RuntimeError(
                "Replay buffer has no entries below the active max_lead_steps "
                f"cap ({max_lead_steps})."
            )
        if len(eligible) >= batch_size:
            draw = torch.randperm(
                len(eligible),
                generator=self.generator,
                device="cpu",
            )[:batch_size]
        else:
            draw = torch.randint(
                len(eligible),
                (batch_size,),
                generator=self.generator,
                device="cpu",
            )
        return [eligible[int(i)] for i in draw]

    def random_indices(self, count: int) -> list[int]:
        if not self.entries:
            return []
        if len(self.entries) >= count:
            draw = torch.randperm(
                len(self.entries),
                generator=self.generator,
                device="cpu",
            )[:count]
        else:
            draw = torch.randint(
                len(self.entries),
                (count,),
                generator=self.generator,
                device="cpu",
            )
        return [int(i) for i in draw]

    def state_dict(self, *, world_size: int, rank: int) -> dict[str, Any]:
        return {
            "buffer_size": self.buffer_size,
            "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            "world_size": world_size,
            "rank": rank,
            "generator_state": self.generator.get_state(),
            "entries": [
                {
                    "state": entry.state.cpu(),
                    "cursor": dataclasses.asdict(entry.cursor),
                }
                for entry in self.entries
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["buffer_size"] != self.buffer_size:
            logger.warning(
                "Replay buffer_size changed on resume: checkpoint=%s current=%s. "
                "Loading available entries into the current-sized buffer.",
                state_dict["buffer_size"],
                self.buffer_size,
            )
        self.generator.set_state(state_dict["generator_state"])
        self.entries = []
        for raw_entry in state_dict["entries"][: self.buffer_size]:
            cursor = ReplayCursor(**raw_entry["cursor"])
            self.append(ReplayEntry(state=raw_entry["state"], cursor=cursor))

    def _prepare_entry(self, entry: ReplayEntry) -> ReplayEntry:
        state = entry.state.detach().to(
            device="cpu",
            dtype=self.storage_dtype,
            copy=True,
        )
        if self.pin_memory and torch.cuda.is_available():
            try:
                state = state.pin_memory()
            except RuntimeError as e:
                logger.warning(
                    "Could not pin replay buffer state; continuing unpinned. Error: %s",
                    e,
                )
                self.pin_memory = False
        return ReplayEntry(state=state, cursor=entry.cursor)


def replay_sidecar_path(checkpoint_path: Path, rank: int) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.replay_rank{rank}.pt")
