from pathlib import Path

import torch


class Profiler:
    def __init__(
        self,
        output_dir: Path,
        cuda_snapshot_frequency: int | None,
    ) -> None:
        self.output_dir = output_dir
        self.cuda_snapshot_frequency = cuda_snapshot_frequency

    def start(self) -> None:
        if self.cuda_snapshot_frequency is not None:
            torch.cuda.memory._record_memory_history()

    def after_batch(self, num_batches_seen: int) -> None:
        if self.cuda_snapshot_frequency is not None:
            if num_batches_seen % self.cuda_snapshot_frequency == 0:
                torch.cuda.memory._dump_snapshot(
                    str(
                        self.output_dir
                        / f"cuda_memory_snapshot_{num_batches_seen}.pickle"
                    )
                )
