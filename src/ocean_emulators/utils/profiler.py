from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity


class Profiler:
    def __init__(
        self,
        output_dir: Path,
        cuda_snapshot_frequency: int | None,
        tracing_activities: list[ProfilerActivity], # [] means disable
    ) -> None:
        self.output_dir = output_dir
        self.cuda_snapshot_frequency = cuda_snapshot_frequency
        self.tracing_activities = tracing_activities
        self._tracer: None | profile = None

    def start(self) -> None:
        if self.cuda_snapshot_frequency is not None:
            torch.cuda.memory._record_memory_history()

        if self.tracing_activities:
            self._tracer = profile(
                with_stack=True,
                activities=self.tracing_activities,
            )
            self._tracer.start()

    def after_batch(self, num_batches_seen: int) -> None:
        if self.cuda_snapshot_frequency is not None:
            if num_batches_seen % self.cuda_snapshot_frequency == 0:
                torch.cuda.memory._dump_snapshot(
                    str(
                        self.output_dir
                        / f"cuda_memory_snapshot_{num_batches_seen}.json"
                    )
                )
        if self._tracer:
            self._tracer.step()

    def after_epoch(self, epoch: int) -> None: # TODO: make after entire epoch, not train
        if self._tracer and epoch == 2: #TODO HACK
            self._tracer.stop()
            self._tracer.export_chrome_trace(f"{self.output_dir}/chrome_trace_{epoch}.json")
            self._tracer = None
