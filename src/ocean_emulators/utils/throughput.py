import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol


class DataSizeCalculator(Protocol):
    """Protocol for calculating data size from loaded data."""

    def __call__(self, data: Any) -> int:
        """Return data size in bytes."""
        ...


def calculate_numpy_size(data: Any) -> int:
    """Calculate size of numpy arrays, tensors, or dict of arrays."""
    import numpy as np

    total_bytes = 0

    if hasattr(data, "nbytes"):  # numpy arrays, torch tensors
        return int(data.nbytes)
    elif isinstance(data, dict):
        for v in data.values():
            total_bytes += calculate_numpy_size(v)
        return total_bytes
    elif isinstance(data, (list, tuple)):
        for item in data:
            total_bytes += calculate_numpy_size(item)
        return total_bytes
    elif hasattr(data, "dtype") and hasattr(data, "shape"):
        # Generic array-like object
        return int(np.prod(data.shape) * np.dtype(data.dtype).itemsize)
    else:
        return 0


@contextmanager
def measure_throughput(
    size_calculator: DataSizeCalculator = calculate_numpy_size, print_stats: bool = True
) -> Iterator["ThroughputMeasurer"]:
    """Context manager for measuring data loading throughput.

    Args:
        size_calculator: Function to calculate data size in bytes
        print_stats: Whether to print throughput statistics on exit

    Yields:
        ThroughputMeasurer: Object to record data loads

    Example:
        with measure_throughput() as measurer:
            for batch in data_loader:
                measurer.record(batch)
        # Prints: "Throughput: 2.34 GB/s (loaded 4.68 GB in 2.00s)"
    """
    measurer = ThroughputMeasurer(size_calculator)
    measurer.start()
    try:
        yield measurer
    finally:
        measurer.stop()
        if print_stats:
            print(measurer.get_stats())


class ThroughputMeasurer:
    """Measures data loading throughput."""

    def __init__(self, size_calculator: DataSizeCalculator = calculate_numpy_size):
        self.size_calculator = size_calculator
        self.total_bytes = 0
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.num_batches = 0

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.perf_counter()
        self.total_bytes = 0
        self.num_batches = 0

    def record(self, data: Any) -> None:
        """Record a data batch."""
        if self.start_time is None:
            raise RuntimeError("Must call start() before recording data")

        batch_size = self.size_calculator(data)
        self.total_bytes += batch_size
        self.num_batches += 1

    def stop(self) -> None:
        """Stop timing."""
        self.end_time = time.perf_counter()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    def get_throughput_bps(self) -> float:
        """Get throughput in bytes per second."""
        elapsed = self.get_elapsed_time()
        return self.total_bytes / elapsed if elapsed > 0 else 0.0

    def get_throughput_gbps(self) -> float:
        """Get throughput in GB per second."""
        return self.get_throughput_bps() / (1024**3)

    def get_throughput_mbps(self) -> float:
        """Get throughput in MB per second."""
        return self.get_throughput_bps() / (1024**2)

    def get_stats(self) -> str:
        """Get formatted throughput statistics."""
        elapsed = self.get_elapsed_time()
        total_gb = self.total_bytes / (1024**3)
        throughput_gbps = self.get_throughput_gbps()
        throughput_mbps = self.get_throughput_mbps()

        if total_gb >= 0.1:
            return (
                f"Throughput: {throughput_gbps:.2f} GB/s (loaded {total_gb:.2f} GB"
                f" in {elapsed:.2f}s, {self.num_batches} batches)"
            )
        else:
            total_mb = self.total_bytes / (1024**2)
            return (
                f"Throughput: {throughput_mbps:.2f} MB/s (loaded {total_mb:.2f} MB"
                f" in {elapsed:.2f}s, {self.num_batches} batches)"
            )
