import torch
from contextlib import contextmanager

@contextmanager
def nvtx_range(profiler_active, label):
    if profiler_active:
        torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        if profiler_active:
            torch.cuda.nvtx.range_pop()

@contextmanager
def profiler_context(profiling_enabled, is_main_process, profiling_start_epoch, profiling_end_epoch, epoch, data_iter_step, data_iter_len):
    profiler_active = False
    if profiling_enabled and profiling_start_epoch <= epoch <= profiling_end_epoch:
        if is_main_process and data_iter_step == 0 and profiling_start_epoch == epoch:
            print("Starting profiler at epoch: ", epoch)
            torch.cuda.profiler.start()
        profiler_active = True
    try:
        yield profiler_active
    finally:
        if profiling_enabled and data_iter_step == data_iter_len - 1 and is_main_process and epoch == profiling_end_epoch:
            print("Ending profiler at epoch: ", epoch)
            torch.cuda.profiler.stop()