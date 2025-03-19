import torch

_CHOSEN_DEVICE: torch.device | None = None


def using_gpu() -> bool:
    return get_device().type == "cuda"


# TODO(jder): We'd like to remove this (and other global singletons)
# by moving to some training framework + context objects once we have
# benchmarking set up.
def set_device(device: torch.device) -> None:
    global _CHOSEN_DEVICE
    _CHOSEN_DEVICE = device


def get_device() -> torch.device:
    """The primary device for inference.

    We avoid setting this as torch.device so we can choose when
    to shuttle tensors to the device.
    """
    if _CHOSEN_DEVICE is not None:
        return _CHOSEN_DEVICE

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
