# TODO: Enable setting parameters for activation functions
import torch


class ReLU(torch.nn.Module):
    """
    Implements a ReLU.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: passed to torch.nn.ReLU
        """
        super().__init__()
        self.relu = torch.nn.ReLU(**kwargs)

    def forward(self, inputs):
        x = self.relu(inputs)
        return x


class _CappedActivation(torch.nn.Module):
    """An activation clipped to a maximum value.

    `cap` is a buffer because every checkpoint written so far carries one, so
    dropping it would break strict loading. The forward clamps against a plain
    float instead: the tensor overload of `clamp` broadcasts a 0-d tensor,
    which is a slower kernel for the same answer, and calling `.item()` in the
    forward would sync the GPU on every activation.
    """

    def __init__(self, activation: torch.nn.Module, cap_value: float):
        super().__init__()
        self.activation = activation
        self.cap = torch.nn.Buffer(torch.tensor(cap_value, dtype=torch.float32))
        self._cap_value = float(cap_value)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # A checkpoint may carry a different cap than the constructor default.
        self._cap_value = float(self.cap)

    def forward(self, inputs):
        return torch.clamp(self.activation(inputs), max=self._cap_value)


class CappedLeakyReLU(_CappedActivation):
    """
    Implements a LeakyReLU with capped maximum value.
    """

    def __init__(self, cap_value=10.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeakyReLU
        """
        super().__init__(torch.nn.LeakyReLU(**kwargs), cap_value)


class CappedGELU(_CappedActivation):
    """
    Implements a GELU with capped maximum value.
    """

    def __init__(self, cap_value=10.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.GELU
        """
        super().__init__(torch.nn.GELU(**kwargs), cap_value)
