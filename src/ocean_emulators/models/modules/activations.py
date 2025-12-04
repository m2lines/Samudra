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


class CappedLeakyReLU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.
    """

    def __init__(self, cap_value=10.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.relu = torch.nn.LeakyReLU(**kwargs)
        self.cap = torch.nn.Buffer(torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.relu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x


class CappedGELU(torch.nn.Module):
    """
    Implements a ReLU with capped maximum value.
    """

    def __init__(self, cap_value=10.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.gelu = torch.nn.GELU(**kwargs)
        self.cap = torch.nn.Buffer(torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        return CappedGELUFunction.apply(inputs, self.cap)


# TODO: write a test
class CappedGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cap):
        ctx.save_for_backward(x)
        ctx.cap = cap
        result = torch.nn.functional.gelu(x)
        result.clamp_(max=cap)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        cap = ctx.cap

        # recompute gelu for clamp mask
        g = torch.nn.functional.gelu(x)
        mask = (g < cap).to(grad_output.dtype)
        del g

        grad_output.mul_(mask)
        del mask

        # gelu backward gives grad_output * gelu'(x)
        return torch.ops.aten.gelu_backward(grad_output, x, approximate="none"), None
