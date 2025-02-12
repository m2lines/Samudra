# TODO(#21): Replace with real project tests.
import torch


def test_placeholder(device):
    """A placeholder test that should fail when run on GPU."""
    assert device != torch.device("cuda:0")
