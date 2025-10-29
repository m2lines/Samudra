"""Fourier positional encodings.

Sources taken from MSFT's Auroral model:
  https://github.com/microsoft/aurora/blob/main/aurora/model/fourier.py

Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Modifications made by Open Athena AI Foundation contributors.
"""

import math

import numpy as np
import torch
from torch import nn

__author__ = "@wesselb"


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(
            self.lower <= x.abs(), torch.all(x.abs() <= self.upper)
        )
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.
