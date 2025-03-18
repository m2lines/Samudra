import numpy as np

from utils.data import flatten_masks, mask, unflatten_masks


def test_mask_roundtrip(data_source):
    data = data_source.data

    unflattened = unflatten_masks(data.copy())
    flattened = flatten_masks(unflattened.copy())

    assert flattened == data, "Assume a safe roundtrip"


def test_mask__zeros_data(data_source):
    data = data_source.data

    unflattened = unflatten_masks(data.copy())
    masked = mask(unflattened)

    for orig_da, maked_da in zip(unflattened.values(), masked.values()):
        assert np.count_nonzero(maked_da.values) < np.count_nonzero(orig_da.values)
