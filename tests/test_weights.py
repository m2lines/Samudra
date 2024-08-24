# https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html

import torch
import pytest


def load_model_weights(path):
    """Helper function to load model weights from a .pt file"""
    return torch.load(path).get("model")


def compare_model_weights(model1_state_dict, model2_state_dict):
    """Compares two model state dicts for equality"""
    are_equal = True
    for key in model1_state_dict:
        if not torch.equal(model1_state_dict[key], model2_state_dict[key]):
            print(f"Difference found in layer: {key}")
            are_equal = False
    return are_equal


def test_models_have_same_weights(model1_path, model2_path):
    torch.cuda.empty_cache()
    model1_state_dict = load_model_weights(model1_path)
    model2_state_dict = load_model_weights(model2_path)

    assert compare_model_weights(
        model1_state_dict, model2_state_dict
    ), "The models have different weights!"
