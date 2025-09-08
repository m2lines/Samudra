import pytest
import torch

from ocean_emulators.config import TrainConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS
from ocean_emulators.datasets import TrainData
from ocean_emulators.models.modules.patchembed import PerceiverPatchEmbed

from .test_datasets import make_loader


def test_makes_patches():
    x = torch.randn(1, 10, 4, 8)

    patch_embed = PerceiverPatchEmbed(
        n_channels=10,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1, 2, 4)


def test_makes_patches__high_res():
    x = torch.randn(1, 10, 8, 16)

    patch_embed = PerceiverPatchEmbed(
        n_channels=10,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 2, 4, 4)


def test_makes_patches__more_variables():
    x = torch.randn(1, 20, 4, 8)

    patch_embed = PerceiverPatchEmbed(
        n_channels=20,
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1, 2, 4)


@pytest.mark.skip(reason="Computationally expensive!")
def test_patch_embed__on_real_data(train_config: TrainConfig):
    prognostic_vars = PROGNOSTIC_VARS[train_config.experiment.prognostic_vars_key]
    boundary_vars = BOUNDARY_VARS[train_config.experiment.boundary_vars_key]
    input_vars = prognostic_vars + boundary_vars

    patch_embed = PerceiverPatchEmbed(
        n_channels=len(input_vars) * (1 + train_config.data.hist),
        patch_size=4,
        embed_dim=4,
        perceiver_depth=2,
    )

    with make_loader(train_config) as loader:
        for td in loader:
            assert isinstance(td, TrainData), (
                "The loader must return a TrainData! This is for the type checker."
            )
            # The forward pass of our base model manages a rollout across steps, where the model
            # only ever needs to predict one step. There are two possible inputs in this step rollout
            # system: the initial state, or a "merged" state, where we merge our latest prognostic
            # prediction with the boundary forcings for that step.
            # Thus, to ensure our patch encoder works as expected (for the one-step prediction part),
            # we only need to test that the patch embedding works for either the initial input or the
            # merged input.
            initial_input = td.get_initial_input()
            patches = patch_embed(initial_input)
            assert patches.shape[0] == 1
            assert patches.shape[-1] == 1024

            prev_prediction = td.get_label(
                max(0, len(td) - 2)
            )  # Let's assume our predictor does a perfect job.
            merged_input = td.merge_prognostic_and_boundary(
                prognostic=prev_prediction, step=len(td) - 1
            )
            patches = patch_embed(merged_input)
            assert patches.shape[0] == 1
            assert patches.shape[-1] == 1024
