import torch

from ocean_emulators.config import TrainConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS
from ocean_emulators.datasets import TrainData
from ocean_emulators.models.modules.patchembed import PatchEmbed2d

from .test_datasets import make_loader


def test_makes_patches():
    x = torch.randn(1, 20, 180, 360)

    patch_embed = PatchEmbed2d(
        input_vars=[f"var_{i}" for i in range(x.shape[1])],
        patch_size=4,
        embed_dim=1024,
        hist=0,
        norm=None,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1024, 4050)


def test_makes_patches__high_res():
    x = torch.randn(1, 20, 360, 720)

    patch_embed = PatchEmbed2d(
        input_vars=[f"var_{i}" for i in range(x.shape[1])],
        patch_size=4,
        embed_dim=1024,
        hist=0,
        norm=None,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1024, 16200)


def test_makes_patches__more_variables():
    x = torch.randn(1, 71, 180, 360)

    patch_embed = PatchEmbed2d(
        input_vars=[f"var_{i}" for i in range(x.shape[1])],
        patch_size=4,
        embed_dim=1024,
        hist=0,
        norm=None,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1024, 4050)


def test_makes_patches__with_layer_norm():
    x = torch.randn(1, 20, 180, 360)

    patch_embed = PatchEmbed2d(
        input_vars=[f"var_{i}" for i in range(x.shape[1])],
        patch_size=4,
        embed_dim=1024,
        hist=0,
        norm=torch.nn.LayerNorm,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1024, 4050)


def test_makes_patches__with_history():
    x = torch.randn(1, 20, 180, 360)

    patch_embed = PatchEmbed2d(
        input_vars=[f"var_{i}" for i in range(x.shape[1] // 2)],
        patch_size=4,
        embed_dim=1024,
        hist=1,
        norm=None,
    )

    patches = patch_embed(x)

    assert patches.shape == (1, 1024, 4050)


def test_patch_embed__on_real_data(train_config: TrainConfig):
    prognostic_vars = PROGNOSTIC_VARS[train_config.experiment.prognostic_vars_key]
    boundary_vars = BOUNDARY_VARS[train_config.experiment.boundary_vars_key]
    input_vars = prognostic_vars + boundary_vars

    patch_embed = PatchEmbed2d(input_vars, hist=train_config.data.hist)

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
            assert patches.shape == (1, 1024, 4050)

            prev_prediction = td.get_label(
                max(0, len(td) - 2)
            )  # Let's assume our predictor does a perfect job.
            merged_input = td.merge_prognostic_and_boundary(
                prognostic=prev_prediction, step=len(td) - 1
            )
            patches = patch_embed(merged_input)
            assert patches.shape == (1, 1024, 4050)
