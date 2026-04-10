import pytest
import torch

from ocean_emulators.constants import LoaderVersion
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG


def _with_rust_loader(train_config):
    return train_config.model_copy(
        update={
            "data": train_config.data.model_copy(
                update={
                    "loader_version": str(LoaderVersion.OM4_RUST_V0.value),
                    "num_workers": 0,
                }
            )
        }
    )


@pytest.mark.parametrize(
    "data_source,config_name",
    [
        ("mock-om4", DEFAULT_CONFIG),
        ("mock-om4", "test/train_default_2step.yaml"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_tide_loader_matches_torch_loader(train_config):
    pytest.importorskip("tide")
    rust_config = _with_rust_loader(train_config)

    with MultitonScope():
        torch_trainer = Trainer(train_config)
        torch_trainer.init_data_loaders(cur_step=train_config.steps[0])
        torch_batch = torch_trainer.train_loader[0]

    with MultitonScope():
        rust_trainer = Trainer(rust_config)
        rust_trainer.init_data_loaders(cur_step=rust_config.steps[0])
        rust_batch = rust_trainer.train_loader[0]

    assert torch.allclose(torch_batch.get_input(0), rust_batch.get_input(0))
    assert torch.allclose(torch_batch.get_label(0), rust_batch.get_label(0))

    for step in range(1, len(torch_batch)):
        prev_prediction = torch.randn_like(torch_batch.get_label(step - 1))
        assert torch.allclose(
            torch_batch.merge_prognostic_and_boundary(prev_prediction, step),
            rust_batch.merge_prognostic_and_boundary(prev_prediction, step),
        )
        assert torch.allclose(torch_batch.get_label(step), rust_batch.get_label(step))


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_tide_trainer_smoke_cpu(train_config):
    pytest.importorskip("tide")
    rust_config = _with_rust_loader(train_config)

    with MultitonScope():
        trainer = Trainer(rust_config)
        trainer.run()
