import torch

import ocean_emulators.constants as c
from ocean_emulators.config import SamudraConfig, UNetBackboneConfig
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource, Normalize


def test_positional_parameters_update(dummy_src: DataSource):
    """Verify that positional parameters can learn something in a tiny example."""
    src = dummy_src
    h, w = src.grid_size
    masks = src.masks

    # Create the model itself with learned positional embeddings
    config = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        pos_channels=1,
    )
    tensor_map = c.TensorMap.get_instance()
    normalize = Normalize.get_instance()
    model = config.build(
        in_channels=2,
        out_channels=1,
        hist=0,
        static_data_for_corrector=None,
        srcs=[src],
        tensor_map=tensor_map,
        normalize=normalize,
        dataset_spec=src.dataset_spec,
    )

    # Verify we have created the positional embeddings
    assert model.positional_params is not None
    assert model.positional_params.shape == (config.pos_channels, h, w)
    assert not torch.allclose(
        model.positional_params.detach(),
        torch.zeros_like(model.positional_params),
    )

    # Run a step and confirm they have changed
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(1, 2, h, w)
    optimizer.zero_grad()
    out = model.forward_once(
        x, GridContext(masks.prognostic, src.resolution, src.resolution)
    )
    loss = out.sum()
    loss.backward()
    before = model.positional_params.detach().clone()
    optimizer.step()
    assert not torch.allclose(model.positional_params.detach(), before)
