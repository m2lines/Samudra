import torch

from ocean_emulators.config import SamudraConfig, UNetBackboneConfig
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource


def _build_small_model(src: DataSource, *, pos_channels: int):
    config = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        pos_channels=pos_channels,
    )
    model = config.build(
        in_channels=2,
        out_channels=1,
        hist=0,
        static_data_for_corrector=None,
        srcs=[src],
    )
    return config, model


def test_positional_parameters_update(dummy_src: DataSource):
    """Verify that positional parameters can learn something in a tiny example."""
    src = dummy_src
    h, w = src.grid_size
    masks = src.masks

    # Create the model itself with learned positional embeddings
    config, model = _build_small_model(src, pos_channels=1)

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
    out = model.forward_once(x, GridContext(masks.prognostic, src.resolution))
    loss = out.sum()
    loss.backward()
    before = model.positional_params.detach().clone()
    optimizer.step()
    assert not torch.allclose(model.positional_params.detach(), before)


def test_positional_channels_do_not_change_unet_input_channels(dummy_src: DataSource):
    src = dummy_src
    _, model = _build_small_model(src, pos_channels=3)
    assert model.unet.in_channels == 2
    assert model.unet.pos_projs is not None
    assert len(model.unet.pos_projs) == model.unet.num_steps + 1


def test_positional_injection_path_affects_forward_output(dummy_src: DataSource):
    src = dummy_src
    h, w = src.grid_size
    masks = src.masks
    _, model = _build_small_model(src, pos_channels=1)
    assert model.positional_params is not None
    assert model.unet.pos_projs is not None
    assert model.unet.pos_scales is not None

    model.eval()
    with torch.no_grad():
        model.positional_params.fill_(1.0)
        for proj in model.unet.pos_projs:
            proj.weight.fill_(1.0)
        model.unet.pos_scales.fill_(1.0)

    x = torch.randn(1, 2, h, w)
    ctx = GridContext(masks.prognostic, src.resolution)
    with torch.no_grad():
        out_with_pos = model.forward_once(x, ctx)
        model.unet.pos_scales.zero_()
        out_without_pos = model.forward_once(x, ctx)

    assert not torch.allclose(out_with_pos, out_without_pos)
