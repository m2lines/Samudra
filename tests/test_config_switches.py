import pytest
import torch

from ocean_emulators.config import (
    CorrectorConfig,
    DynamicLossConfig,
    SamudraConfig,
    UNetBackboneConfig,
    WeightInitConfig,
    build_loss_fn,
)
from ocean_emulators.datasets import TrainData
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.data import DataSource
from ocean_emulators.utils.loss import DynamicLoss


def _make_samudra(
    src: DataSource,
    *,
    in_channels: int,
    out_channels: int,
    num_input_states: int,
    num_output_states: int,
    weight_init: WeightInitConfig | None = None,
):
    return SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        pos_channels=0,
        weight_init=weight_init or WeightInitConfig(),
    ).build(
        in_channels=in_channels,
        out_channels=out_channels,
        num_input_states=num_input_states,
        num_output_states=num_output_states,
        static_data_for_corrector=None,
        srcs=[src],
    )


def test_samudra_supports_explicit_input_output_states(dummy_src: DataSource):
    src = dummy_src
    h, w = src.grid_size
    model = _make_samudra(
        src,
        in_channels=3,
        out_channels=1,
        num_input_states=2,
        num_output_states=1,
    )

    assert model.num_input_states == 2
    assert model.num_output_states == 1
    assert model.hist == 1

    train_data = TrainData(
        num_prognostic_channels=2,
        ctx=GridContext(src.masks.prognostic, src.resolution),
    )
    for _ in range(3):
        train_data.append(
            torch.randn(1, 3, h, w, requires_grad=True),
            torch.randn(1, 1, h, w),
        )

    loss = model(train_data, loss_fn=torch.nn.MSELoss())

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_samudra_ignores_disabled_corrector_config_for_asymmetric_state_counts(
    dummy_src: DataSource,
):
    model = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        corrector=CorrectorConfig(),
        pos_channels=0,
    ).build(
        in_channels=3,
        out_channels=1,
        num_input_states=2,
        num_output_states=1,
        static_data_for_corrector=None,
        srcs=[dummy_src],
    )

    assert model.corrector is None
    assert model.num_input_states == 2
    assert model.num_output_states == 1


def test_samudra_rejects_enabled_corrector_for_asymmetric_state_counts(
    dummy_src: DataSource,
):
    with pytest.raises(
        ValueError,
        match="Correctors currently require matching input and output state counts.",
    ):
        SamudraConfig(
            unet=UNetBackboneConfig(
                ch_width=[2],
                dilation=[1],
                n_layers=[1],
            ),
            corrector=CorrectorConfig(non_negative_corrector_names=["thetao_0"]),
            pos_channels=0,
        ).build(
            in_channels=3,
            out_channels=1,
            num_input_states=2,
            num_output_states=1,
            static_data_for_corrector=None,
            srcs=[dummy_src],
        )


def test_samudra_honors_hist_when_explicit_state_counts_are_omitted(
    dummy_src: DataSource,
):
    model = SamudraConfig(
        unet=UNetBackboneConfig(
            ch_width=[2],
            dilation=[1],
            n_layers=[1],
        ),
        pos_channels=0,
    ).build(
        in_channels=4,
        out_channels=2,
        hist=1,
        static_data_for_corrector=None,
        srcs=[dummy_src],
    )

    assert model.num_input_states == 2
    assert model.num_output_states == 2
    assert model.hist == 1


def test_build_loss_fn_supports_inverse_sqrt_weighting():
    loss_fn = build_loss_fn(
        DynamicLossConfig(metric="mse", weighting="inverse_sqrt_loss", limit=None),
        device=torch.device("cpu"),
        num_channels=2,
        pad_mode="circular",
    )
    assert isinstance(loss_fn, DynamicLoss)

    loss_fn.update(torch.tensor([4.0, 9.0]))
    expected = (
        torch.ones(2) * (DynamicLoss.N_WINDOW - 1) + torch.tensor([0.5, 1 / 3])
    ) / DynamicLoss.N_WINDOW

    assert torch.allclose(loss_fn.loss_scale_per_channel().cpu(), expected)


def test_samudra_kaiming_normal_init_zeroes_biases(dummy_src: DataSource):
    model = _make_samudra(
        dummy_src,
        in_channels=2,
        out_channels=1,
        num_input_states=1,
        num_output_states=1,
        weight_init=WeightInitConfig(type="kaiming_normal"),
    )

    biases = [
        module.bias.detach()
        for module in model.modules()
        if isinstance(
            module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
        )
        and module.bias is not None
    ]

    assert biases
    assert all(torch.count_nonzero(bias) == 0 for bias in biases)
