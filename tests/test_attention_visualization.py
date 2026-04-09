import torch
from torch import nn

from ocean_emulators.aggregator.validate import attention as attention_module
from ocean_emulators.config import (
    AttentionBlockConfig,
    BottleneckBlockConfig,
    UNetAttentionConfig,
    UNetBackboneConfig,
)
from ocean_emulators.models.modules.blocks import AxialAttention, FullAttention


class DummyModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.unet = backbone

    def forward(self, x):
        return self.unet(x)


def test_capture_attention_toggles_axial_and_full_modules():
    backbone = UNetBackboneConfig(
        ch_width=[4, 8],
        dilation=[1, 1],
        n_layers=[1, 1],
        up_sampling_block="bilinear_upsample",
        attention=UNetAttentionConfig(
            encoder=[AttentionBlockConfig(attention_type="axial", num_heads=2), None],
            bottleneck=BottleneckBlockConfig(
                block_type="attention",
                attention=AttentionBlockConfig(attention_type="full", num_heads=2),
            ),
        ),
    ).build(in_channels=2, pad="circular", checkpointing=None)
    model = DummyModel(backbone)

    attn_modules = [
        module
        for module in model.modules()
        if isinstance(module, AxialAttention | FullAttention)
    ]
    assert attn_modules
    assert all(not module.capture_weights for module in attn_modules)

    with attention_module.capture_attention(model):
        assert all(module.capture_weights for module in attn_modules)

    assert all(not module.capture_weights for module in attn_modules)


def test_attention_aggregator_logs_axial_and_full_blocks(monkeypatch):
    backbone = UNetBackboneConfig(
        ch_width=[4, 8],
        dilation=[1, 1],
        n_layers=[1, 1],
        up_sampling_block="bilinear_upsample",
        attention=UNetAttentionConfig(
            encoder=[AttentionBlockConfig(attention_type="axial", num_heads=2), None],
            bottleneck=BottleneckBlockConfig(
                block_type="attention",
                attention=AttentionBlockConfig(attention_type="full", num_heads=2),
            ),
            decoder=[None, AttentionBlockConfig(attention_type="axial", num_heads=2)],
        ),
    ).build(in_channels=2, pad="circular", checkpointing=None)
    model = DummyModel(backbone)

    monkeypatch.setattr(
        attention_module,
        "plot_attention_map",
        lambda *args, **kwargs: ("map", kwargs["axis"]),
    )
    monkeypatch.setattr(
        attention_module,
        "plot_attention_receptive_field",
        lambda *args, **kwargs: "axial_rf",
    )
    monkeypatch.setattr(
        attention_module,
        "plot_full_attention_receptive_field",
        lambda *args, **kwargs: "full_rf",
    )

    aggregator = attention_module.AttentionAggregator(model, query_lat=0, query_lon=0)
    with attention_module.capture_attention(model):
        _ = model(torch.randn(1, 2, 8, 16))

    aggregator.record_batch(
        loss=torch.tensor(0.0),
        target_data=None,
        gen_data=None,
        input_data=None,
        target_data_norm=None,
        gen_data_norm=None,
        input_data_norm=None,
    )

    logs = aggregator.get_logs("attention")

    assert "attention/encoder_0/height" in logs
    assert "attention/encoder_0/width" in logs
    assert "attention/encoder_0/receptive_field" in logs
    assert "attention/bottleneck/matrix" in logs
    assert "attention/bottleneck/receptive_field" in logs
    assert "attention/decoder_1/height" in logs
    assert "attention/decoder_1/width" in logs
    assert "attention/decoder_1/receptive_field" in logs
