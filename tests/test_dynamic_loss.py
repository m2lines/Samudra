import torch

from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.loss import DynamicLoss, loss_fn_from_metric


def _make_ctx(channels: int, lat: int, lon: int) -> GridContext:
    mask = torch.ones(channels, lat, lon)
    lats = torch.linspace(-89.5, 89.5, lat)
    lons = torch.linspace(0.5, 359.5, lon)
    return GridContext(label_mask=mask, input_resolution_cpu=(lats, lons))


def test_dynamic_loss_preserves_unit_mean_scale():
    torch.manual_seed(0)
    num_channels = 3
    hist = 2
    lat = 8
    lon = 16
    channels_with_history = hist * num_channels

    loss = DynamicLoss(
        loss_fn=loss_fn_from_metric("mse"),
        limit=100.0,
        device=torch.device("cpu"),
        num_channels=num_channels,
    )
    ctx = _make_ctx(channels_with_history, lat, lon)

    pred = torch.randn(4, channels_with_history, lat, lon)
    target = torch.zeros_like(pred)
    for _ in range(5):
        loss.update(pred, target, ctx)
        pred = pred * 1.1

    scales = loss.loss_scale_per_channel()
    assert torch.isclose(
        scales.mean(), torch.tensor(1.0, dtype=scales.dtype), atol=1e-6
    )
    assert not torch.allclose(scales, torch.ones_like(scales))
