import torch

from ocean_emulators.config import SpatialDynamicLossConfig, build_loss_fn
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.loss import SpatialDynamicLoss


def _make_ctx(channels: int, lat: int, lon: int) -> GridContext:
    mask = torch.ones(channels, lat, lon)
    lats = torch.linspace(-89.5, 89.5, lat)
    lons = torch.linspace(0.5, 359.5, lon)
    return GridContext(label_mask=mask, input_resolution_cpu=(lats, lons))


def test_build_loss_fn_dynamic_spatial():
    loss = build_loss_fn(
        SpatialDynamicLossConfig(
            metric="mse",
            limit=100.0,
            ema_window=100,
            spatial_resolution_lat=2.0,
        ),
        device=torch.device("cpu"),
        num_channels=2,
        pad_mode="circular",
    )
    assert isinstance(loss, SpatialDynamicLoss)


def test_dynamic_spatial_exact_unscaled_loss_across_calls():
    torch.manual_seed(0)
    num_channels = 2
    hist = 2
    lat = 12
    lon = 24
    channels_with_history = hist * num_channels

    loss = SpatialDynamicLoss(
        metric="mse",
        limit=None,
        device=torch.device("cpu"),
        num_channels=num_channels,
        ema_window=100,
        spatial_resolution_lat=2.0,
    )
    ctx = _make_ctx(channels_with_history, lat, lon)

    pred_a = torch.randn(1, channels_with_history, lat, lon)
    pred_b = torch.randn(1, channels_with_history, lat, lon)
    target = torch.zeros_like(pred_a)

    loss.start_batch()
    _ = loss(pred_a, target, ctx)
    _ = loss(pred_b, target, ctx)
    loss.end_batch()

    expected_unscaled = (
        (
            ((pred_a - target) ** 2).mean(dim=(0, 2, 3))
            + ((pred_b - target) ** 2).mean(dim=(0, 2, 3))
        )
        .reshape(hist, num_channels)
        .mean(dim=0)
    )

    actual_unscaled = loss.last_unscaled_loss_per_channel()
    assert actual_unscaled is not None
    assert torch.allclose(actual_unscaled, expected_unscaled)


def test_dynamic_spatial_update_creates_channel_spread_stats():
    num_channels = 2
    hist = 2
    lat = 12
    lon = 24
    channels_with_history = hist * num_channels

    loss = SpatialDynamicLoss(
        metric="mse",
        limit=None,
        device=torch.device("cpu"),
        num_channels=num_channels,
        ema_window=100,
        spatial_resolution_lat=30.0,
    )
    ctx = _make_ctx(channels_with_history, lat, lon)

    pred = torch.ones(1, channels_with_history, lat, lon)
    target = torch.zeros_like(pred)
    pred[:, 0, : lat // 2, : lon // 2] = 10.0
    pred[:, 2, lat // 2 :, lon // 2 :] = 6.0

    loss.update(pred, target, ctx)
    scale_mean = loss.loss_scale_per_channel()
    scale_std = loss.loss_scale_std_per_channel()

    assert torch.isfinite(scale_mean).all()
    assert torch.isfinite(scale_std).all()
    assert (scale_std > 0).any()


def test_dynamic_spatial_state_dict_roundtrip():
    num_channels = 2
    hist = 2
    lat = 12
    lon = 24
    channels_with_history = hist * num_channels
    ctx = _make_ctx(channels_with_history, lat, lon)

    loss = SpatialDynamicLoss(
        metric="mse",
        limit=50.0,
        device=torch.device("cpu"),
        num_channels=num_channels,
        ema_window=100,
        spatial_resolution_lat=30.0,
    )
    pred = torch.randn(1, channels_with_history, lat, lon)
    target = torch.zeros_like(pred)
    loss.update(pred, target, ctx)

    restored = SpatialDynamicLoss(
        metric="mse",
        limit=50.0,
        device=torch.device("cpu"),
        num_channels=num_channels,
        ema_window=25,
        spatial_resolution_lat=5.0,
    )
    restored.load_state_dict(loss.state_dict())

    assert torch.allclose(
        loss.loss_scale_per_channel(), restored.loss_scale_per_channel()
    )
    assert torch.allclose(
        loss.loss_scale_std_per_channel(), restored.loss_scale_std_per_channel()
    )
