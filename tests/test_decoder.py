import torch

from ocean_emulators.models.modules.decoder import PerceiverDecoder


def test_reconstructs_patches():
    """Test basic patch reconstruction with square patches."""
    # Input: processed patches from UNet backbone
    x = torch.randn(1, 4, 1, 2)  # (batch, embed_dim, h_patches, w_patches)

    decoder = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        grid_size=(4, 8),
        perceiver_depth=2,
        perceiver_latent_dim=3,
    )

    reconstructed = decoder(x)

    # Should reconstruct to full spatial resolution
    assert reconstructed.shape == (1, 10, 4, 8)  # (batch, out_channels, height, width)


def test_reconstructs_patches__high_res():
    """Test reconstruction with higher resolution input."""
    x = torch.randn(1, 5, 2, 4)  # (batch, embed_dim, h_patches, w_patches)

    decoder = PerceiverDecoder(
        in_channels=5,
        out_channels=10,
        grid_size=(8, 16),
        perceiver_depth=2,
        perceiver_latent_dim=3,
    )

    reconstructed = decoder(x)

    assert reconstructed.shape == (1, 10, 8, 16)


def test_reconstructs_patches__more_output_channels():
    """Test reconstruction with more output channels."""
    x = torch.randn(1, 4, 1, 2)  # (batch, embed_dim, h_patches, w_patches)

    decoder = PerceiverDecoder(
        in_channels=4,
        out_channels=20,
        grid_size=(4, 8),
        perceiver_depth=2,
        perceiver_latent_dim=3,
    )

    reconstructed = decoder(x)

    assert reconstructed.shape == (1, 20, 4, 8)


def test_batch_size_independence():
    """Test that decoder works with different batch sizes."""
    decoder = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        grid_size=(4, 8),
        perceiver_depth=2,
        perceiver_latent_dim=3,
    )

    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 4, 1, 2)
        output = decoder(x)
        assert output.shape == (batch_size, 10, 4, 8)
