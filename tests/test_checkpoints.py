import tempfile
from pathlib import Path

import torch

from ocean_emulators.models.samudra import Samudra
from ocean_emulators.utils.train import CheckpointPaths


def test_checkpoint_save_load():
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create checkpoint paths
        ckpt_paths = CheckpointPaths(temp_path)

        # Create a simple model configuration
        class MockConfig:
            def __init__(self):
                self.ch_width = [4, 8, 16]
                self.n_out = 2
                self.pred_residuals = False
                self.last_kernel_size = 3
                self.pad = "circular"
                self.dilation = [1, 2]
                self.n_layers = [1, 1]
                self.core_block = type(
                    "obj",
                    (object,),
                    {
                        "block_type": "conv_next_block",
                        "kernel_size": 3,
                        "activation": "capped_gelu",
                        "upscale_factor": 4,
                        "norm": "batch",
                    },
                )

        # Create mock data
        batch_size = 2
        height, width = 32, 32
        wet_mask = torch.ones((height, width), dtype=torch.bool)
        area_weights = torch.ones((height, width))
        static_data = None

        # Create model
        config = MockConfig()
        model = Samudra(
            config,
            hist=1,
            wet=wet_mask,
            area_weights=area_weights,
            static_data=static_data,
        )

        # Create some test data
        test_input = torch.randn((batch_size, config.ch_width[0], height, width))

        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
            "epoch": 1,
            "best_val_loss": 0.0,
            "best_inf_loss": 0.0,
            "ema": None,
        }
        torch.save(checkpoint, ckpt_paths.latest_checkpoint_path)

        # Create a new model instance
        new_model = Samudra(
            config,
            hist=1,
            wet=wet_mask,
            area_weights=area_weights,
            static_data=static_data,
        )

        # Load checkpoint
        loaded_checkpoint = torch.load(ckpt_paths.latest_checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint["model"])

        # Verify models produce same output
        model.eval()
        new_model.eval()
        with torch.no_grad():
            output1 = model.forward_once(test_input)
            output2 = new_model.forward_once(test_input)

        # Check outputs are equal
        assert torch.allclose(output1, output2), (
            "Model outputs differ after loading checkpoint"
        )

        # Verify model can make predictions
        assert output1.shape == (batch_size, config.n_out, height, width), (
            "Output shape is incorrect"
        )
        assert not torch.isnan(output1).any(), "Model produced NaN values"
        assert not torch.isinf(output1).any(), "Model produced infinite values"
