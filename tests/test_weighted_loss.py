import torch

from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.loss import WeightedLoss
from ocean_emulators.utils.multiton import MultitonScope


def test_weighted_loss_default_channel_scales():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        weighted_loss = WeightedLoss(
            loss_fn=lambda pred, target: torch.ones(
                2 * len(tensor_map.prognostic_var_names), device=pred.device
            ),
            device=torch.device("cpu"),
            num_channels=len(tensor_map.prognostic_var_names),
        )

        scales = weighted_loss.loss_scale_per_channel()
        assert scales[tensor_map.prognostic_var_names.index("U_0")].item() == 1.0
        assert scales[tensor_map.prognostic_var_names.index("V_0")].item() == 1.0
        assert scales[tensor_map.prognostic_var_names.index("Theta_0")].item() == 1.5
        assert scales[tensor_map.prognostic_var_names.index("Salt_0")].item() == 1.5
        assert scales[tensor_map.prognostic_var_names.index("Eta")].item() == 1.5
