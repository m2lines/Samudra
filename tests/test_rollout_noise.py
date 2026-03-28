import torch

from ocean_emulators.datasets import TrainData
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.rollout_noise import RolloutNoiseInjector


class IdentityModel(BaseModel):
    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        return fts[:, : self.out_channels]


def _build_train_data(num_steps: int, height: int = 8, width: int = 8) -> TrainData:
    train_data = TrainData(num_prognostic_channels=1)
    for _ in range(num_steps):
        input_tensor = torch.zeros((1, 2, height, width), dtype=torch.float32)
        input_tensor[:, 0] = 1.0
        label = torch.zeros((1, 1, height, width), dtype=torch.float32)
        train_data.append(input_tensor, label)
    return train_data


def test_rollout_noise_gaussian_perturbs_state() -> None:
    injector = RolloutNoiseInjector(
        wet=torch.ones((1, 8, 8), dtype=torch.bool),
        probability=1.0,
        gaussian_std=0.2,
        structured_scale=0.0,
    )
    injector.train()

    torch.manual_seed(0)
    state = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    noisy = injector(state)

    assert not torch.allclose(noisy, state)


def test_rollout_noise_lattice_gaussian_is_spatially_coherent() -> None:
    injector = RolloutNoiseInjector(
        wet=torch.ones((1, 32, 32), dtype=torch.bool),
        probability=1.0,
        gaussian_std=0.2,
        gaussian_mode="lattice",
        gaussian_lattice_stride=8,
        gaussian_blur_kernel=7,
        gaussian_blur_sigma=2.0,
        structured_scale=0.0,
    )
    injector.train()

    torch.manual_seed(0)
    state = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    noise = injector(state) - state

    adjacent_similarity = (noise[:, :, :, 1:] * noise[:, :, :, :-1]).mean()
    assert adjacent_similarity > 0.0


def test_structured_noise_concentrates_near_fronts() -> None:
    injector = RolloutNoiseInjector(
        wet=torch.ones((1, 32, 32), dtype=torch.bool),
        probability=1.0,
        gaussian_std=0.0,
        structured_scale=0.5,
        structured_front_power=2.0,
        structured_mask_smoothing_kernel=3,
        structured_seed_prob=0.25,
        structured_patch_kernel=11,
        structured_patch_sigma=2.5,
        structured_patch_quantile=0.6,
        structured_sign_mode="patch",
    )
    injector.train()

    state = torch.full((1, 1, 32, 32), 0.5, dtype=torch.float32)
    state[:, :, :, 16:] += 1.0

    torch.manual_seed(0)
    noisy = injector(state)
    delta = (noisy - state).abs()[0, 0]

    near_front = delta[:, 14:18].mean()
    far_from_front = torch.cat((delta[:, :4], delta[:, -4:]), dim=-1).mean()
    assert near_front > far_from_front


def test_rollout_noise_is_training_only() -> None:
    injector = RolloutNoiseInjector(
        wet=torch.ones((1, 8, 8), dtype=torch.bool),
        probability=1.0,
        gaussian_std=0.5,
        structured_scale=0.0,
    )
    model = IdentityModel(
        in_channels=2,
        out_channels=1,
        wet=torch.ones((1, 8, 8), dtype=torch.bool),
        hist=0,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        static_data=None,
        gradient_detach_interval=0,
        rollout_noise_injector=injector,
    )
    train_data = _build_train_data(num_steps=2)

    torch.manual_seed(0)
    model.train()
    outputs_train = model(train_data)

    model.eval()
    outputs_eval = model(train_data)

    expected = torch.ones_like(outputs_eval[1])
    assert not torch.allclose(outputs_train[1], expected)
    assert torch.allclose(outputs_eval[1], expected)
