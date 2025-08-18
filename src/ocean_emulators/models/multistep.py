import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from ocean_emulators.constants import Grid
from ocean_emulators.datasets import TrainData
from ocean_emulators.models.samudrax import Samudrax


class MultiStepModel(eqx.Module):
    model: Samudrax

    def __init__(self, model: Samudrax):
        self.model = model

    def __call__(self, x: TrainData) -> Float[Grid, "steps prognostic_vars"]:
        outputs: list[Float[Grid, " prognostic_vars"]] = []
        num_steps = len(x)
        for step in range(num_steps):
            if step == 0:
                input_tensor = x.get_initial_input()
            else:
                input_tensor = x.merge_prognostic_and_boundary(
                    prognostic=outputs[-1], step=step
                )
            outputs.append(self.model(input_tensor))
        return jnp.stack(outputs)
