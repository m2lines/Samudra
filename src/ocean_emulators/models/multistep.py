from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from ocean_emulators.constants import Grid
from ocean_emulators.datasets import TrainData
from ocean_emulators.models.samudrax import Samudrax


class MultiStepModel(eqx.Module):
    model: Samudrax

    def __init__(self, model: Samudrax):
        self.model = model

    def __call__(
        self, x: TrainData, state
    ) -> tuple[Float[Grid, "steps prognostic_vars"], Any]:
        outputs: list[Float[Grid, " prognostic_vars"]] = []
        num_steps = len(x)
        for step in range(num_steps):
            if step == 0:
                input_tensor = x.get_initial_input()
            else:
                input_tensor = x.merge_prognostic_and_boundary(
                    prognostic=outputs[-1], step=step
                )

            result, state = jax.vmap(
                self.model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
            )(input_tensor, state)

            outputs.append(result)
        return jnp.stack(outputs), state
