import equinox as eqx
from jaxtyping import Float

from ocean_emulators.constants import Grid


class Samudrax(eqx.Module):
    def __init__(self):
        pass

    def __call__(
        self, x: Float[Grid, " prognostic_vars+boundary_vars"]
    ) -> Float[Grid, " prognostic_vars"]:
        return x[:154, :, :]
