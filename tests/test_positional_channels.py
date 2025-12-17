import numpy as np
import torch
import xarray as xr

from ocean_emulators.config import SamudraConfig, UNetBackboneConfig
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.data import DataSource, Masks, Normalize
from ocean_emulators.utils.multiton import MultitonScope


def test_positional_parameters_update():
    """Verify that positional parameters can learn something in a tiny example."""
    h, w = 4, 5
    with MultitonScope():
        # Create some tiny data
        TensorMap.init_instance("thetao_1", "hfds")
        coords = {"lev": [0], "y": np.arange(h), "x": np.arange(w)}
        data = xr.Dataset(
            {
                "thetao": (("lev", "y", "x"), np.zeros((1, h, w))),
                "hfds": (("y", "x"), np.zeros((h, w))),
            },
            coords=coords,
        )
        ones = xr.Dataset(
            {
                "thetao": (("lev", "y", "x"), np.ones((1, h, w))),
                "hfds": (("y", "x"), np.ones((h, w))),
            },
            coords=coords,
        )
        masks = Masks(torch.ones(h, w), torch.ones(h, w))
        src = DataSource(name="dummy", data=data, means=data, stds=ones, masks=masks)
        Normalize.init_instance(
            src,
            TensorMap.get_instance().prognostic_var_names,
            TensorMap.get_instance().boundary_var_names,
        )

        # Create the model itself with learned positional embeddings
        config = SamudraConfig(
            unet=UNetBackboneConfig(
                ch_width=[2],
                dilation=[1],
                n_layers=[1],
            ),
            pos_channels=1,
        )
        model = config.build(
            in_channels=2,
            out_channels=1,
            hist=0,
            wet=torch.ones(1, h, w, dtype=torch.bool),
            area_weights=torch.ones(h, w),
            static_data=None,
            lat=torch.from_numpy(src.data.y.values),
            lon=torch.from_numpy(src.data.x.values),
        )

        # Verify we have created the positional embeddings
        assert model.positional_params is not None
        assert model.positional_params.shape == (config.pos_channels, h, w)
        assert not torch.allclose(
            model.positional_params.detach(),
            torch.zeros_like(model.positional_params),
        )

        # Run a step and confirm they have changed
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(1, 2, h, w)
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        before = model.positional_params.detach().clone()
        optimizer.step()
        assert not torch.allclose(model.positional_params.detach(), before)
