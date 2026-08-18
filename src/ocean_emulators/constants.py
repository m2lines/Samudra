import enum
import logging
from typing import TypeAlias

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float
from torch import Tensor
import numpy as np

from ocean_emulators.utils.multiton import Multiton

# Common Type Aliases
# See "Existing jaxtyping annotations" section of
#  https://docs.kidger.site/jaxtyping/api/array/#array

Lat = Float[Tensor, "lat"]
Lon = Float[Tensor, "lon"]
Grid = Float[Tensor, "lat lon"]
Prognostic = Float[
    Grid, "*batch prognostic_vars"
]  # equivalent to "*batch prognostic_vars lat lon"
Boundary = Float[Grid, "*batch boundary_vars"]
# A note from jaxtyping (why we can't do "prognostic_vars+boundary_vars"):
#   In practice you should usually only use symbolic axes in annotations
#   for return types, referring only to axes annotated for arguments.
# So, we'll leave this default and use symbolic axes locally.
Input: TypeAlias = Float[Grid, "*batch total_vars"]

Example = tuple[Input, Prognostic]

GridMask = Bool[Tensor, "lat lon"]
PrognosticMask = Bool[GridMask, "prognostic_vars"]

SingleChannelVar = Float[Tensor, "batch time lat lon"]
DictSingleChannelVar = dict[str, SingleChannelVar]
SinglePrognosticTimeSeries = Float[Grid, "*batch time"]

SingleTimeSeriesOutput = Float[Tensor, "batch=1 time prognostic_vars lat lon"]
BatchTimeSeriesOutput = Float[Tensor, "batch time=(hist+1) prognostic_vars lat lon"]
HistBatched = Float[Tensor, "batch_hist prognostic_vars lat lon"]
HistChanneled = Float[Tensor, "batch hist_prognostic_vars lat lon"]


MAX_TRAIN_MODEL_STEPS_FORWARD = 200

# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

# These represent depth centers
DEPTH_LEVELS = [
    0.5,
    1.57,
    2.79,
    4.185,
    5.78,
    7.595,
    9.66,
    12.01,
    14.68,
    17.705,
    21.125,
    24.99,
    29.345,
    34.24,
    39.725,
    45.855,
    52.69,
    60.28,
    68.685, 
    77.965, 
    88.175, 
    99.37, 
    111.6, 
    124.915, 
    139.365, 
    154.99,
    171.825,
    189.9, 
    209.235, 
    229.855, 
    251.77, 
    274.985, 
    299.505, 
    325.32, 
    352.42,
    380.79, 
    410.41, 
    441.255, 
    473.305, 
    506.54, 
    540.935, 
    576.465, 
    613.11,
    650.855, 
    689.685, 
    729.595, 
    770.585, 
    812.66, 
    855.835, 
    900.135, 
    945.595,
]

# SELECT A SUBSET OF THE TOTAL DEPTH LEVELS
# SELECT A SUBSET OF THE TOTAL DEPTH LEVELS
#DEPTH_LEVELS = DEPTH_LEVELS[::3]

NEXT_DEPTH_LEVEL = 1000#900.135#1000

# Depth thicknesses
DEPTH_THICKNESS = [n - p for p, n in zip(DEPTH_LEVELS, DEPTH_LEVELS[1:] + [NEXT_DEPTH_LEVEL])]


def _depth_interfaces(centers: list[float]) -> list[float]:
    """Depth of every cell face, derived from the cell centres.

    A centre is halfway between its two faces, so pinning the surface at 0 gives
    the rest: face[i+1] = 2*centre[i] - face[i]. This reproduces LLC4320's own
    layer thicknesses (1.00, 1.14, 1.30, 1.49 ... m) exactly.
    """
    faces = [0.0]
    for center in centers:
        faces.append(2.0 * center - faces[-1])
    return faces


DEPTH_INTERFACES = _depth_interfaces(DEPTH_LEVELS)

# Variables on cell faces (`k_p1`) rather than cell centres (`k`). W_i is the
# top face of cell i, so consecutive W levels are one layer thickness apart, not
# one centre-to-centre spacing -- tens of metres different at depth.
INTERFACE_VARS = frozenset({"W"})


def depth_of_channel(channel_name: str) -> float:
    """Depth in metres of a 3D channel such as `Theta_7` or `W_7`."""
    base, level = channel_name.rsplit("_", 1)
    table = DEPTH_INTERFACES if base in INTERFACE_VARS else DEPTH_LEVELS
    return float(table[int(level)])


N = len(DEPTH_LEVELS)
DEPTH_I_LEVELS = [str(i) for i in range(N)]
MASK_VARS = [f"wetmask_{i}" for i in range(N)]


MASK_ALL_LEVELS_VAR = "wetmask"
HEAT_VAR_NAME = "Theta"

RHO_0 = 1035.0  # DENSITY_OF_WATER_CM4 kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER_CM4 J/kg/K
SECONDS_PER_TIME_STEP = 60 # hourly
# Would the above be MINUTES per time step? I don't see this wiring in anywhere. Keep note of it though.

PrognosticVarNames = list[str]
PROGNOSTIC_VARS: dict[str, PrognosticVarNames] = {
    "single_1": [f"Theta_{DEPTH_I_LEVELS[0]}"],
    "single_2": [
        k + str(j) for k in ["Theta_"] for j in DEPTH_I_LEVELS[:2]
    ],
    "thermo_51": [
        k + str(j) for k in ["Theta_", "Salt_"] for j in DEPTH_I_LEVELS
    ],
    "all": [
        k + str(j) for k in ["U_", "V_", "Theta_", "Salt_"] for j in DEPTH_I_LEVELS
    ]
    + ["Eta"]
    # Add "W_" below to train with vertical velocity. It goes last so the
    # existing channel indices do not move. Needs W_lev_* in the means/stds and
    # W channels in the patch cache.
    # + [k + str(j) for k in ["W_"] for j in DEPTH_I_LEVELS],
}

BoundaryVarNames = list[str]
BOUNDARY_VARS: dict[str, BoundaryVarNames] = {
    "single": ["oceQnet"],
    "double": ["oceQnet", "Eta"],
    "all": ["oceTAUX", "oceTAUY", "oceQnet", "Eta"],
}

DEFAULT_METADATA = {
    "Theta": {
        "long_name": "Sea Water Potential Temperature",
        "units": r"\degree C",
    },
    "Salt": {
        "long_name": "Sea Water Salinity",
        "units": "psu",
    },
    "U": {
        "long_name": "Sea Water X Velocity",
        "units": "m/s",
    },
    "V": {
        "long_name": "Sea Water Y Velocity",
        "units": "m/s",
    },
    "Eta": {
        "long_name": "Sea surface height above geoid",
        "units": "m",
    },
    "oceTAUX": {
        "long_name": "Surface Downward X Stress",
        "units": "N/m^2",
    },
    "oceTAUY": {
        "long_name": "Surface Downward Y Stress",
        "units": "N/m^2",
    },
    "oceQnet": {
        "long_name": "Surface ocean heat flux from "
        "SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat",
        "units": "W/m^2",
    },
}

def construct_metadata(data: xr.Dataset) -> dict[str, dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[str(var)] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[str(var)]
            elif (key := str(var).split("_")[0]) in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[key]
            else:
                logger.info(f"{var} does not have any default metadata")
                metadata[str(var)] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class LoaderVersion(enum.Enum):
    OM4_TORCH = "om4-torch"


# TODO(#95): See if this can be removed and replaced.
class TensorMap(Multiton):
    def _initialize(self, prognostic_vars_key: str, boundary_vars_key: str):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.prognostic_vars_key = prognostic_vars_key
        self.VAR_3D_IDX: dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: dict[str, torch.Tensor] = {}

        self.INPT_BOUNDARY_IDX: dict[str, torch.Tensor] = {}
        var_set_2d: list[str] = []
        var_set_3d: list[str] = []
        for out in PROGNOSTIC_VARS[prognostic_vars_key]:
            var_split = out.split("_")
            if len(var_split) == 1:
                var_set_2d.append(var_split[0])
            else:
                var_set_3d.append(var_split[0])

        # One entry per variable rather than one per channel: callers iterate
        # these to reach a variable's whole depth column through VAR_3D_IDX.
        self.VAR_SET_2D = list(dict.fromkeys(var_set_2d))
        self.VAR_SET_3D = list(dict.fromkeys(var_set_3d))

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(
                [out.split("_")[0] for out in PROGNOSTIC_VARS[prognostic_vars_key]]
            )
        )

        assert 51 == len(DEPTH_I_LEVELS) == len(DEPTH_THICKNESS) == len(DEPTH_LEVELS) == len(MASK_VARS)

        levels_str = prognostic_vars_key.split("_")[-1]
        if "all" in levels_str:
            levels = 51
        else:
            levels = int(levels_str)

        self.DEPTH_SET = DEPTH_I_LEVELS[:levels]
        self.prognostic_var_names = PROGNOSTIC_VARS[prognostic_vars_key]
        self.boundary_var_names = BOUNDARY_VARS[boundary_vars_key]
        self.dz = torch.tensor(DEPTH_THICKNESS[:levels])

        # Depth of every prognostic channel, NaN for the 2D channels which do
        # not have one. Anything that differences adjacent levels needs this
        # rather than the level index: the levels are not evenly spaced, so an
        # index is not a unit of depth. Face variables (W) resolve to their
        # interface depth, so their spacing is the layer thickness.
        self.channel_depth_centers: Float[Tensor, " prognostic_vars"] = torch.tensor(
            [
                depth_of_channel(name) if "_" in name else float("nan")
                for name in self.prognostic_var_names
            ],
            dtype=torch.float32,
        )

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()
        self._populate_boundary_idx()

    def vertical_spacing(self, variable: str) -> Float[Tensor, " levels"]:
        """Centre-to-centre depth spacing, in metres, between a variable's levels.

        Returns one value per adjacent pair of levels, so a variable with `n`
        levels yields `n - 1` spacings. On LLC4320 these run from ~1.07 m at the
        surface to ~45.5 m at depth, a factor of 42, which is why a vertical
        difference has to be divided by them to mean anything comparable across
        depth.
        """
        if variable not in self.VAR_3D_IDX:
            raise ValueError(
                f"{variable} is not a prognostic variable of this tensor map."
            )
        depths = self.channel_depth_centers[self.VAR_3D_IDX[variable].long()]
        if depths.numel() < 2 or bool(torch.isnan(depths).any()):
            raise ValueError(
                f"{variable} does not span at least two depth levels, so it has "
                "no vertical spacing."
            )
        spacing = depths[1:] - depths[:-1]
        if not bool((spacing > 0).all()):
            raise ValueError(
                f"{variable} channels are not ordered by increasing depth; "
                f"got centres {depths.tolist()}."
            )
        return spacing

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)

        self.DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
            [
                self.DP_3D_IDX[self.DEPTH_SET[0]],
                torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]).to(torch.int32),
            ]
        )

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.boundary_var_names):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])
