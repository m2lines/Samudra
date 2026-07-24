# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import logging
from typing import Literal, NamedTuple, Self

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float
from torch import Tensor

# Common Type Aliases
# See "Existing jaxtyping annotations" section of
#  https://docs.kidger.site/jaxtyping/api/array/#array

Lat = Float[Tensor, "lat"]
Lon = Float[Tensor, "lon"]
Grid = Float[Tensor, "lat lon"]
GridSize = tuple[int, int]
Prognostic = Float[
    Grid, "*batch prognostic_vars"
]  # equivalent to "*batch prognostic_vars lat lon"
Boundary = Float[Grid, "*batch boundary_vars"]
# A note from jaxtyping (why we can't do "prognostic_vars+boundary_vars"):
#   In practice you should usually only use symbolic axes in annotations
#   for return types, referring only to axes annotated for arguments.
# So, we'll leave this default and use symbolic axes locally.
type Input = Float[Grid, "*batch total_vars"]


class RolloutStep(NamedTuple):
    prognostic: Prognostic
    boundary: Boundary
    label: Prognostic


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

# Horizontal grid geometry. "gaussian" is a regular/rectilinear lat-lon grid whose
# 2D lat/lon are the outer product of the 1D axes, so they can be reconstructed by
# broadcasting. "tripolar" is curvilinear: lat/lon vary along both horizontal dims
# and cannot be rebuilt by broadcasting, so the real 2D coordinates must be carried
# with the data. Downstream code that reconstructs geometry must branch on this.
GridType = Literal["gaussian", "tripolar"]
PrognosticVarNames = list[str]
BoundaryVarNames = list[str]


@dataclasses.dataclass(frozen=True)
class DataLayout:
    """Canonical channel layout, physical metadata, and tensor index mappings.

    Raw dataset conventions belong to source-specific canonicalizers, not here.
    """

    depth_levels: tuple[float, ...]
    depth_thickness: tuple[float, ...]
    prognostic_var_names: PrognosticVarNames
    boundary_var_names: BoundaryVarNames
    default_metadata: dict[str, dict[str, str]]
    ocean_heat_temperature_var: str
    grid_type: GridType = "gaussian"
    variable_indices: dict[str, torch.Tensor] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    depth_indices: dict[str, torch.Tensor] = dataclasses.field(
        init=False, repr=False, compare=False
    )
    variables: list[str] = dataclasses.field(init=False, compare=False)
    depths: list[str] = dataclasses.field(init=False, compare=False)
    dz: torch.Tensor = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.depth_levels) != len(self.depth_thickness):
            raise ValueError(
                "depth_levels and depth_thickness must have the same length"
            )

        def split_channel(channel: str) -> tuple[str, str | None]:
            name, separator, suffix = channel.rpartition("_")
            return (name, suffix) if separator and suffix.isdigit() else (channel, None)

        split_channels = [
            split_channel(channel) for channel in self.prognostic_var_names
        ]
        var_set_2d = [name for name, depth in split_channels if depth is None]
        var_set = list(dict.fromkeys(name for name, _ in split_channels))
        depth_set = list(self.depth_i_levels[: self.num_prognostic_depth_levels])

        var_indices = {
            name: torch.tensor(
                [
                    index
                    for index, (channel_name, _) in enumerate(split_channels)
                    if name == channel_name
                ],
                dtype=torch.int32,
            )
            for name in var_set
        }
        depth_indices = {
            depth: torch.tensor(
                [
                    index
                    for index, (_, channel_depth) in enumerate(split_channels)
                    if channel_depth == depth
                ],
                dtype=torch.int32,
            )
            for depth in depth_set
        }
        if depth_set and var_set_2d:
            depth_indices[depth_set[0]] = torch.cat(
                [
                    depth_indices[depth_set[0]],
                    *(var_indices[name] for name in var_set_2d),
                ]
            )

        object.__setattr__(self, "variable_indices", var_indices)
        object.__setattr__(self, "depth_indices", depth_indices)
        object.__setattr__(self, "variables", var_set)
        object.__setattr__(self, "depths", depth_set)
        object.__setattr__(
            self,
            "dz",
            torch.tensor(self.depth_thickness[: self.num_prognostic_depth_levels]),
        )

    @property
    def depth_i_levels(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(len(self.depth_levels)))

    @property
    def num_prognostic_depth_levels(self) -> int:
        depth_indices = []
        for var_name in self.prognostic_var_names:
            _, sep, suffix = var_name.rpartition("_")
            if sep and suffix.isdigit():
                depth_indices.append(int(suffix))
        return max(depth_indices) + 1 if depth_indices else 1

    def to(self, device: torch.device) -> Self:
        """Return a copy whose tensor indices live on ``device``."""
        moved = dataclasses.replace(self)
        for mapping_name in ("variable_indices", "depth_indices"):
            mapping = getattr(moved, mapping_name)
            for key, value in mapping.items():
                mapping[key] = value.to(device)
        object.__setattr__(moved, "dz", moved.dz.to(device))
        return moved


# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

RHO_0 = 1035.0  # DENSITY_OF_WATER_CM4 kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER_CM4 J/kg/K


def _select_var_names(
    vars_by_key: dict[str, list[str]],
    key: str,
    dataset_type: str,
    var_kind: str,
) -> list[str]:
    try:
        return list(vars_by_key[key])
    except KeyError as exc:
        choices = ", ".join(sorted(vars_by_key))
        raise ValueError(
            f"Unsupported {dataset_type} {var_kind} variable key {key!r}. "
            f"Expected one of: {choices}"
        ) from exc


def build_om4_layout(
    prognostic_vars_key: str = "thermo_dynamic_all",
    boundary_vars_key: str = "tau_hfds",
    grid_type: GridType = "gaussian",
) -> DataLayout:
    return DataLayout(
        depth_levels=(
            2.5,
            10.0,
            22.5,
            40.0,
            65.0,
            105.0,
            165.0,
            250.0,
            375.0,
            550.0,
            775.0,
            1050.0,
            1400.0,
            1850.0,
            2400.0,
            3100.0,
            4000.0,
            5000.0,
            6000.0,
        ),
        depth_thickness=(
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            50.0,
            70.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            400.0,
            500.0,
            600.0,
            800.0,
            1000.0,
            1000.0,
            1000.0,
        ),
        prognostic_var_names=_select_var_names(
            {
                "thetao_1": ["thetao_0"],
                "thermo_dynamic_5": [
                    k + str(j)
                    for k in ["uo_", "vo_", "thetao_", "so_"]
                    for j in range(5)
                ]
                + ["zos"],
                "thermo_dynamic_all": [
                    k + str(j)
                    for k in ["uo_", "vo_", "thetao_", "so_"]
                    for j in range(19)
                ]
                + ["zos"],
                "thermo_all": [
                    k + str(j) for k in ["thetao_", "so_"] for j in range(19)
                ]
                + ["zos"],
            },
            prognostic_vars_key,
            "om4",
            "prognostic",
        ),
        boundary_var_names=_select_var_names(
            {
                "hfds": ["hfds"],
                "tau_hfds": ["tauuo", "tauvo", "hfds"],
                "tau_hfds_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"],
            },
            boundary_vars_key,
            "om4",
            "boundary",
        ),
        default_metadata={
            "thetao": {
                "long_name": "Sea Water Potential Temperature",
                "units": r"\degree C",
            },
            "so": {
                "long_name": "Sea Water Salinity",
                "units": "psu",
            },
            "uo": {
                "long_name": "Sea Water X Velocity",
                "units": "m/s",
            },
            "vo": {
                "long_name": "Sea Water Y Velocity",
                "units": "m/s",
            },
            "zos": {
                "long_name": "Sea surface height above geoid",
                "units": "m",
            },
            "tos": {
                "long_name": "Sea surface temperature",
                "units": r"\degree C",
            },
            "tauuo": {
                "long_name": "Surface Downward X Stress",
                "units": "N/m^2",
            },
            "tauvo": {
                "long_name": "Surface Downward Y Stress",
                "units": "N/m^2",
            },
            "hfds": {
                "long_name": "Surface ocean heat flux from "
                "SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat",
                "units": "W/m^2",
            },
            "hfds_anomalies": {
                "long_name": "hfds anomalies",
                "units": "W/m^2",
            },
        },
        ocean_heat_temperature_var="thetao",
        grid_type=grid_type,
    )


def build_llc_layout(
    prognostic_vars_key: str = "single_1",
    boundary_vars_key: str = "single_1",
) -> DataLayout:
    return DataLayout(
        depth_levels=(
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
        ),
        depth_thickness=(
            1.07,
            1.22,
            1.395,
            1.595,
            1.815,
            2.065,
            2.35,
            2.67,
            3.025,
            3.42,
            3.865,
            4.355,
            4.895,
            5.485,
            6.13,
            6.835,
            7.59,
            8.405,
            9.28,
            10.21,
            11.195,
            12.23,
            13.315,
            14.45,
            15.625,
            16.835,
            18.075,
            19.335,
            20.62,
            21.915,
            23.215,
            24.52,
            25.815,
            27.1,
            28.37,
            29.62,
            30.845,
            32.05,
            33.235,
            34.395,
            35.53,
            36.645,
            37.745,
            38.83,
            39.91,
            40.99,
            42.075,
            43.175,
            44.3,
            45.46,
            54.405,
        ),
        prognostic_var_names=_select_var_names(
            {
                "single_1": ["Theta_0"],
                "all": [
                    k + str(j)
                    for k in ["U_", "V_", "Theta_", "Salt_"]
                    for j in range(51)
                ]
                + ["Eta"],
            },
            prognostic_vars_key,
            "llc",
            "prognostic",
        ),
        boundary_var_names=_select_var_names(
            {
                "single_1": ["oceQnet"],
                "all": ["oceTAUX", "oceTAUY", "oceQnet", "Eta"],
            },
            boundary_vars_key,
            "llc",
            "boundary",
        ),
        default_metadata={
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
        },
        ocean_heat_temperature_var="Theta",
        # LLC (lat-lon-cap) is curvilinear, so its 2D geometry can't be broadcast
        # from 1D axes -- same broadcast-unsafe class as the tripolar grid.
        grid_type="tripolar",
    )


def construct_metadata(
    data: xr.Dataset,
    data_layout: DataLayout,
) -> dict[str, dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[str(var)] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in data_layout.default_metadata.keys():
                metadata[str(var)] = data_layout.default_metadata[str(var)]
            elif (key := str(var).split("_")[0]) in data_layout.default_metadata.keys():
                metadata[str(var)] = data_layout.default_metadata[key]
            else:
                logger.info(f"{var} does not have any default metadata")
                metadata[str(var)] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class LoaderVersion(enum.Enum):
    OM4_TORCH = "om4-torch"
