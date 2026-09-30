# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the observation target grid directly from Samudra's OM4 data source."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from samudra.config import DataConfig
from samudra.config_base import resolve_config_path
from samudra.observations.prepare import save_atomic
from samudra.utils.location import LocalLocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--om4", type=Path, required=True)
    parser.add_argument("--config", default="observation/om4.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DataConfig.model_validate(
        yaml.safe_load(resolve_config_path(args.config).read_text())
    )
    bundle = config.build(LocalLocation(path=args.om4))
    source = bundle.train_sources[0]
    names = bundle.data_layout.prognostic_var_names
    lat, lon = source.resolution
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(
        args.output,
        lat=lat.cpu().numpy(),
        lon=lon.cpu().numpy(),
        mask=source.masks.prognostic.cpu().numpy().astype(bool),
        mean=np.asarray(source.statistics(names).mean, dtype=np.float32),
        std=np.asarray(source.statistics(names).std, dtype=np.float32),
        names=np.array(names),
    )


if __name__ == "__main__":
    main()
