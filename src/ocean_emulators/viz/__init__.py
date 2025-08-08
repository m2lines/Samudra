from pathlib import Path

from ocean_emulators.utils.location import LocalLocation
from ocean_emulators.viz.config import VizConfig
from ocean_emulators.viz.core import Viz

__all__ = ["Viz", "VizConfig", "main"]


def main(cfg: VizConfig):
    # TODO(jder): set up and use logging for all this, tee to file, maybe wandb?

    print(f"Writing results to {cfg.output_path}")
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(cfg.output_path / "config.yaml")

    viz = cfg.build(LocalLocation(path=Path.cwd()))

    # TODO(jder): would be nice to specify which plots to make,
    # parallelize, re-run just needed plots on errors, etc.
    viz.run()


if __name__ == "__main__":
    main(VizConfig.from_yaml_and_cli())
