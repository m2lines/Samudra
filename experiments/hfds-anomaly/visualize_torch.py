"""Native paired visualization using the resident Torch dataset."""

import dask

from samudra.viz.config import VizConfig, main

if __name__ == "__main__":
    cfg = VizConfig.from_yaml_and_cli()
    with dask.config.set(scheduler="synchronous"):
        main(cfg)
    (cfg.output_path / "COMPLETE.txt").write_text(
        "Native visualization finished successfully\n"
    )
