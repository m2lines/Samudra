"""Run the paired native visualization suite on a verified local source."""

from local_cache import stage

from samudra.utils.location import LocalLocation
from samudra.viz.config import VizConfig, main

if __name__ == "__main__":
    cfg = VizConfig.from_yaml_and_cli()
    cfg.data.sources[0].data_location = LocalLocation(path=stage())
    main(cfg)
    (cfg.output_path / "COMPLETE.txt").write_text(
        "Native visualization finished successfully\n"
    )
