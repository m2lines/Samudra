"""Native trainer with RNG restoration after construction on preemptible restart."""

import random

import numpy as np
import torch

from samudra.config import TrainConfig
from samudra.train import Trainer
from samudra.utils.logging import handle_logging, handle_warnings


def main():
    from runtime import install

    install()
    cfg = TrainConfig.from_yaml_and_cli()
    from local_cache import stage

    from samudra.utils.location import LocalLocation

    cfg.data.sources[0].data_location = LocalLocation(path=stage())
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    trainer = Trainer(cfg)
    if cfg.resume_ckpt_path:
        state = torch.load(cfg.resume_ckpt_path, map_location="cpu")
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        random.setstate(state["rng_python"])
        ns = state["rng_numpy"]
        np.random.set_state(
            (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
        )
        print("RNG_RESTORED", trainer.start_epoch, trainer.num_batches_seen, flush=True)
    trainer.run()


if __name__ == "__main__":
    main()
