import logging

from torch.utils.data import DistributedSampler

from ocean_emulators.config import TrainConfig
from ocean_emulators.train import Trainer
from ocean_emulators.utils.distributed import is_main_process
from ocean_emulators.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


def _validation_epoch(trainer: Trainer) -> int:
    # Completed checkpoints resume at the next epoch; incomplete emergency
    # checkpoints resume mid-epoch, so keep the current epoch label.
    if trainer.start_batch_in_epoch > 0:
        return trainer.start_epoch
    return max(1, trainer.start_epoch - 1)


def main():
    cfg = TrainConfig.from_yaml_and_cli()
    if cfg.resume_ckpt_path is None:
        raise ValueError(
            "resume_ckpt_path must be set; try --resume_ckpt_path=path/to/checkpoint"
        )

    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    trainer = Trainer(cfg)
    epoch = _validation_epoch(trainer)
    cur_step = trainer.get_current_step(epoch)
    cur_temporal_stride = trainer.get_current_temporal_stride(epoch)
    trainer.temporal_stride = cur_temporal_stride
    trainer.init_data_loaders(cur_step, cur_temporal_stride)

    if isinstance(trainer.val_sampler, DistributedSampler):
        trainer.val_sampler.set_epoch(epoch)

    try:
        logger.info(
            "Running standalone one-step validation for checkpoint %s at epoch label %s",
            cfg.resume_ckpt_path,
            epoch,
        )
        val_stats = trainer.validate_one_epoch(epoch)
        if is_main_process():
            trainer.wandb_logger.log({"epoch": epoch, **val_stats}, step=trainer.num_batches_seen)
        logger.info("Standalone validation summary: %s", val_stats)
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()
