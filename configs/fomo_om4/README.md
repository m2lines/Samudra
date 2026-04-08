# FOMO OM4

These configurations are for the FOMO (encoder-processor-decoder) model,
trained on OM4 data. The intent is for this to support heterogeneous
resolutions and modalities.

## KR1: Multi-Scale Training (1/4°, 1/2°, 1°)

Train a single FOMO model on all three OM4 resolutions simultaneously using
a "match" schedule (each batch pairs input and label from the same resolution).

### Prerequisites

1. **Rebuild the container** from the `u/alxmrs/experiments/kr1` branch so that
   the updated configs (3 data sources + decoder config) are baked in:

   ```bash
   gh workflow run "Container PhysicsNeMo 25.11" --ref u/alxmrs/experiments/kr1
   # Wait ~12 min for the build, then note the SHA:
   git rev-parse HEAD  # e.g. dfc5abd9...
   ```

2. **Data on Torch** — all three resolutions under `/scratch/jr7309/data/`:

   ```
   /scratch/jr7309/data/
   ├── om4_quarterdeg_v2/   # 720×1440  (OM4.zarr, OM4_means.zarr, OM4_stds.zarr)
   ├── om4_halfdeg_v4/      # 360×720
   └── om4_onedeg_v3/       # 180×360
   ```

   Override with `export DATA_ROOT=/scratch/$USER/data` if you have a local copy.

3. **W&B** (optional) — `export WANDB_API_KEY=<key>` to enable online logging.

### Launch

```bash
# On torch, from the repo root:
export CONTAINER_HASH=<git_sha_from_step_1>
# export WANDB_API_KEY=<key>   # optional

bash scripts/launch_kr1_train.sh
```

This submits an 8×RTX6000 single-node job for 72 hours (70 epochs × 3 scales).

### What the launch script does

- Sets `CONFIG=configs/fomo_om4/train_multiscale.yaml` (baked into the container)
- Sets `DATA_ROOT=/scratch/$USER/data` (parent of all three resolution dirs)
- Sets `NAME_SUFFIX=kr1_fomo_multiscale` → run name `YYYY-MM-DD-kr1_fomo_multiscale`
- Passes `--batch_size=2` via ARGS

### Checkpoints

Output lands at `/scratch/$USER/runs/<run_name>/`. Checkpoints saved:

| Path | When |
|------|------|
| `saved_nets/latest_ckpt.pt` | Every epoch |
| `saved_nets/ema_ckpt.pt` | Every epoch (EMA weights) |
| `saved_nets/ckpt_epoch_N.pt` | Every 5 epochs |
| `saved_nets/best_val_ckpt.pt` | Best validation loss |

### Eval (after training)

```bash
export CONFIG=configs/fomo_om4/eval.yaml   # or a multiscale eval config
export NAME_SUFFIX=kr1_fomo_multiscale_eval
export TARGET_CHECKPOINT=<run_name>/saved_nets/ema_ckpt.pt
export CONTAINER_HASH=<same_sha>
export DATA_ROOT=/scratch/$USER/data

sbatch \
  --account=torch_pr_347_courant \
  --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --mem=128GB \
  --gres=gpu:rtx6000:1 \
  --time=04:00:00 \
  scripts/slurm_apptainer_eval.sbatch
```

### Model config summary

The FOMO model for this run uses reduced dimensions to fit 3 scales in memory
on 8×RTX6000 (24GB each):

| Component | Key params |
|-----------|-----------|
| **Encoder** (Perceiver) | depth=2, latent_dim=48, num_latents=128 |
| **Processor** (U-Net) | ch_width=[200,256,320], checkpointing=all |
| **Decoder** (PerceiverIO) | depth=2, latent_dim=48, num_latents=128, window_patches=4096 |
| **Global** | embedding_dim=128, bfloat16=true, batch_size=2 |
