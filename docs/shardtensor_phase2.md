# ShardTensor Phase 2: curriculum training

Phase 2 runs one logical Samudra sample across a single 2D GPU cluster. The
domain leader reads and prepares the full patch; `DomainParallelContext` then
scatters every input and label over H/W. Model parameters are replicated while
activations are spatially sharded.

## Supported surface

- One cluster, normally `cluster_shape: [2, 2]` on four GPUs.
- Non-replay curriculum training, including multi-step autoregressive batches.
- Constant model padding, custom shard-safe GroupNorm, AvgPool, and bilinear
  upsampling.
- `mse`, `mae`, or `mse_mae` loss.
- One-step validation. Validation outputs are gathered for the existing dense
  metric aggregators.
- Dense model checkpoints suitable for the existing single-GPU evaluator.

Replay, FSDP/multiple clusters, horizontal-gradient loss, rollout inference,
and checkpoint resume are intentionally rejected until their later phases.

## Configuration

Add or override the following training fields:

```yaml
backend: nccl
replay:
  enabled: false
inference_epochs: []
loss: mse
domain_parallel:
  enabled: true
  cluster_shape: [2, 2]
  use_fsdp: false
  leader_scatter: true
  strict_equivalence: false
```

For a four-GPU single-node run:

```bash
torchrun --standalone --nproc_per_node=4 -m ocean_emulators.train \
  configs/samudra_llc/train.yaml \
  --backend nccl \
  --domain_parallel.enabled true \
  --domain_parallel.cluster_shape '[2, 2]' \
  --replay.enabled false \
  --loss mse
```

The loaded global H/W must split evenly over the mesh and each local tile must
be divisible by 16 for the four U-Net levels. For a 2x2 mesh, this means each
global dimension must be divisible by 32. The deepest local tile must also be
at least eight cells wide for the dilation-8 halo; 320, 512, and 1088 are safe
examples. The trainer validates both constraints before building the model.

Start with a short all-ocean or ordinary small-patch run on four L40S GPUs. Log
the first several batch losses and compare their trend against a dense run with
the same seed, batch size, learning rate, and data order before scaling to the
1088x1088 H200 configuration.

The provided batch script uses the same overrides for both sides of that check:

```bash
sbatch JOBS/other/shardtensor_train_phase2.sh
sbatch --export=ALL,MODE=dense JOBS/other/shardtensor_train_phase2.sh
```

`MODE=dp` is the default four-rank run. `MODE=dense` uses one GPU from the same
allocation as the reference, which intentionally leaves the other three idle.
