<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Running Samudra

## Training the model

To train the model on a single GPU, you can run:

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
uv run -m samudra.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH --experiment.name <my-experiment-name>
```

Unless you override `--experiment.output_dir`, this will write to a `.LOCAL` directory.
You can run `uv run -m samudra.train --help` to see all the options available.

To train on multiple GPUs, you can use skypilot, `torchrun`, or SLURM.

### SkyPilot

If you use a model that requires Flash Attention, make sure to install the `cuda` extra first, like so:

```bash
uv sync --extra cuda
```
Of course, this will only work on CUDA-enabled machines.

To run a remote training job with SkyPilot, use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/train.sky.yaml  --env WANDB_API_KEY --env-file <my-vars>.env --env NAME <my-experiment-name> --env CONFIG configs/samudra_om4/train.yaml
```

Please read the docstring in the `train.sky.yaml` for more information.

### torchrun

To use torchrun on a single host with 8 GPUs, use something like:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 python -m samudra.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH
```

See the [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html) for other examples.

### SLURM

For SLURM, you want to allocate the same number of tasks to a given node as you have allocated GPUs to that *node* (not task).
You want to avoid using `--gpus-per-task` or `--gpu-bind` as it restricts the GPU's visibility to a given task, which
prevents cross-GPU communication. So you want something like (for 2 nodes with 4 GPUs each):

```bash
srun --nodes=2 --ntasks-per-node=4 --gres=gpu:4 -- uv run python -m samudra.train configs/samudra_om4/train.yaml --experiment.data_root $DATA_PATH
```

Each task will see all GPUs on the node, but they know how to choose the correct one for their work.

To learn more about other datasets used during training, please see the [data documentation](data.md).

## Evaluating the model

```bash
DATA_PATH=path/to/save/data
uv run scripts/clone_data.py $DATA_PATH
# (then put a checkpoint of the model at path/to/checkpoint)
uv run -m samudra.eval configs/samudra_om4/eval.yaml --ckpt_path path/to/checkpoint --eval.data_root $DATA_PATH --experiment.name <my-experiment-name>-eval
```

This produces a `predictions.zarr` file in the output directory (by default `.LOCAL`) with the rollout of the model.

You can run `uv run -m samudra.eval --help` to see all the options available.

To learn more about other datasets used during training, please see the [data documentation](data.md).

To run a remote training job with SkyPilot, use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/eval.sky.yaml  --env WANDB_API_KEY --env-file <my-vars>.env --env NAME <my-experiment-name>-eval --env CONFIG configs/samudra_om4/eval.yaml
```

Please read the `eval.sky.yaml` docstring for more information.

## Visualizing outputs from the model

```bash
uv run -m samudra.viz configs/viz_om4.yaml --data_root path/to/data --name <my-experiment-name>-viz --runs='[{"name": "my_experiment", "location": "path/to/<my-experiment-name>-eval/predictions.zarr"}]'
```

You can run `uv run -m samudra.viz --help` to see all the options available.

After making changes to the visualization code, you can run the following command to compare old and new plots:

```bash
uv run -m samudra.utils.compare path/to/old/viz path/to/new/viz
```

To run a remote viz job with SkyPilot, please use the following command:

```shell
# export WANDB_API_KEY=<my-key>  # Get your key at https://wandb.ai/authorize
uv run sky launch skypilot/viz.sky.yaml \
  --env WANDB_API_KEY \
  --env-file <my-vars>.env \
  --env NAME=<my-experiment-name>-viz \
  --env BASIN_PATH=basin_masks_original.zarr \
  --env RUNS='[{"name": "my_experiment", "location": "/inputs/<my-experiment-name>-eval/predictions.zarr"}]'

```

## Managing SkyPilot Clusters

All of the `sky launch` commands above will create a 1-node cluster with the needed
resources for that job. You can then run (or queue) additional jobs on that same cluster by passing
its name to `sky exec` commands:

```shell
uv run sky exec -c my-cluster-name skypilot/eval.sky.yaml ...
```

SkyPilot will complain if you try to use a cluster with the wrong resources for your job.
Note that we didn't use `sky launch` for this. The `launch` command sets up the cluster
from scratch again, which can break running jobs. Even when using `sky exec`, your local directory
is *immediately* copied up to the cluster which means other jobs running on it will
immediately see that new code. So, we recommend you not change code versions or other local
files before running another job.

When you're done with the cluster you can shut it down:

```shell
uv run sky down my-cluster-name
```

If you like, you can also have it automatically take itself down after it becomes idle:

```shell
# shut down after 30 minutes of idleness
uv run sky autostop --down my-cluster-name -i 30
```

See the [SkyPilot docs](https://docs.skypilot.co/) for more.

## samudra-multi Model

The samudra-multi model (in configs/samudra_multi_om4/) requires Flash Attention. Make sure to install the `cuda` extra first, like so:

```bash
uv sync --extra cuda
```

Of course, this will only work on CUDA-enabled machines.

You can then train/eval/etc as described above using the `configs/samudra_multi_om4/*` files.
