<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2 2-degree architecture search

This experiment applies Samudra's successive-halving runner to 18 variants of
the native-SDPA Perceiver v2 model. Nine causal architecture hypotheses are
each evaluated at learning rates 0.0004 and 0.0008. The rungs allocate 1, 3, 6,
and 12 cumulative epochs, promoting half the candidates at each boundary and
retaining at least two finalists.

The first search holds the processor, data split, random seed, effective batch
size, loss, and forecast target fixed. It varies the decoder topology, encoder
latent count, decoder value-transport width, local context, and physical patch
extent. Validation normalized MSE is the promotion objective; epoch histories
and public artifacts support later diagnosis. Short rollout skill should be
used as a finalist check, not as an expensive inner-loop objective.

The controller uses `container_python.sh`, so controller and training jobs run
the exact same branch environment. At launch, export and pass the immutable SIF
and code layer produced for the experiment commit:

```bash
export SIF_PATH=/scratch/$USER/.apptainer-images/physicsnemo-26.05-COMMIT.sif
export CODE_LAYER=/scratch/$USER/.apptainer-code-layers/samudra-code-COMMIT.img

experiments/perceiver_search_2deg/container_python.sh \
  -m samudra.search experiments/perceiver_search_2deg/search.yaml \
  --executor.sif_path=/scratch/$USER/.apptainer-images/physicsnemo-26.05-COMMIT.sif \
  --executor.code_layer=/scratch/$USER/.apptainer-code-layers/samudra-code-COMMIT.img
```

The resolved config, exact code provenance, W&B identities, scheduler logs,
metrics, and retained finalist checkpoints are published beneath the generated
search run ID.
