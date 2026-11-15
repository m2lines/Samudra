<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Constant learning-rate continuations

Three branches continue the final 25,000-update large all-fields classifier for another 3,000 updates at constant learning rates 0.0002, 0.0004 and 0.0008. These are 1×, 2× and 4× the original peak rate. The available full checkpoint is the final checkpoint after cosine cooldown. Each branch therefore restarts at its assigned constant rate after that cooldown.

All branches restore the same raw weights, optimizer moments, EMA state, per-rank NumPy and Torch RNG states, and frame-stream state. The original 216.37M-parameter architecture, 33×33 crop, four GPUs, effective batch 256, 128 crops per loaded frame per GPU and coverage proposal are retained. Each arm adds 768,000 examples, reaching 7,168,000 cumulative examples at global update 28,000. The original checkpoint remains the immutable branch source, identified by its exact SHA-256 in every configuration.

Validation uses the original eight dates and 64 points per depth per date. Both raw and EMA weights are evaluated before training and after 50, 100, 250, 500, 1,000, 2,000 and 3,000 additional updates. Primary reporting will use final-update EMA field/depth normalized MSE, with hard-bin NLL and persistence alongside the original 25k classifier and matched original regressor. Two-hot training remains enabled. Per-arm best tracking starts again at the branch boundary, while global optimizer and EMA update counts continue from 25,000.

A separate explicit branch loader verifies the parent checksum, source receipt, world size and scientific configuration before restoring state. Ordinary preemption resume retains exact configuration and source checks. The short qualification runs four updates for each rate before releasing the three production arms. This experiment uses one seed and a shared starting state. No full-grid inference is planned for this sweep.
