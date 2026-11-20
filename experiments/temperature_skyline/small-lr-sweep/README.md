<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small classifier learning-rate sweep from scratch

Three fresh all-fields classifiers train at peak learning rates 0.0002, 0.0004 and 0.0008. Each warms up linearly for 100 updates, holds its peak through update 25,000, then cools down with a cosine schedule over 5,000 updates to the same absolute rate of 0.00002. The experiment therefore runs 30,000 updates per arm. The common final rate is a design choice for this sweep.

The model matches the earlier small all-fields classifier's 18.52M parameters, width 512, four transformer blocks, eight heads and 17×17 native quarter-degree crop. Inputs are continuous, and scalar outputs use argmax over 8,192 channel-specific fixed centers. Two-hot training loss and Gaussian-smoothed head initialization with sigma four bins are retained. The current NumPy CPU initializer avoids the previously observed Torch convolution crash. Its formula matches the historical initializer up to floating-point rounding.

All arms use seed 20261110, one RTX6000 GPU, effective batch 256, logical microbatch 32 with eight accumulation steps, physical microbatch 64 with four steps, and prefetched batches. Each loaded time window supplies 2,048 accepted crops. The original coverage proposal, EMA, masks, normalization and four-date validation samples are retained. RTX6000 hardware is held fixed to keep the learning-rate comparison and timing comparable across arms.

Raw and EMA validation runs at initialization and updates 1, 50, 100, 200, 500, 750, then every 1,000 updates through 30,000. Full optimizer/EMA/RNG/frame-stream checkpoints are retained at initialization, 10k, 25k and 30k. Every validation retains its model weights. The fixed 30k EMA checkpoint is the primary endpoint, with 25k results showing behavior immediately before cooldown. The historical small 25k run used cosine decay throughout training and supplies a schedule benchmark.

A short qualification runs all three rates for 12 updates with the production data pipeline. Tests cover warmup and cooldown boundaries, restart across cooldown, immutable checkpoint retention and the existing branch and soft-prior paths. Reporting will include matched per-field/depth MSE, hard-bin NLL and persistence curves and CSVs. This is a single-seed experiment/results record and is not intended to merge.
