<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Colab quickstart configuration

This directory contains a Samudra 2 training run sized for a free-tier Google
Colab T4 GPU. It pairs with `notebooks/quickstart.ipynb` and uses a public slice
of the 1° OM4 dataset without credentials. The model retains the production
Samudra 2 widths and dynamic loss; only the data depth, rollout length, batch
size, and training duration are reduced for the tutorial.

The notebook filters the remote store to `thermo_dynamic_5` and `tau_hfds`
before transfer and writes ten-time-step chunks, avoiding the unused deep-ocean
variables and one-file-per-time-step layout of a full clone.

The run is an end-to-end onboarding smoke test, not a production-quality ocean
emulator. Use `configs/samudra_om4_v2/` as the starting point for real training.
