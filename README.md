<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra

[![Open Quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m2lines/Samudra/blob/main/notebooks/quickstart.ipynb)

[![Pre-commit](https://github.com/m2lines/Samudra/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/m2lines/Samudra/actions/workflows/pre-commit.yml)
[![Test CPU](https://github.com/m2lines/Samudra/actions/workflows/test.yml/badge.svg)](https://github.com/m2lines/Samudra/actions/workflows/test.yml)
[![Benchmark CPU](https://github.com/m2lines/Samudra/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/m2lines/Samudra/actions/workflows/benchmarks.yml)
[Benchmark Results](https://m2lines.github.io/Samudra/dev/bench/)

Samudra is a global ocean emulator described in ["Samudra: An AI Global Ocean Emulator for Climate"](https://arxiv.org/abs/2412.03795) and updated in ["Samudra 2: Scaling Ocean Emulators across Resolutions"](https://m2lines.github.io/Samudra/samudra2/). Samudra efficiently emulates the ocean component of a state-of-the-art climate model, accurately reproducing key ocean variables including sea surface height, horizontal velocities, temperature, and salinity, across their full depth.

To try the complete Samudra 2 training workflow in a browser without HPC access or cloud
credentials, open the [free-tier Colab quickstart](https://colab.research.google.com/github/m2lines/Samudra/blob/main/notebooks/quickstart.ipynb).

![](/docs/static/assets/sst_tropical_pacific_ultra_small.gif "Sea Surface Temperature of the Tropical Pacific: Ground Truth vs Samudra v2")

> Ground truth (left) vs. Samudra 2 prediction (right) for sea surface temperature in the tropical Pacific.

We are actively and openly developing this emulator to support new tasks and data sources with the goal of building a broadly useful foundation model for ocean and climate. Please see [our docs](https://m2lines.github.io/Samudra/docs/) for more or [our contributing guide](https://m2lines.github.io/Samudra/docs/contributing/) to join in!

<p>
  <a href="https://www.nyu.edu/"><img src="/docs/static/assets/nyu-logo.png" alt="NYU" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.princeton.edu/"><img src="/docs/static/assets/princeton-university-logo.png" alt="Princeton University" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://openathena.ai/"><img src="/docs/static/assets/open-athena-logo.png" alt="Open Athena" /></a>
</p>
