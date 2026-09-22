#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate wave-three audits, comparison tables and figures from packed bytes."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts", default="docs/experiments/surface-wave3-results/artifacts"
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source, output = Path(a.artifacts).resolve(), Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run(script, *args):
        subprocess.run(
            [sys.executable, "scripts/" + script, *map(str, args)], check=True
        )

    run("pack_surface_adaptation_results.py", "--output", source, "--verify-only")
    run(
        "audit_initializer_wave.py",
        "--raw",
        source / "raw",
        "--output",
        output / "analysis",
    )
    run(
        "summarize_initializer_diagnostics.py",
        "--raw",
        source / "raw",
        "--analysis",
        output / "analysis",
    )
    for reference in ["A", "C", "A-s2718", "A-joint"]:
        run(
            "compare_initializer_wave.py",
            "--origins",
            output / "analysis/origins.csv",
            "--reference",
            reference,
            "--output",
            output / f"analysis/paired-vs-{reference}.csv",
        )
    run(
        "audit_initializer_batch_reference.py",
        "--current",
        output / "analysis",
        "--matched",
        source / "reference-analysis",
        "--previous",
        "docs/experiments/surface-wave2-results/raw/A",
        "--current-raw",
        source / "raw",
        "--matched-raw",
        source / "reference-raw",
        "--output",
        output / "batch-reference-audit.json",
    )
    run(
        "plot_initializer_wave.py",
        "--analysis",
        output / "analysis",
        "--output",
        output / "figures",
    )
    run(
        "plot_initializer_followups.py",
        "--summary",
        output / "analysis/summary.csv",
        "--output",
        output / "figures",
    )
    run(
        "plot_initializer_maps.py",
        "--raw",
        source / "raw",
        "--reference",
        "docs/experiments/surface-wave2-results/initialization/raw/snapshots.npz",
        "--output",
        output / "figures",
    )
    run(
        "plot_initializer_learning.py",
        "--raw",
        source / "training/primary",
        "--output",
        output / "figures",
        "--title",
        "Six primary initializers: completed training and validation",
    )
    learning = output / "learning-input"
    learning.mkdir(exist_ok=True)
    for group in ["primary", "stability"]:
        for directory in (source / "training" / group).iterdir():
            if directory.is_dir():
                link = learning / directory.name
                if not link.exists():
                    link.symlink_to(directory, target_is_directory=True)
    run(
        "plot_initializer_learning.py",
        "--raw",
        learning,
        "--output",
        output / "figures/learning",
        "--title",
        "Primary initializers and lower-rate attention follow-ups",
    )


if __name__ == "__main__":
    main()
