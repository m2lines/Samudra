# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic JSONL rows for the learned-inverse experiment funnel."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stage: str
    gate: str
    seed: int
    command: list[str]
    hypothesis: str


def synthetic_command(
    *,
    architecture: str,
    source_size: int,
    output_size: int,
    longitude_shift: float,
    learning_rate: float,
    seed: int,
    steps: int = 500,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/probe_perceiver_decoder.py",
        "--architecture",
        architecture,
        "--data-mode",
        "analytic",
        "--grid-size",
        str(source_size),
        "--output-longitude-shift",
        str(longitude_shift),
        "--learning-rate",
        str(learning_rate),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
    ]
    if output_size != source_size:
        command.extend(("--output-grid-size", str(output_size)))
    command.extend(("--device", "cpu"))
    return command


def identity_command(
    *,
    model_file: str,
    geometry: str,
    learning_rate: float,
    seed: int,
    run_id: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "samudra.identity",
        "configs/samudra_multi_om4/identity_1deg.yaml",
        f"--model=@configs/samudra_multi_om4/{model_file}",
        f"--model.encoder.geometry_mode={geometry}",
        f"--learning_rate={learning_rate}",
        "--epochs=20",
        f"--experiment.rand_seed={seed}",
        f"--experiment.name={run_id}",
    ]


def build_manifest(s0_learning_rate: float) -> list[RunSpec]:
    rows: list[RunSpec] = []
    for learning_rate in (3e-4, 1e-3, 3e-3):
        for seed in (0, 1):
            run_id = f"s0-lr-{learning_rate:g}-seed-{seed}"
            rows.append(
                RunSpec(
                    run_id=run_id,
                    stage="S0-learning-rate",
                    gate="screen",
                    seed=seed,
                    command=synthetic_command(
                        architecture="coordinate-resample-projection",
                        source_size=8,
                        output_size=8,
                        longitude_shift=0,
                        learning_rate=learning_rate,
                        seed=seed,
                    ),
                    hypothesis="Calibrate optimization on the physical-resampling control.",
                )
            )

    routes = {
        "same": (8, 8, 0.0),
        "up": (8, 16, 0.0),
        "down": (16, 8, 0.0),
        "shift": (8, 8, 0.5),
    }
    architectures = (
        "resample-projection",
        "coordinate-resample-projection",
        "anchored-cross-attention",
        "resample-attention-residual",
    )
    for architecture in architectures:
        for route, (source_size, output_size, shift) in routes.items():
            for seed in (0, 1):
                run_id = f"s0-{architecture}-{route}-seed-{seed}"
                rows.append(
                    RunSpec(
                        run_id=run_id,
                        stage="S0-architecture",
                        gate="screen",
                        seed=seed,
                        command=synthetic_command(
                            architecture=architecture,
                            source_size=source_size,
                            output_size=output_size,
                            longitude_shift=shift,
                            learning_rate=s0_learning_rate,
                            seed=seed,
                        ),
                        hypothesis="Separate coordinate routing, interpolation, and learned correction effects.",
                    )
                )

    model_by_decoder = {
        "base": "model_learned_inverse_resample.yaml",
        "hybrid": "model_learned_inverse_hybrid.yaml",
    }
    for learning_rate in (3e-4, 6e-4, 1e-3):
        run_id = f"s1-lr-{learning_rate:g}-seed-15"
        rows.append(
            RunSpec(
                run_id=run_id,
                stage="S1-learning-rate",
                gate="screen",
                seed=15,
                command=identity_command(
                    model_file=model_by_decoder["base"],
                    geometry="additive",
                    learning_rate=learning_rate,
                    seed=15,
                    run_id=run_id,
                ),
                hypothesis="Calibrate the learned ocean encoder/base decoder jointly.",
            )
        )
    for decoder_name, model_file in model_by_decoder.items():
        for geometry in ("additive", "none"):
            run_id = f"s1-{decoder_name}-{geometry}-seed-15"
            rows.append(
                RunSpec(
                    run_id=run_id,
                    stage="S1-geometry-decoder",
                    gate="screen",
                    seed=15,
                    command=identity_command(
                        model_file=model_file,
                        geometry=geometry,
                        learning_rate=6e-4,
                        seed=15,
                        run_id=run_id,
                    ),
                    hypothesis="Measure held-out effects of additive encoder geometry and the local correction.",
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-learning-rate", type=float, default=3e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    content = "\n".join(
        json.dumps(asdict(row), sort_keys=True)
        for row in build_manifest(args.s0_learning_rate)
    )
    if args.output is None:
        print(content)
    else:
        args.output.write_text(content + "\n")


if __name__ == "__main__":
    main()
