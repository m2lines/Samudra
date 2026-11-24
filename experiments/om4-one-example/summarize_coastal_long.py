"""Audit exact continuation boundaries and compare independently verified endpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--continued", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--continued-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoints = {
        1000: read_json(args.source_report / "verification.json")["arms"],
        2000: read_json(args.continued_report / "verification.json")["arms"],
    }
    objectives = {
        1000: read_json(args.source_report / "objective-verification.json")["arms"],
        2000: read_json(args.continued_report / "objective-verification.json")["arms"],
    }
    summary = {}
    trajectories = {}
    for arm in ("interior", "wet_neighbors"):
        source = args.source / arm
        continued = args.continued / arm
        assert (source / "COMPLETE").exists() and (continued / "COMPLETE").exists()
        assert (continued / "RESUME_COPY_VERIFIED").exists()
        source_lines = (source / "history.jsonl").read_text().splitlines()
        continued_lines = (continued / "history.jsonl").read_text().splitlines()
        assert continued_lines[:len(source_lines)] == source_lines
        history = [json.loads(line) for line in continued_lines]
        before, resumed = history[len(source_lines) - 1:len(source_lines) + 1]
        assert before["step"] == resumed["step"] == 1000
        assert history[-1]["step"] == 2000
        for key in ("closure_evaluations", "internal_iterations"):
            assert before[key] == resumed[key], key
        # A fresh forward/backward measurement must reproduce the checkpoint metrics.
        for key in ("eval_mse", "training_objective", "highpass_mse",
                    "objective_gradient_norm", "per_channel_mse"):
            np.testing.assert_allclose(resumed[key], before[key], rtol=3e-6, atol=1e-12)
        assert resumed["seconds"] >= before["seconds"]
        assert history[-1]["closure_evaluations"] > resumed["closure_evaluations"]
        provenance = read_json(continued / "resume-provenance.json")
        assert provenance["source_outer_step"] == 1000
        assert provenance["target_outer_step"] == 2000
        hashes = (continued / "checkpoint-copy.sha256").read_text().splitlines()
        assert len(hashes) == 2 and hashes[0].split()[0] == hashes[1].split()[0]
        for step in endpoints:
            for key in ("state_mse", "highpass_mse"):
                np.testing.assert_allclose(endpoints[step][arm][key], objectives[step][arm][key], rtol=1e-10)
        old, new = endpoints[1000][arm], endpoints[2000][arm]
        ratios = np.array(new["per_channel_mse"]) / old["per_channel_mse"]
        summary[arm] = {
            "resume_verified": "Exact history prefix and recorded checkpoint-copy hashes; restored optimizer counters and recomputed loss/gradient metrics match at step1000.",
            "state_mse_reduction_1000_to_2000": old["state_mse"] / new["state_mse"],
            "highpass_mse_reduction_1000_to_2000": old["highpass_mse"] / new["highpass_mse"],
            "channels_with_lower_state_mse": int((ratios < 1).sum()),
            "additional_closures": history[-1]["closure_evaluations"] - resumed["closure_evaluations"],
            "additional_measured_seconds": history[-1]["seconds"] - before["seconds"],
            "endpoints": {
                step: {
                    "state_mse": endpoints[step][arm]["state_mse"],
                    "highpass_mse": endpoints[step][arm]["highpass_mse"],
                    "quiet_patch": {
                        field: endpoints[step][arm]["selected_fields"][field]["patch"]
                        for field in ("T1850", "S1850", "SSH")
                    },
                } for step in endpoints
            },
        }
        trajectories[arm] = history
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "scope": "Same October seed16 single-example fits continued with intact L-BFGS state; no new independent example or generalization test. Endpoint metrics independently verified from raw arrays. Gradient norms are diagnostics, not convergence certificates.",
        "arms": summary,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for arm, rows in trajectories.items():
        for ax, key, title in zip(axes.flat,
            ("eval_mse", "highpass_mse", "training_objective", "objective_gradient_norm"),
            ("Wet-cell state MSE", "Interior high-pass error MSE", "Each arm's own training objective", "Each arm's own objective gradient norm")):
            x = [row["closure_evaluations"] for row in rows]
            ax.semilogy(x, [row[key] for row in rows], label=arm.replace("_", " "))
            start = next(row for row in rows if row["step"] == 1000)
            ax.scatter(start["closure_evaluations"], start[key], marker="o", s=35)
            ax.set_title(title)
            ax.set_xlabel("Cumulative L-BFGS closure evaluations")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Longer optimization; circles mark the 1000-step continuation boundary")
    fig.savefig(args.output / "convergence.png", dpi=160)
    plt.close(fig)
    print({arm: {key: value for key, value in values.items() if key != "endpoints"}
           for arm, values in summary.items()})


if __name__ == "__main__":
    main()
