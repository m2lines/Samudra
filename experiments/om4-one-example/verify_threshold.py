"""Verify the matched curvature-threshold intervention and saved physical fields."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_coastal import components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", choices=("interior", "wet_neighbors"), default="interior")
    parser.add_argument("--initial-steps", type=int, default=2000)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--map-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = np.load(args.data / "prognostic.npy", mmap_mode="r")[2]
    wet = np.load(args.data / "mask.npy").astype(bool)
    source_lines = (args.source / "history.jsonl").read_text().splitlines()
    source_end = json.loads(source_lines[-1])
    assert source_end["step"] == args.initial_steps
    map_results = json.loads(args.map_verification.read_text())["arms"]
    result = {}
    pair_hashes = []
    for arm, threshold in (("default", 1e-10), ("lowered", 1e-20)):
        path = args.root / arm
        assert (path / "COMPLETE").exists() and (path / "RESUME_COPY_VERIFIED").exists()
        protocol = json.loads((path / "protocol.json").read_text())
        assert protocol["curvature_threshold"] == threshold
        assert protocol["weight"] == 100 and protocol["boundary"] == args.boundary
        assert protocol["steps"] == 2500 and protocol["parameters"] == 84066474
        lines = (path / "history.jsonl").read_text().splitlines()
        assert lines[:len(source_lines)] == source_lines
        rows = [json.loads(line) for line in lines]
        resumed, end = rows[len(source_lines)], rows[-1]
        assert resumed["step"] == args.initial_steps and end["step"] == 2500
        for key in ("closure_evaluations", "internal_iterations"):
            assert resumed[key] == source_end[key]
        for key in ("eval_mse", "highpass_mse", "training_objective", "per_channel_mse"):
            np.testing.assert_allclose(resumed[key], source_end[key], rtol=3e-6, atol=1e-12)
        assert resumed["curvature_candidates"] == resumed["curvature_accepted"] == 0
        hashes = (path / "checkpoint-copy.sha256").read_text().splitlines()
        assert len(hashes) == 2 and hashes[0].split()[0] == hashes[1].split()[0]
        pair_hashes.append(hashes[0].split()[0])
        with np.load(path / f"maps-{args.initial_steps}.npz") as initial, np.load(args.source / f"maps-{args.initial_steps}.npz") as source:
            assert set(initial.files) == set(source.files)
            assert all(np.array_equal(initial[key], source[key]) for key in initial.files)
        with np.load(path / "maps-2500.npz") as saved:
            assert np.array_equal(saved["truth"][0], truth)
            assert np.array_equal(saved["mask"], wet)
            values = components(saved["prediction"][0], truth, wet)
        for key in ("state_mse", "highpass_mse"):
            np.testing.assert_allclose(values[key], map_results[arm][key], rtol=1e-10)
        objective = values["state_mse"] + 100 * values["highpass_mse"]
        if args.boundary == "wet_neighbors":
            objective += 100 * values["extra_boundary_curvature_mse"]
        np.testing.assert_allclose(objective, end["training_objective"], rtol=1e-5)
        assert end["curvature_candidates"] > 0
        result[arm] = {
            "threshold": threshold, "state_mse": values["state_mse"],
            "highpass_mse": values["highpass_mse"], "objective": objective,
            "curvature_candidates": end["curvature_candidates"],
            "curvature_accepted": end["curvature_accepted"],
            "curvature_acceptance_fraction": end["curvature_accepted"] / end["curvature_candidates"],
            "additional_closures": end["closure_evaluations"] - resumed["closure_evaluations"],
            "additional_measured_seconds": end["seconds"] - source_end["seconds"],
            "quiet_salinity": map_results[arm]["selected_fields"]["S1850"]["patch"],
        }
    assert pair_hashes[0] == pair_hashes[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "verification": "Identical copied checkpoint hashes, source history prefix, restored optimizer counters, recomputed starting errors; exact saved initial maps and prepared truth/mask; CPU float64 endpoint objectives agree with logs and independent all77 map verifier.",
        "scope": "Matched continuation of a single example; only curvature acceptance threshold differs. This is not a test of the earlier Adam plateau or rollout generalization.",
        "arms": result,
        "default_over_lowered_state_mse": result["default"]["state_mse"] / result["lowered"]["state_mse"],
        "default_over_lowered_highpass_mse": result["default"]["highpass_mse"] / result["lowered"]["highpass_mse"],
    }, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
