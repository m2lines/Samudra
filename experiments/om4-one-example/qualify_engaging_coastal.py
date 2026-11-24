"""Verify the two-step GPU pilot against exact prepared data and source maps."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_coastal import components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (args.root / "INPUTS_VERIFIED").exists()
    results = {}
    for context, boundary, season in (("seed16_oct", "interior", "oct"), ("seed15_jan", "wet_neighbors", "jan")):
        arm = args.root / "pilot" / context / boundary
        assert (arm / "COMPLETE").exists()
        assert (arm / "latest.pt").stat().st_size > 0
        protocol = json.loads((arm / "protocol.json").read_text())
        assert protocol["steps"] == 2 and protocol["weight"] == 100
        assert protocol["boundary"] == boundary
        assert protocol["parameters"] == 84066474
        assert protocol["torch"] == "2.12.0a0+0291f960b6.nv26.04.48445190"
        assert "RTX PRO 6000 Blackwell" in protocol["gpu"]
        assert protocol["campaign_deadline_utc"] == "2026-10-19T19:00:00Z"
        with np.load(args.root / "source" / context / "maps-5000.npz") as source, np.load(arm / "maps-0.npz") as initial:
            assert set(source.files) == set(initial.files)
            assert all(np.array_equal(source[key], initial[key]) for key in source.files)
        history = [json.loads(line) for line in (arm / "history.jsonl").read_text().splitlines()]
        assert [row["step"] for row in history] == [0, 2]
        assert history[-1]["training_objective"] < history[0]["training_objective"]
        truth = np.load(args.root / "data" / season / "prognostic.npy", mmap_mode="r")[2]
        wet = np.load(args.root / "data" / season / "mask.npy").astype(bool)
        with np.load(arm / "maps-2.npz") as final:
            assert np.array_equal(final["truth"][0], truth)
            assert np.array_equal(final["mask"], wet)
            assert np.isfinite(final["prediction"]).all()
            values = components(final["prediction"][0], truth, wet)
        loss = values["state_mse"] + 100 * values["highpass_mse"]
        if boundary == "wet_neighbors":
            loss += 100 * values["extra_boundary_curvature_mse"]
        np.testing.assert_allclose(loss, history[-1]["training_objective"], rtol=1e-5)
        for key, log_key in (("state_mse", "eval_mse"), ("highpass_mse", "highpass_mse")):
            np.testing.assert_allclose(values[key], history[-1][log_key], rtol=3e-6)
        results[context] = {
            "boundary": boundary, "torch": protocol["torch"], "gpu": protocol["gpu"],
            "initial_source_maps_bitwise_equal": True, "prepared_target_mask_exact": True,
            "initial_objective": history[0]["training_objective"], "final_objective": loss,
            "state_mse": values["state_mse"], "highpass_mse": values["highpass_mse"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"qualification": results}, indent=2) + "\n")
    print("Both GPU pilots qualified; this is environment validation, not a scientific endpoint.")


if __name__ == "__main__":
    main()
