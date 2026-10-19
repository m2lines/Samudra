"""Independently verify completed full-output fits using raw arrays on CPUs."""

import argparse
import json
from pathlib import Path

import numpy as np


def highpass(values):
    return values - sum(
        np.roll(values, shift, axis)
        for axis, shift in ((-2, -1), (-2, 1), (-1, -1), (-1, 1))
    ) / 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arms", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, default=5000)
    args = parser.parse_args()
    prepared = np.load(args.data / "prognostic.npy", mmap_mode="r")
    truth = prepared[2].astype(np.float64)
    previous = prepared[1].astype(np.float64)
    wet = np.load(args.data / "mask.npy").astype(bool)
    assert truth.shape == wet.shape == (77, 180, 360)
    inside = wet.copy()
    for axis, shift in ((-2, -1), (-2, 1), (-1, -1), (-1, 1)):
        inside &= np.roll(wet, shift, axis)
    inside[:, (0, -1), :] = False
    target_hp = highpass(truth)
    target_hp_mse = (target_hp**2 * inside).sum((-2, -1)) / inside.sum((-2, -1))
    assert np.all(target_hp_mse > 0)
    increment = truth - previous
    increment_hp = highpass(increment)
    selected = {13: "U1850", 32: "V1850", 51: "T1850", 70: "S1850", 76: "SSH"}
    patch = (slice(50, 82), slice(200, 232))
    arms = sorted(path.parent for path in args.root.glob("*/COMPLETE"))
    assert len(arms) == args.expected_arms, (len(arms), args.expected_arms)
    results = {}
    for arm in arms:
        protocol = json.loads((arm / "protocol.json").read_text())
        assert protocol["supervision"] in ("first target, all 77 channels", "all",
            "first target all77 channels; two input timestamps; train-mode BatchNorm; no temporal feedback")
        if protocol["supervision"] == "all":
            assert protocol["training_channel_slice"] == [None, None]
        history = [json.loads(line) for line in (arm / "history.jsonl").read_text().splitlines()]
        end = history[-1]
        assert end["step"] == protocol["steps"] == args.expected_steps
        with np.load(arm / f"maps-{args.expected_steps}.npz") as saved:
            assert np.array_equal(saved["truth"][0], truth)
            assert np.array_equal(saved["mask"], wet)
            prediction = saved["prediction"][0].astype(np.float64)
        assert np.isfinite(prediction).all()
        error = prediction - truth
        hp_error = highpass(error)
        channel_mse = (error**2 * wet).sum((-2, -1)) / wet.sum((-2, -1))
        channel_hp = (hp_error**2 * inside).sum((-2, -1)) / inside.sum((-2, -1))
        np.testing.assert_allclose(channel_mse, end["per_channel_mse"], rtol=3e-6, atol=1e-12)
        np.testing.assert_allclose(channel_mse.mean(), end["eval_mse"], rtol=3e-6)
        np.testing.assert_allclose(channel_hp.mean(), end["highpass_mse"], rtol=3e-6)
        fields = {}
        for channel, name in selected.items():
            assert inside[channel][patch].all()
            fields[name] = {}
            for domain, selection in (("global", inside[channel]), ("patch", patch)):
                e, h, t, d, hd = [
                    values[channel][selection]
                    for values in (error, hp_error, target_hp, increment, increment_hp)
                ]
                fields[name][domain] = {
                    "mse": float(np.mean(e**2)),
                    "highpass_mse": float(np.mean(h**2)),
                    "hp_error_over_target_rms": float(np.sqrt(np.mean(h**2) / np.mean(t**2))),
                    "hp_error_over_increment_rms": float(np.sqrt(np.mean(h**2) / np.mean(hd**2))),
                    "mse_over_persistence": float(np.mean(e**2) / np.mean(d**2)),
                }
        results[arm.name] = {
            "state_mse": float(channel_mse.mean()),
            "highpass_mse": float(channel_hp.mean()),
            "per_channel_mse": channel_mse.tolist(),
            "per_channel_highpass_mse": channel_hp.tolist(),
            "per_channel_hp_error_over_target_rms": np.sqrt(channel_hp / target_hp_mse).tolist(),
            "selected_fields": fields,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "verification": "CPU float64 calculations on raw saved grids; exact prepared target/mask "
                        "identity; all 77 channel errors and aggregate metrics reproduce logs. "
                        "Global selected-field metrics exclude mask edges and latitude endpoints. "
                        "Patch rows50:82 cols200:232. No interpolation or smoothing.",
        "arms": results,
    }, indent=2) + "\n")
    print(f"Verified {len(results)} full-output endpoints")


if __name__ == "__main__":
    main()
