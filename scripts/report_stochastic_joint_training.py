# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit generator/critic training scales from the completed joint-noise screen."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    "B-learned-control",
    "B-joint-coarse",
    "B-joint-multires",
    "B-conditional-gan",
]
LABELS = [
    "Learned-noise control",
    "Joint coarse",
    "Joint multiresolution",
    "Conditional GAN",
]


def read_events(root):
    steps, critic, sources = [], [], []
    for arm in ARMS:
        path = root / arm / "events.jsonl"
        data = path.read_bytes()
        sources.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
        records = [json.loads(line) for line in data.splitlines()]
        if [r["step"] for r in records] != list(range(1, 2001)):
            raise ValueError(f"Missing/duplicate generator updates: {arm}")
        for r in records:
            step = r["step"]
            expected = "observation" if step % 2 else "om4"
            if r["source"] != expected:
                raise ValueError(f"Unexpected source schedule: {arm}/{step}")
            monthly, native, initial = r["components"]
            row = dict(
                arm=arm,
                step=step,
                source=r["source"],
                loss=r["loss"],
                monthly=monthly,
                native=native,
                initializer=initial,
                weighted_initializer=0.1 * initial,
                weighted_forecast=0.5 * (monthly + native),
                generator_preclip_gradient_norm=r["grad_norm"],
                seconds=r["seconds"],
                data_wait_seconds=r["data_wait_seconds"],
                critic_seconds=r.get("gan_critic_seconds", 0),
                critic_data_wait_seconds=r.get("gan_critic_data_wait_seconds", 0),
            )
            if not np.isfinite(
                [v for v in row.values() if isinstance(v, (int, float))]
            ).all():
                raise ValueError(f"Nonfinite training log: {arm}/{step}")
            np.testing.assert_allclose(
                row["loss"],
                row["weighted_initializer"] + row["weighted_forecast"],
                rtol=2e-6,
                atol=1e-5,
            )
            steps.append(row)
            batch_records = r.get("gan_critic_records", [])
            if arm == ARMS[-1]:
                if len(batch_records) != (5 if expected == "om4" else 0):
                    raise ValueError(f"Wrong critic schedule: {step}")
                if (
                    r["gan_critic_updates"] != step // 2 * 5
                    or r["gan_critic_examples"] != step // 2 * 40
                ):
                    raise ValueError(f"Wrong critic exposure: {step}")
            elif batch_records:
                raise ValueError("Critic records on non-GAN arm")
            for offset, value in enumerate(batch_records):
                entry = dict(
                    generator_step=step,
                    critic_update=(step // 2 - 1) * 5 + offset + 1,
                    **value,
                )
                entry["mixed_minus_fake"] = entry["mixed_score"] - entry["fake_score"]
                entry["common_score_offset"] = 0.5 * (
                    entry["mixed_score"] + entry["fake_score"]
                )
                entry["weighted_gp"] = 10 * entry["gradient_penalty"]
                entry["weighted_drift"] = 0.001 * entry["drift"]
                np.testing.assert_allclose(
                    entry["wasserstein"],
                    -entry["mixed_minus_fake"],
                    rtol=1e-5,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    entry["critic_loss"],
                    entry["wasserstein"]
                    + entry["weighted_gp"]
                    + entry["weighted_drift"],
                    rtol=1e-5,
                    atol=1e-4,
                )
                critic.append(entry)
    if len(critic) != 5000:
        raise ValueError("Expected exactly 5000 critic optimizer updates")
    return pd.DataFrame(steps), pd.DataFrame(critic), sources


def rolling(ax, frame, x, y, *, label=None, color=None, width=50):
    ax.plot(
        frame[x],
        frame[y].rolling(width, min_periods=1).mean(),
        label=label,
        color=color,
    )


def figures(steps, critic, output):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for arm, label, color in zip(ARMS, LABELS, colors, strict=True):
        frame = steps[steps.arm == arm]
        obs, om4 = frame[frame.source == "observation"], frame[frame.source == "om4"]
        rolling(axes[0, 0], obs, "step", "loss", label=label, color=color)
        rolling(axes[0, 1], om4, "step", "weighted_forecast", label=label, color=color)
        rolling(
            axes[1, 0],
            obs,
            "step",
            "generator_preclip_gradient_norm",
            label=label,
            color=color,
        )
        rolling(
            axes[1, 1],
            om4,
            "step",
            "generator_preclip_gradient_norm",
            label=label,
            color=color,
        )
        if arm != ARMS[-1]:
            rolling(
                axes[0, 2],
                om4,
                "step",
                "weighted_initializer",
                label=label,
                color=color,
            )
    gan = steps[(steps.arm == ARMS[-1]) & (steps.source == "om4")]
    rolling(
        axes[1, 2],
        gan,
        "step",
        "weighted_initializer",
        label="0.1 × adversarial initializer",
        color=colors[-1],
    )
    rolling(
        axes[1, 2],
        gan,
        "step",
        "weighted_forecast",
        label="0.5 × (monthly + native CRPS)",
        color="black",
    )
    titles = [
        "Observation training loss",
        "OM4 forecast contribution (same objective)",
        "OM4 initializer CRPS contribution",
        "Observation G gradient norm (pre-clip)",
        "OM4 G gradient norm (pre-clip)",
        "GAN OM4 objective contributions",
    ]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set(title=title, xlabel="Generator optimizer updates")
        ax.grid(alpha=0.2)
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 2].legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "50 same-source update rolling means; GAN scalar loss is not a forecast score"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"generator-training.{ext}", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    rolling(axes[0, 0], critic, "generator_step", "gradient_norm", width=250)
    axes[0, 0].axhline(1, color="black", linestyle="--", label="GP target = 1")
    axes[0, 0].set(title="Critic input-gradient norm", yscale="log")
    axes[0, 0].legend()
    for key, label in [
        ("fake_score", "Generated pair"),
        ("mixed_score", "Real/generated pair"),
        ("common_score_offset", "Common score offset"),
    ]:
        rolling(axes[0, 1], critic, "generator_step", key, label=label, width=250)
    axes[0, 1].set(title="Critic scores and their shared offset")
    axes[0, 1].legend(fontsize=8)
    for key, label in [
        ("wasserstein", "Fake − mixed score"),
        ("weighted_gp", "10 × gradient penalty"),
        ("weighted_drift", "0.001 × drift"),
    ]:
        rolling(axes[1, 0], critic, "generator_step", key, label=label, width=250)
    axes[1, 0].set(title="Critic objective contributions", yscale="symlog")
    axes[1, 0].legend(fontsize=8)
    rolling(axes[1, 1], critic, "generator_step", "mixed_minus_fake", width=250)
    axes[1, 1].set(title="Mixed − fake score gap (offset cancels)")
    for ax in axes.flat:
        ax.set_xlabel("Generator optimizer updates")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "GAN critic: 250 critic-update rolling means; five critic updates per OM4 generator update"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"critic-training.{ext}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    steps, critic, sources = read_events(args.production)
    steps.to_csv(args.output / "generator-by-update.csv", index=False)
    critic.to_csv(args.output / "critic-by-update.csv", index=False)
    rows = []
    for end in (500, 1000, 1500, 2000):
        start = end - 499
        for (arm, source), frame in steps[
            (steps.step >= start) & (steps.step <= end)
        ].groupby(["arm", "source"]):
            row = dict(
                arm=arm,
                source=source,
                first_generator_step=start,
                last_generator_step=end,
                updates=len(frame),
            )
            for key in (
                "loss",
                "monthly",
                "native",
                "initializer",
                "weighted_initializer",
                "weighted_forecast",
                "generator_preclip_gradient_norm",
                "seconds",
                "data_wait_seconds",
                "critic_seconds",
                "critic_data_wait_seconds",
            ):
                row[key + "_mean"] = frame[key].mean()
            row["generator_preclip_gradient_norm_max"] = (
                frame.generator_preclip_gradient_norm.max()
            )
            row["fraction_generator_updates_clipped"] = (
                frame.generator_preclip_gradient_norm > 1
            ).mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.output / "generator-window-summary.csv", index=False)
    window = critic.assign(window_end=((critic.generator_step - 1) // 500 + 1) * 500)
    summary = window.groupby("window_end").agg(
        {
            key: ["mean", "max"]
            for key in [
                "gradient_norm",
                "gradient_penalty",
                "fake_score",
                "mixed_score",
                "common_score_offset",
                "mixed_minus_fake",
                "weighted_gp",
                "weighted_drift",
                "critic_loss",
            ]
        }
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(args.output / "critic-window-summary.csv")
    figures(steps, critic, args.output)
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                complete=True,
                sources=sources,
                generator_updates_per_arm=2000,
                critic_updates=5000,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations="Training log summaries; aggregate gradient norms do not identify the separate adversarial gradient contribution. Critic score offsets are not forecast error.",
            ),
            indent=2,
        )
        + "\n"
    )
    for path in args.output.iterdir():
        if path.is_file() and path.suffix != ".license":
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )


if __name__ == "__main__":
    main()
