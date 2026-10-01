# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical endpoint members and reference-relative spatial roughness."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.observation_pilot import digest


def roughness(field, mask, weights):
    terms = []
    for axis in (-1, -2):
        # Longitude periodic; latitude is never wrapped across the poles.
        other = np.roll(field, -1, axis=axis)
        pairs = mask & np.roll(mask, -1, axis=axis)
        if axis == -2:
            pairs[-1] = False
        w = weights * pairs
        terms.append(np.sum(np.where(pairs, field - other, 0) ** 2 * w) / w.sum())
    return float(np.sqrt(np.mean(terms)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = {}
    contracts = {}
    metadata = {}
    for seed in (1729, 1730):
        for mode in ("obs-adapted", "om4-pretrained", "om4-adapted"):
            root = a.root / f"D-{seed}" / mode
            complete = json.loads((root / "COMPLETE.json").read_text())
            protocol = json.loads((root / "protocol.json").read_text())
            if complete["protocol"] != protocol or len(complete["files"]) != 3:
                raise ValueError("Incomplete endpoint contract")
            contracts[f"{seed}/{mode}"] = protocol
            for name, sha in complete["files"].items():
                path = root / name
                if digest(path) != sha:
                    raise ValueError("Endpoint checksum mismatch")
                sources[str(path)] = sha
                sidecar = path.with_suffix(".json")
                sources[str(sidecar)] = digest(sidecar)
                metadata[str(path)] = json.loads(sidecar.read_text())
                with np.load(path) as z:
                    z = dict(z)
                if z["members"].shape != (8, 2, 4, 180, 360):
                    raise ValueError("Endpoint layout differs")
                lat = z["lat"]
                weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
                year = name[:4]
                for j, lead in enumerate(z["leads_days"]):
                    for c, channel in enumerate(z["channels"]):
                        if mode == "obs-adapted" and c >= 2:
                            continue
                        members = z["members"][:, j, c]
                        truth = z["reference"][j, c]
                        mean = members.mean(0)
                        mask = (
                            z["mask"][c].astype(bool)
                            & np.isfinite(truth)
                            & (np.abs(lat[:, None]) <= 60)
                        )
                        w = weights * mask

                        def rms(x):
                            return float(
                                np.sqrt(np.sum(np.where(mask, x, 0) ** 2 * w) / w.sum())
                            )

                        tr = roughness(truth, mask, weights)
                        rows.append(
                            dict(
                                seed=seed,
                                mode=mode,
                                year=year,
                                lead_days=int(lead),
                                channel=str(channel),
                                mean_rmse=rms(mean - truth),
                                member1_rmse=rms(members[0] - truth),
                                member_deviation_rms=float(
                                    np.sqrt(
                                        np.mean([rms(m - mean) ** 2 for m in members])
                                    )
                                ),
                                reference_neighbor_rms=tr,
                                mean_neighbor_rms=roughness(mean, mask, weights),
                                member_neighbor_rms=float(
                                    np.sqrt(
                                        np.mean(
                                            [
                                                roughness(m, mask, weights) ** 2
                                                for m in members
                                            ]
                                        )
                                    )
                                ),
                            )
                        )
                        if mode == "om4-pretrained":
                            continue
                        if mode == "obs-adapted":
                            fields = [truth, mean, *members[:4]]
                            titles = [
                                "Observations, matching five-day bin",
                                "Ensemble mean (8)",
                                *[f"Member {i}" for i in range(1, 5)],
                            ]
                            label = "Observation history + ERA5, adapted checkpoint"
                        else:
                            before_path = a.root / f"D-{seed}" / "om4-pretrained" / name
                            with np.load(before_path) as before:
                                np.testing.assert_array_equal(
                                    before["reference"], z["reference"]
                                )
                                pre = before["members"][:, j, c]
                            fields = [
                                truth,
                                pre.mean(0),
                                pre[0],
                                mean,
                                members[0],
                                members[1],
                            ]
                            titles = [
                                "OM4 reference",
                                "OM4-pretrained: mean (8)",
                                "OM4-pretrained: member 1",
                                "Observation-adapted: mean (8)",
                                "Observation-adapted: member 1",
                                "Observation-adapted: member 2",
                            ]
                            label = "OM4 surface history + native forcing; interior is inferred"
                        unit = {
                            "thetao_0": "°C",
                            "zos": "m",
                            "thetao_9": "°C at 550 m",
                            "so_9": "Salinity at 550 m",
                        }[str(channel)]
                        grid(
                            fields,
                            titles,
                            z["mask"][c].astype(bool),
                            lat,
                            a.output / f"{mode}-{seed}-{year}-day{lead}-{channel}.png",
                            f"{label}\n{year}, seed {seed}, day {lead}, {channel}. Independent readouts; not time averages.",
                            unit,
                            show_loss_boundary=False,
                            distinguish_missing=True,
                        )
    for seed in (1729, 1730):
        obs = contracts[f"{seed}/obs-adapted"]
        native = contracts[f"{seed}/om4-adapted"]
        pre = contracts[f"{seed}/om4-pretrained"]
        if (
            obs["checkpoint_sha256"] != native["checkpoint_sha256"]
            or obs["training_protocol"]["pretrained_sha256"] != pre["checkpoint_sha256"]
        ):
            raise ValueError("Pretrained/adapted checkpoint lineage differs")
    with (a.output / "endpoint-diagnostics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "endpoint-provenance.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                metadata=metadata,
                contracts=contracts,
                scope="Three origins; endpoint fields, no monthly or annual averaging. Diagnostics on common finite wet support within 60S-60N. Neighbor RMS is grid-scale roughness, not a causal noise decomposition; latitude pairs do not wrap.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
