# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare verified latent reports to physical diffusion and deterministic reports."""

import argparse
import gzip
import json
from pathlib import Path

from samudra.experiments.diffusion_scratch_summary import compare_crps, read_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--physical-reports", type=Path, required=True)
    parser.add_argument("--latent-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, base_records, base_contract = read_report(args.baseline, scratch=False)
    families, comparisons = {}, {}
    for family in ("physical", "latent"):
        summaries, records = {}, {}
        for seed in (1729, 1730):
            name = f"{'B' if family == 'physical' else 'D'}-{seed}"
            root = (
                args.physical_reports / name
                if family == "physical"
                else args.latent_runs / name / "report-v1"
            )
            summaries[name], records[name], contract = read_report(
                root, scratch=True, architecture=family
            )
            if contract != base_contract:
                raise ValueError("Models use different observation or annual inputs")
        families[family] = summaries
        comparisons[family] = compare_crps(base_records, records)
    result = dict(
        scope="Persistent latent diffusion vs physical-state diffusion and unchanged deterministic baseline",
        interpretation="System comparison: latent dynamics are learned jointly; physical diffusion uses frozen pretrained dynamics. Training compute differs.",
        uncertainty="Latent readouts are independently sampled along one deterministic latent trajectory; physical diffusion propagates sampled physical initializations. Monthly uncertainty has different temporal covariance.",
        aggregation="Wet-area pooled over origins/bins, normalized per supported channel, then equal supported-channel average",
        calibration_caution="Eight-member quantiles lack nominal finite-ensemble coverage; report ranks and spread alongside raw coverage",
        baseline=baseline,
        families=families,
        interior_crps_comparison=comparisons,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True).encode()
    (args.output / "summary.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
