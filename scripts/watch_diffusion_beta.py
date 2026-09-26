#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Hourly dependency watcher: stage final baseline, then submit one bounded qualification.

No production submission or automatic recovery of failed qualification. A durable
submission intent prevents duplicate jobs on uncertain SSH responses. Source
selection uses only the upstream frozen validation score after BOTH final runs
and their monthly/annual reports have completed; held-out scores never select.
"""

import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

SOURCE = "/scratch/jr7309/runs/2026-09-25-observation-instance/handoff-v2"
BETA = "/mnt/home/jrusak/data/diffusion-interior-beta"

PROBE = r"""
import hashlib,json,math
from pathlib import Path
root=Path("/scratch/jr7309/runs/2026-09-25-observation-instance/handoff-v2")
reports=[]
for p in root.rglob("COMPLETE.json"):
    reports.append((str(p),json.loads(p.read_text())))
rows=[]
for name in ["scratch-main","transfer-main"]:
    run=root/name
    if not (run/"TRAIN_COMPLETE.json").exists():
        print(json.dumps({"ready":False,"waiting":name+" training"}));raise SystemExit
    done=json.loads((run/"TRAIN_COMPLETE.json").read_text())
    best=json.loads((run/"best.json").read_text())
    manifest=json.loads((run/"manifest.json").read_text())
    if done["best_checkpoint_sha256"]!=best["checkpoint_sha256"] or done["best_score"]!=best["score"]:
        raise ValueError("Final checkpoint selection differs from completion")
    if not math.isfinite(best["score"]) or manifest["arguments"]["normalization"]!="instance":
        raise ValueError("Upstream baseline contract changed")
    monthly=[p for p,d in reports if d.get("selected_sha256")==best["checkpoint_sha256"] and d.get("origins")==96]
    annual=[p for p,d in reports if d.get("inputs",{}).get("checkpoint_sha256")==best["checkpoint_sha256"] and d.get("inputs",{}).get("split")=="test"]
    if not monthly or not annual:
        print(json.dumps({"ready":False,"waiting":name+" final monthly/annual results"}));raise SystemExit
    rows.append({"name":name,"score":best["score"],"checkpoint_sha256":best["checkpoint_sha256"],"producer":manifest["code_commit"],"monthly_reports":monthly,"annual_reports":annual,"data_manifest_sha256":manifest["data_manifest_sha256"],"statistics_sha256":manifest["statistics_sha256"],"selection_reference_sha256":hashlib.sha256((run/"selection-reference.json").read_bytes()).hexdigest()})
for key in ["data_manifest_sha256","statistics_sha256","selection_reference_sha256"]:
    if rows[0][key]!=rows[1][key]:raise ValueError("Unmatched source comparison: "+key)
chosen=min(rows,key=lambda x:x["score"])
with (root/chosen["name"]/"best.pt").open("rb") as f:
    if hashlib.file_digest(f,"sha256").hexdigest()!=chosen["checkpoint_sha256"]:raise ValueError("Source checkpoint hash mismatch")
print(json.dumps({"ready":True,"selected":chosen["name"],"selection":"minimum final upstream validation score; held-out reports required but never ranked","candidates":rows}))
"""


def remote(host, command, timeout=90):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def record(output, state):
    state["time_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    temporary = output / "latest.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(output / "latest.json")
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(state) + "\n")
    print(json.dumps(state), flush=True)


def stage(selection, output):
    name = selection["selected"]
    if name not in ("scratch-main", "transfer-main"):
        raise ValueError("Unexpected source run")
    archive = output / "baseline.tar"
    files = [
        f"{name}/{f}"
        for f in (
            "best.pt",
            "best.json",
            "manifest.json",
            "TRAIN_COMPLETE.json",
            "selection-reference.json",
        )
    ]
    # Large reads go through the Torch DTN; no new object-store publication.
    command = shlex.join(["tar", "-C", SOURCE, "-cf", "-", *files])
    with archive.open("wb") as stream:
        subprocess.run(
            ["ssh", "torch", "ssh dtn011 " + shlex.quote(command)],
            stdout=stream,
            check=True,
            timeout=1800,
        )
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    subprocess.run(
        ["scp", str(archive), str(selection_path), f"beta:{BETA}/checkpoints/"],
        check=True,
        timeout=1800,
    )
    remote(
        "beta",
        f"tar -xf {BETA}/checkpoints/baseline.tar -C {BETA}/checkpoints",
        timeout=180,
    )
    expected = next(
        r["checkpoint_sha256"] for r in selection["candidates"] if r["name"] == name
    )
    actual = remote(
        "beta",
        shlex.join(["sha256sum", f"{BETA}/checkpoints/{name}/best.pt"]),
        timeout=180,
    ).split()[0]
    if actual != expected:
        raise ValueError("Beta checkpoint read-back hash differs")


def monitor_job(output, job, interval):
    while True:
        command = shlex.join(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job,
                "-o",
                "JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode",
            ]
        )
        raw = remote("beta", "bash -lc " + shlex.quote(command))
        rows = [
            line.split("|") for line in raw.splitlines() if line.split("|")[0] == job
        ]
        if not rows:
            record(output, {"phase": "waiting_accounting", "job": job})
        else:
            row = rows[0]
            state = row[1].split()[0].rstrip("+")
            record(
                output,
                {
                    "phase": "qualification_" + state.lower(),
                    "job": job,
                    "accounting": row,
                },
            )
            if state == "COMPLETED":
                results = {}
                for role in ("baseline", "deterministic", "diffusion", "geometry"):
                    results[role] = json.loads(
                        remote(
                            "beta",
                            shlex.join(
                                [
                                    "cat",
                                    f"{BETA}/runs/qualification/{role}/QUALIFIED.json",
                                ]
                            ),
                        )
                    )
                if (
                    results["deterministic"]["input_sha256"]
                    != results["diffusion"]["input_sha256"]
                ):
                    raise ValueError("A/B qualification inputs differ")
                (output / "qualification-results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                record(
                    output,
                    {
                        "phase": "qualified_awaiting_report_and_production_caps",
                        "job": job,
                        "allocated_gpu_hours": int(row[2]) * 4 / 3600,
                    },
                )
                return
            if state not in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
            }:
                raise ValueError(
                    f"Qualification {job} stopped with {state}; inspect before recovery"
                )
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.interval < 600
        or len(args.code_commit) != 40
        or any(c not in "0123456789abcdef" for c in args.code_commit)
    ):
        parser.error(
            "Require an immutable commit and at least 600 seconds between checks"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                selection = json.loads(
                    remote("torch", shlex.join(["python3", "-c", PROBE]))
                )
                data_ready = (
                    remote(
                        "beta",
                        f"test -f {BETA}/DATA_READY.json && echo ready || echo waiting",
                    ).strip()
                    == "ready"
                )
                if not selection["ready"] or not data_ready:
                    record(
                        args.output,
                        {
                            "phase": "waiting_dependencies",
                            "upstream": selection,
                            "data_ready": data_ready,
                        },
                    )
                elif (args.output / "submission-intent.json").exists():
                    record(
                        args.output,
                        {
                            "phase": "submission_already_attempted",
                            "action": "Inspect submission.json or reconcile Slurm before retrying",
                        },
                    )
                    return
                else:
                    stage(selection, args.output)
                    intent = {
                        "selection": selection,
                        "code_commit": args.code_commit,
                        "maximum_allocated_gpu_hours": 8,
                        "campaign_cap_gpu_hours": 576,
                    }
                    (args.output / "submission-intent.json").write_text(
                        json.dumps(intent, indent=2) + "\n"
                    )
                    command = shlex.join(
                        [
                            "sbatch",
                            "--parsable",
                            f"--export=ALL,CODE_COMMIT={args.code_commit}",
                            f"{BETA}/code/{args.code_commit}/scripts/slurm_diffusion_beta_qualify.sbatch",
                        ]
                    )
                    result = remote("beta", "bash -lc " + shlex.quote(command))
                    job = result.strip().split(";")[0]
                    if not job.isdigit():
                        raise ValueError(f"Ambiguous Slurm submission: {result!r}")
                    record(
                        args.output,
                        {"phase": "qualification_submitted", "job": job, **intent},
                    )
                    (args.output / "submission.json").write_text(
                        json.dumps({"job": job, **intent}, indent=2) + "\n"
                    )
                    monitor_job(args.output, job, args.interval)
                    return
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                record(args.output, {"phase": "blocked", "error": str(exc)})
                # Do not retry a possible submission. Connection failures while
                # waiting are retried hourly without launching extra resources.
                if (args.output / "submission-intent.json").exists():
                    return
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
