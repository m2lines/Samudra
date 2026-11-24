"""Copy the verified January interior trajectory into matched threshold controls."""

import argparse
import json
from pathlib import Path
import shutil

from stage_coastal_resume import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert (root / "JANUARY_PAIR_VERIFIED").exists()
    source = root / "coastal-confirm/seed15_jan/interior"
    assert (source / "COMPLETE").exists()
    protocol = json.loads((source / "protocol.json").read_text())
    history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
    assert protocol["steps"] == history[-1]["step"] == 1000
    assert protocol["boundary"] == "interior" and protocol["parameters"] == 84066474
    expected = digest(source / "latest.pt")
    for arm in ("default", "lowered"):
        output = root / "curvature-threshold-jan" / arm
        assert not (output / "COMPLETE").exists()
        if (output / "RESUME_COPY_VERIFIED").exists():
            continue
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "staging.pt"
        shutil.copyfile(source / "latest.pt", temporary)
        assert digest(temporary) == expected
        temporary.replace(output / "latest.pt")
        for name in ("maps-0.npz", "maps-1000.npz", "history.jsonl"):
            shutil.copyfile(source / name, output / name)
            assert digest(source / name) == digest(output / name)
        shutil.copyfile(source / "protocol.json", output / "source-protocol.json")
        (output / "checkpoint-copy.sha256").write_text(
            f"{expected}  {source / 'latest.pt'}\n{expected}  {output / 'latest.pt'}\n")
        (output / "resume-provenance.json").write_text(json.dumps({
            "source": str(source), "source_outer_step": 1000, "target_outer_step": 2500,
            "restore": "Exact model, optimizer history, RNG and elapsed state. Both arms add identical passive counters; only curvature acceptance threshold differs.",
        }, indent=2) + "\n")
        (output / "RESUME_COPY_VERIFIED").write_text("Full checkpoint copy read-back verified.\n")
        print(f"Verified {arm}", flush=True)


if __name__ == "__main__":
    main()
