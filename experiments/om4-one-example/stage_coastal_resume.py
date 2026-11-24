"""Copy completed 1,000-step states into separate, verified continuation runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assert (args.root / "COASTAL_OCT_PAIR_VERIFIED").exists()
    for arm in ("interior", "wet_neighbors"):
        source = args.root / "coastal-confirm/seed16_oct" / arm
        output = args.root / "coastal-long/seed16_oct" / arm
        assert (source / "COMPLETE").exists()
        protocol = json.loads((source / "protocol.json").read_text())
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        assert protocol["steps"] == history[-1]["step"] == 1000
        assert protocol["boundary"] == arm and protocol["weight"] == 100
        assert protocol["parameters"] == 84066474
        if (output / "RESUME_COPY_VERIFIED").exists():
            print(f"Already staged; leaving {output} unchanged.")
            continue
        assert not (output / "COMPLETE").exists()
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "staging.pt"
        shutil.copyfile(source / "latest.pt", temporary)
        expected = digest(source / "latest.pt")
        assert digest(temporary) == expected
        temporary.replace(output / "latest.pt")
        for name in ("maps-0.npz", "maps-1000.npz", "history.jsonl"):
            shutil.copyfile(source / name, output / name)
            assert digest(source / name) == digest(output / name)
        shutil.copyfile(source / "protocol.json", output / "source-protocol.json")
        (output / "checkpoint-copy.sha256").write_text(
            f"{expected}  {source / 'latest.pt'}\n{expected}  {output / 'latest.pt'}\n"
        )
        (output / "resume-provenance.json").write_text(json.dumps({
            "source": str(source), "source_outer_step": 1000, "target_outer_step": 2000,
            "intended_restore": "Same model, L-BFGS optimizer history, RNG states and elapsed counter; no loss or architecture changes.",
            "copied_history": "Original steps0-1000 retained; resumed measurement adds another step1000 record before updates.",
            "checkpoint_verification": "Full SHA256 read-back of source and copied checkpoint.",
        }, indent=2) + "\n")
        (output / "RESUME_COPY_VERIFIED").write_text("Exact completed checkpoint copy verified; source left unchanged.\n")
        print(f"Verified resume copy for {arm}.")


if __name__ == "__main__":
    main()
