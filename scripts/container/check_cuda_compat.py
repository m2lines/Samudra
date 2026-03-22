#!/usr/bin/env python3

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available inside the container.", file=sys.stderr)
        return 1

    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    supported_arches = sorted(
        arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")
    )

    print(f"Detected CUDA device architecture {device_arch}")
    print(f"Current PyTorch build supports: {supported_arches}")

    try:
        probe = (torch.ones(1, device="cuda") + 1).item()
        print(f"CUDA runtime probe succeeded: {probe}")
    except Exception as exc:
        print(f"CUDA runtime probe failed: {exc}", file=sys.stderr)
        if device_arch not in supported_arches:
            print(
                "Current runner GPU architecture is unsupported by the current PyTorch build.",
                file=sys.stderr,
            )
            return 2
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
