# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark authenticated HTTP byte ranges with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def fetch(host: str, port: int, token: str, start: int, length: int) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=30)
    began = time.perf_counter()
    connection.request(
        "GET",
        "/object",
        headers={
            "Authorization": f"Bearer {token}",
            "Range": f"bytes={start}-{start + length - 1}",
        },
    )
    response = connection.getresponse()
    body = response.read()
    elapsed = time.perf_counter() - began
    content_range = response.getheader("Content-Range")
    connection.close()
    if response.status != 206 or len(body) != length or content_range is None:
        raise RuntimeError(
            f"bad response status={response.status} bytes={len(body)} "
            f"content_range={content_range}"
        )
    return {"seconds": elapsed, "bytes": len(body)}


def trial(
    host: str,
    port: int,
    token: str,
    object_size: int,
    length: int,
    concurrency: int,
    requests: int,
) -> dict[str, Any]:
    max_start = object_size - length
    starts = [((index * 37_000_003) % max_start) for index in range(requests)]
    began = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(
            executor.map(lambda start: fetch(host, port, token, start, length), starts)
        )
    wall = time.perf_counter() - began
    latencies = [item["seconds"] for item in results]
    total = sum(item["bytes"] for item in results)
    return {
        "range_bytes": length,
        "concurrency": concurrency,
        "requests": requests,
        "total_bytes": total,
        "wall_seconds": wall,
        "throughput_mib_s": total / (1 << 20) / wall,
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("token", type=Path)
    args = parser.parse_args()
    endpoint = json.loads(args.endpoint.read_text())
    token = args.token.read_text().strip()
    host = endpoint["addresses"][0]
    port = endpoint["port"]
    results = []
    for length in (1 << 20, 8 << 20, 32 << 20):
        for concurrency in (1, 2, 4, 8):
            results.append(
                trial(
                    host,
                    port,
                    token,
                    endpoint["object_bytes"],
                    length,
                    concurrency,
                    requests=max(8, concurrency * 4),
                )
            )
    print(
        json.dumps(
            {
                "client_host": __import__("socket").getfqdn(),
                "server": {
                    key: value for key, value in endpoint.items() if key != "pid"
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
