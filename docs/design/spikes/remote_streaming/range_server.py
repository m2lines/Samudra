# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Serve one immutable object with authenticated HTTP byte ranges."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeServer(ThreadingHTTPServer):
    object_path: Path
    token: str
    log_path: Path


class RangeHandler(BaseHTTPRequestHandler):
    server: RangeServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def reject(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def parse_range(self, size: int) -> tuple[int, int] | None:
        header = self.headers.get("Range")
        if header is None:
            return 0, size - 1
        if not header.startswith("bytes=") or "," in header:
            return None
        start_text, end_text = header.removeprefix("bytes=").split("-", 1)
        if not start_text:
            length = int(end_text)
            return max(0, size - length), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
        if start < 0 or end < start or start >= size:
            return None
        return start, min(end, size - 1)

    def serve(self, *, include_body: bool) -> None:
        started = time.perf_counter()
        if not secrets.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {self.server.token}"
        ):
            self.reject(HTTPStatus.UNAUTHORIZED)
            return
        if self.path != "/object":
            self.reject(HTTPStatus.NOT_FOUND)
            return
        size = self.server.object_path.stat().st_size
        selected = self.parse_range(size)
        if selected is None:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = selected
        length = end - start + 1
        ranged = self.headers.get("Range") is not None
        self.send_response(HTTPStatus.PARTIAL_CONTENT if ranged else HTTPStatus.OK)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Type", "application/octet-stream")
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        sent = 0
        if include_body:
            with self.server.object_path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    block = source.read(min(1 << 20, remaining))
                    if not block:
                        raise OSError("unexpected end of object")
                    self.wfile.write(block)
                    sent += len(block)
                    remaining -= len(block)
        record = {
            "remote": self.client_address[0],
            "method": self.command,
            "start": start,
            "end": end,
            "bytes": sent,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with self.server.log_path.open("a") as log:
            fcntl_lock(log)
            log.write(json.dumps(record) + "\n")

    def do_HEAD(self) -> None:
        self.serve(include_body=False)

    def do_GET(self) -> None:
        self.serve(include_body=True)


def fcntl_lock(file: object) -> None:
    import fcntl

    fcntl.flock(file, fcntl.LOCK_EX)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("object", type=Path)
    parser.add_argument("state_directory", type=Path)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--duration", type=int, default=600)
    args = parser.parse_args()
    args.state_directory.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    token_path = args.state_directory / "token"
    token_path.write_text(token)
    token_path.chmod(0o600)
    server = RangeServer(("0.0.0.0", args.port), RangeHandler)
    server.object_path = args.object
    server.token = token
    server.log_path = args.state_directory / "requests.jsonl"
    endpoint = {
        "hostname": socket.getfqdn(),
        "addresses": sorted(set(socket.gethostbyname_ex(socket.gethostname())[2])),
        "port": args.port,
        "path": "/object",
        "object_bytes": args.object.stat().st_size,
        "pid": os.getpid(),
    }
    (args.state_directory / "endpoint.json").write_text(
        json.dumps(endpoint, indent=2) + "\n"
    )
    stop = threading.Timer(args.duration, server.shutdown)
    stop.start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
