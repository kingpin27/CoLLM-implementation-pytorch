#!/usr/bin/env python3
"""Simple internet connectivity checker."""

from __future__ import annotations

import argparse
import ssl
import socket
import sys
from urllib.request import Request, urlopen
from typing import List, Tuple


def has_http_internet(url: str, timeout: float) -> bool:
    request = Request(url, method="HEAD")
    context = ssl.create_default_context()
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            return 200 <= response.status < 400
    except OSError:
        return False


def has_internet(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_any_host(hosts: List[Tuple[str, int]], timeout: float) -> bool:
    return any(has_internet(host, port, timeout) for host, port in hosts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the machine has internet access."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Socket timeout in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.gstatic.com/generate_204",
        help="Fallback URL to test HTTP connectivity (default: Google's generate_204)",
    )
    parser.add_argument(
        "--hf-url",
        type=str,
        default="https://huggingface.co/",
        help="URL to verify Hugging Face accessibility (default: https://huggingface.co/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probes: List[Tuple[str, int]] = [
        ("1.1.1.1", 53),      # Cloudflare DNS
        ("8.8.8.8", 53),      # Google DNS
        ("208.67.222.222", 53),  # OpenDNS
        ("www.google.com", 443),
        ("www.cloudflare.com", 80),
    ]

    internet_online = check_any_host(probes, args.timeout) or has_http_internet(args.url, args.timeout)
    hugging_face_online = has_http_internet(args.hf_url, args.timeout)

    if internet_online:
        print("Internet connection: available")
    else:
        print("Internet connection: unavailable")

    if hugging_face_online:
        print(f"Hugging Face ({args.hf_url}) : accessible")
    else:
        print(f"Hugging Face ({args.hf_url}) : blocked or unreachable")

    online = internet_online and hugging_face_online
    if online:
        return 0

    print("Internet connection: unavailable")
    return 1


if __name__ == "__main__":
    sys.exit(main())
