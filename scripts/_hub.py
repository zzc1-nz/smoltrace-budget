"""Resilient Hub access through the hf-mirror.com endpoint.

huggingface.co itself is intermittently unreachable from this network while
hf-mirror.com (a Hub-compatible mirror) responds reliably and serves identical
bytes, so every read goes through the mirror first and falls back to the
canonical host. ``HF_ENDPOINT`` is honoured first when the caller sets one.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, List

MIRRORS: List[str] = []


def _candidates() -> List[str]:
    endpoints: List[str] = []
    env = os.environ.get("HF_ENDPOINT", "").strip()
    if env:
        endpoints.append(env.rstrip("/"))
    for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


def http_get(url: str, *, timeout: float = 60.0, attempts: int = 3, backoff: float = 2.0) -> bytes:
    """GET a URL, retrying across mirrors on connection-level failures."""
    if url.startswith("https://huggingface.co"):
        candidates = [url.replace("https://huggingface.co", base, 1) for base in _candidates()]
    else:
        candidates = [url]

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        for candidate in candidates:
            started = time.time()
            try:
                req = urllib.request.Request(
                    candidate,
                    headers={"User-Agent": "smoltrace-budget/0.1", "Accept": "*/*"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                # A real HTTP answer: try the next mirror, but do not retry blindly.
                last = exc
                print(f"    [http {exc.code}] {candidate} ({time.time() - started:.1f}s)")
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
                print(f"    [net] {candidate} -> {type(exc).__name__} ({time.time() - started:.1f}s)")
                continue
        if attempt < attempts:
            delay = backoff * attempt
            print(f"  [retry {attempt}/{attempts - 1}] sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"GET failed for {url} after {attempts} attempts") from last


def fetch_json(url: str, **kwargs: Any) -> Any:
    return json.loads(http_get(url, **kwargs))
