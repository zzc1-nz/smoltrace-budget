"""Small helpers shared by every script in this project.

Import order matters on this machine: importing this module fixes the SDK
temp-directory problem and puts the project + its vendored dependencies on
``sys.path`` before anything else is imported.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smoltrace_budget import ARTIFACTS_DIR, DATA_DIR, TMP_DIR  # noqa: E402

__all__ = [
    "PROJECT_ROOT",
    "SCRIPTS_DIR",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "TMP_DIR",
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "ensure_dir",
    "pct",
    "http_get",
    "fetch_json",
]


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: os.PathLike | str, payload: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
    return p


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def pct(numerator: float, denominator: float, digits: int = 2) -> float:
    if not denominator:
        return 0.0
    return round(numerator / denominator * 100.0, digits)


def http_get(url: str, *, timeout: float = 60.0, attempts: int = 6, backoff: float = 1.5) -> bytes:
    """GET with retries. This network drops connections intermittently."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "smoltrace-budget/0.1",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if attempt == attempts:
                break
            delay = backoff * attempt
            print(f"  [retry {attempt}/{attempts - 1}] {type(exc).__name__}: {exc} -> sleep {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}") from last


def fetch_json(url: str, **kwargs: Any) -> Any:
    return json.loads(http_get(url, **kwargs))


from _hub import fetch_json as _hub_fetch_json  # noqa: E402,F401  (re-exported)
from _hub import http_get as _hub_http_get  # noqa: E402,F401  (re-exported)

# Hub-aware versions win: huggingface.co is intermittently unreachable here and
# _hub fails over to hf-mirror.com automatically. The plain urllib versions
# above stay for any endpoint that is not on the Hub.
http_get = _hub_http_get
fetch_json = _hub_fetch_json
