"""Runtime bootstrap for the smoltrace-budget project.

Two Windows-specific problems are solved here and nowhere else:

1. DSH's Windows sandbox creates temp directories whose ACLs our process cannot
   write into, so ``tempfile.mkdtemp`` produces unusable directories and pip /
   datasets / multiprocess all break. ``os.makedirs`` in a parent we own works,
   so ``mkdtemp`` is re-implemented on top of it.
2. The project's dependencies live in ``.venvlibs`` (installed with
   ``pip --target`` because the interpreter is externally managed). That path is
   prepended to ``sys.path`` so ``import smoltrace`` resolves to the pinned
   SMOLTRACE 0.2.12 checkout-compatible wheel.

Import this module first, before anything else that might touch temp files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBS_DIR = PROJECT_ROOT / ".venvlibs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TMP_DIR = PROJECT_ROOT / "_tmp" / "rt"


def _install_dependency_path() -> None:
    if LIBS_DIR.is_dir():
        libs = str(LIBS_DIR)
        if libs not in sys.path:
            sys.path.insert(0, libs)
    # Generated scripts/ modules import each other by top-level name.
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _patch_mkdtemp() -> None:
    """Make ``tempfile.mkdtemp`` create directories our process can actually write to."""
    if getattr(tempfile, "_dsh_patched", False):
        return
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    original_mkdtemp = tempfile.mkdtemp

    def mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None):
        base = dir or os.environ.get("DSH_TMPDIR") or str(TMP_DIR)
        os.makedirs(base, exist_ok=True)
        suffix = suffix or ""
        prefix = prefix or "tmp"
        for _ in range(500):
            candidate = os.path.join(base, prefix + os.urandom(6).hex() + suffix)
            try:
                os.makedirs(candidate)
                return candidate
            except FileExistsError:
                continue
        return original_mkdtemp(suffix=suffix, prefix=prefix, dir=base)

    tempfile.mkdtemp = mkdtemp
    tempfile._dsh_patched = True  # noqa: SLF001 - deliberate marker
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ.setdefault(var, str(TMP_DIR))
    os.environ["DSH_TMPDIR"] = str(TMP_DIR)


_install_dependency_path()
_patch_mkdtemp()

__all__ = ["PROJECT_ROOT", "LIBS_DIR", "DATA_DIR", "ARTIFACTS_DIR", "TMP_DIR"]
