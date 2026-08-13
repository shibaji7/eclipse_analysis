"""Path helpers for the GOES/LANL workspace."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
REPO_DIR = PROJECT_DIR.parent.parent
PY_DIR = REPO_DIR / "py"


def ensure_repo_paths() -> None:
    """Expose the repo's shared helpers for local GOES/LANL modules."""
    for path in (str(PY_DIR), str(PROJECT_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

