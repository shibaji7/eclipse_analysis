"""Runtime configuration for OMNI loaders."""

from __future__ import annotations

import os

from .paths import PACKAGE_DIR


CACHE_DIR = PACKAGE_DIR / ".cache"
OMNI_DATA_DIR = CACHE_DIR / "omni_data"
MPLCONFIG_DIR = CACHE_DIR / "mplconfig"


def configure_runtime() -> dict[str, str]:
    """Set writable runtime dirs before importing data loaders."""

    for directory in (OMNI_DATA_DIR, MPLCONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["OMNIDATA_PATH"] = str(OMNI_DATA_DIR)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
    os.environ["OMNI_DATA_DIR"] = str(OMNI_DATA_DIR)
    return {
        "OMNIDATA_PATH": os.environ["OMNIDATA_PATH"],
        "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
        "OMNI_DATA_DIR": os.environ["OMNI_DATA_DIR"],
    }
