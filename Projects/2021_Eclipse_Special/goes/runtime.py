"""Runtime configuration for GOES/LANL loaders."""

from __future__ import annotations

import os

from .paths import PACKAGE_DIR


CACHE_DIR = PACKAGE_DIR / ".cache"
SPACEPY_DIR = CACHE_DIR / "spacepy"
MPLCONFIG_DIR = CACHE_DIR / "mplconfig"
GOES_DATA_DIR = PACKAGE_DIR / "goes_data"
LANL_DATA_DIR = PACKAGE_DIR / "lanl_data"
SPEDAS_DATA_DIR = PACKAGE_DIR.parent


def configure_runtime() -> dict[str, str]:
    """Set writable runtime dirs before importing pyspedas/spacepy."""
    for directory in (SPACEPY_DIR, MPLCONFIG_DIR, GOES_DATA_DIR, LANL_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["SPACEPY"] = str(SPACEPY_DIR)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
    os.environ["SPEDAS_DATA_DIR"] = str(SPEDAS_DATA_DIR)
    os.environ["GOES_DATA_DIR"] = str(GOES_DATA_DIR)
    os.environ["LANL_DATA_DIR"] = str(LANL_DATA_DIR)
    return {
        "SPACEPY": os.environ["SPACEPY"],
        "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
        "SPEDAS_DATA_DIR": os.environ["SPEDAS_DATA_DIR"],
        "GOES_DATA_DIR": os.environ["GOES_DATA_DIR"],
        "LANL_DATA_DIR": os.environ["LANL_DATA_DIR"],
    }

