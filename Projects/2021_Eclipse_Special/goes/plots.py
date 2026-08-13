"""GOES-only plotting helpers for the eclipse analysis."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .paths import ensure_repo_paths
from .runtime import configure_runtime
from publication_style import apply_publication_style


ensure_repo_paths()
configure_runtime()


def _as_datetime(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    from dateutil import parser as dparser

    return dparser.parse(str(value))


def plot_goes_timeseries(
    goes_data: dict[str, object],
    time_range: tuple[dt.datetime | str, dt.datetime | str],
    output_path: str | Path,
    eclipse_obscuration=None,
) -> None:
    """Render a compact GOES-only time-series figure."""

    apply_publication_style(font_size=8)
    start = _as_datetime(time_range[0])
    end = _as_datetime(time_range[1])

    if eclipse_obscuration is None:
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.2), sharex=True, dpi=300)
        obsc_ax = None
        mag_ax, flux_ax = axes
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.0), sharex=True, dpi=300)
        obsc_ax, mag_ax, flux_ax = axes

    any_mag = False
    any_flux = False

    if eclipse_obscuration is not None:
        if isinstance(eclipse_obscuration, dict):
            obsc_t = np.asarray(eclipse_obscuration["times"])
            obsc_v = np.asarray(eclipse_obscuration["values"])
        else:
            obsc_t = np.asarray(eclipse_obscuration.times)
            obsc_v = np.asarray(eclipse_obscuration.values)
        obsc_ax.plot(obsc_t, obsc_v, color="black", lw=1.2)
        obsc_ax.set_ylabel("Occulted area\n[10^6 km$^2$]")
        obsc_ax.grid(alpha=0.2, lw=0.5)

    for sat_name, sat_data in goes_data.items():
        series = sat_data.series.get("mag_h") or sat_data.series.get("mag_total")
        if series is not None:
            mag_ax.plot(series.times, np.asarray(series.values), lw=1.1, label=sat_name)
            any_mag = True

        flux_series = sat_data.series.get("particle_flux")
        if flux_series is not None:
            values = np.asarray(flux_series.values, dtype=float)
            values = np.where(values > 0, values, np.nan)
            flux_ax.semilogy(flux_series.times, values, lw=1.1, label=sat_name)
            any_flux = True

    mag_ax.set_ylabel("GOES MAG H [nT]")
    if any_mag:
        mag_ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")

    flux_ax.set_ylabel("GOES flux")
    flux_ax.set_yscale("log")
    if any_flux:
        flux_ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlim(start, end)

    flux_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    flux_ax.set_xlabel("UT")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
