"""OMNI plotting helpers for the eclipse analysis."""

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


def plot_omni_timeseries(
    omni_data: dict[str, object],
    time_range: tuple[dt.datetime | str, dt.datetime | str],
    output_path: str | Path,
    feature_window: tuple[dt.datetime | str, dt.datetime | str] | None = None,
    eclipse_obscuration=None,
) -> None:
    """Render a stacked OMNI figure on a shared UT time axis."""

    apply_publication_style(font_size=8)
    start = _as_datetime(time_range[0])
    end = _as_datetime(time_range[1])
    _ = feature_window

    if eclipse_obscuration is None:
        fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.4), sharex=True, dpi=300)
        ax_occ = None
        ax_pdyn, ax_b, ax_speed, ax_density = axes
    else:
        fig, axes = plt.subplots(5, 1, figsize=(10.5, 10.2), sharex=True, dpi=300)
        ax_occ, ax_pdyn, ax_b, ax_speed, ax_density = axes

    if eclipse_obscuration is not None:
        if isinstance(eclipse_obscuration, dict):
            occ_t = np.asarray(eclipse_obscuration["times"])
            occ_v = np.asarray(eclipse_obscuration["values"])
        else:
            occ_t = np.asarray(eclipse_obscuration.times)
            occ_v = np.asarray(eclipse_obscuration.values)
        ax_occ.plot(occ_t, occ_v, color="black", lw=1.2)
        ax_occ.set_ylabel("Occulted area\n[10^6 km$^2$]")
        ax_occ.grid(alpha=0.2, lw=0.5)

    for sat_name, sat_data in omni_data.items():
        pdyn = sat_data.series.get("pdyn")
        if pdyn is not None:
            ax_pdyn.plot(pdyn.times, np.asarray(pdyn.values), color="black", lw=1.1, label=sat_name)

        bz = sat_data.series.get("bz_gsm")
        by = sat_data.series.get("by_gsm")
        bmag = sat_data.series.get("b_mag")
        if bmag is not None:
            ax_b.plot(bmag.times, np.asarray(bmag.values), color="#666666", lw=1.0, label="|B|")
        if by is not None:
            ax_b.plot(by.times, np.asarray(by.values), color="#e76f51", lw=1.0, label="By GSM")
        if bz is not None:
            ax_b.plot(bz.times, np.asarray(bz.values), color="#264653", lw=1.0, label="Bz GSM")

        speed = sat_data.series.get("speed")
        if speed is not None:
            ax_speed.plot(speed.times, np.asarray(speed.values), color="#1d3557", lw=1.1, label=sat_name)

        density = sat_data.series.get("density")
        if density is not None:
            ax_density.plot(density.times, np.asarray(density.values), color="#8d99ae", lw=1.1, label=sat_name)

    ax_pdyn.set_ylabel("Pdyn [nPa]")
    ax_b.set_ylabel("IMF [nT]")
    ax_speed.set_ylabel("Vsw [km/s]")
    ax_density.set_ylabel("Np [cm$^{-3}$]")

    ax_b.legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
    ax_pdyn.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
    ax_speed.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
    ax_density.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlim(start, end)

    ax_density.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_density.set_xlabel("UT")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
