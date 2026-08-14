"""Consolidated GRL-style summary figure for the 2021 eclipse event."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np

from publication_style import apply_publication_style

from goes.loaders import CorroborationSatelliteData, GOES_LONGITUDES
from omni.loaders import OmniData

from .loaders import MMSProbeData, MMSSeries
from .plots import PROBE_COLORS


GEO_RADIUS_RE = 6.6

COLOR_EARTH = "#1d3557"
COLOR_GOES = "#111111"
COLOR_ECLIPSE = "#8a8f98"
COLOR_IMF_BMAG = "#6b6b6b"
COLOR_IMF_BY = "#e76f51"
COLOR_IMF_BZ = "#264653"
COLOR_PDY = "#8b1e3f"
COLOR_AE = "#c44536"
COLOR_AL = "#457b9d"
COLOR_MMS_V = "#2a9d8f"
COLOR_MMS_EXB = "#f4a261"
PANEL_FONT_SIZE = 12
PANEL_LETTER_SIZE = 14
PANEL_ANNOTATION_SIZE = 12


def _style_axis(ax, *, facecolor: str | None = None) -> None:
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=PANEL_FONT_SIZE,
    )
    # ax.grid(alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)


def _as_datetime(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    from dateutil import parser as dparser

    return dparser.parse(str(value))


def _series_times_values(series):
    times = np.asarray(series.times, dtype=object)
    values = np.asarray(series.values, dtype=float)
    return times, values


def _vector_magnitude(series) -> tuple[np.ndarray, np.ndarray]:
    times, values = _series_times_values(series)
    if values.ndim == 1:
        mag = values
    else:
        width = min(3, values.shape[1])
        mag = np.linalg.norm(values[:, :width], axis=1)
    return times, mag


def _smooth_series(values: np.ndarray, window: int = 31) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or window <= 1:
        return arr
    window = min(int(window), arr.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return arr
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _normalize_01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if np.isclose(hi, lo):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        0.01,
        0.95,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_SIZE,
        color="black",
        va="top",
        ha="left",
    )


def _add_feature_window(ax, feature_window) -> None:
    if feature_window is None:
        return
    start = _as_datetime(feature_window[0])
    end = _as_datetime(feature_window[1])
    ax.axvspan(start, end, color="#d4b483", alpha=0.16, zorder=0)


def _goes_point(goes_sat: CorroborationSatelliteData, toi: dt.datetime) -> tuple[float, float]:
    probe = goes_sat.satellite.split("-")[-1]
    lon = GOES_LONGITUDES.get(probe, -137.2)
    ut_hour = toi.hour + toi.minute / 60.0 + toi.second / 3600.0
    local_time = (ut_hour + lon / 15.0) % 24.0
    theta = np.deg2rad((12.0 - local_time) * 15.0)
    return GEO_RADIUS_RE * np.cos(theta), GEO_RADIUS_RE * np.sin(theta)


def _probe_track_plane(
    probe_data: MMSProbeData,
    toi: dt.datetime,
    plane: str = "xy",
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    r = probe_data.get("r")
    if r is None:
        raise ValueError("Missing MMS position data.")
    values = np.asarray(r.values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("MMS position track must be 2D.")
    if np.nanmax(np.abs(values)) > 1_000.0:
        values = values / 6371.0
    plane = plane.lower()
    if plane == "yz" and values.shape[1] >= 3:
        x = values[:, 1]
        y = values[:, 2]
    else:
        x = values[:, 0]
        if plane == "xz" and values.shape[1] >= 3:
            y = values[:, 2]
        else:
            y = values[:, 1]
    times = np.asarray(r.times, dtype=object)
    idx = int(np.argmin([abs((t - toi).total_seconds()) for t in times]))
    r_re = float(np.linalg.norm(values[idx, : min(values.shape[1], 3)]))
    y_xy = values[:, 1] if values.shape[1] >= 2 else np.zeros_like(x)
    mlt = (np.degrees(np.arctan2(y_xy[idx], x[idx])) / 15.0 + 12.0) % 24.0
    return x, y, float(x[idx]), float(y[idx]), r_re, mlt


def _plot_location_schematic(
    ax,
    mec_data: dict[str, MMSProbeData],
    goes_sat: CorroborationSatelliteData,
    toi: dt.datetime,
    plane: str = "xy",
) -> None:
    ax.set_aspect("equal")
    ax.set_facecolor("#faf7ef")
    ax.axhline(0, color="0.78", lw=1.0, zorder=0)
    ax.axvline(0, color="0.78", lw=1.0, zorder=0)
    _style_axis(ax, facecolor="w")

    extent = GEO_RADIUS_RE * 1.45
    tracks = {}
    for probe_name, probe_data in mec_data.items():
        if not probe_data.has("r"):
            continue
        x, y, x_toi, y_toi, r_re, mlt = _probe_track_plane(probe_data, toi, plane=plane)
        tracks[probe_name] = (x, y, x_toi, y_toi, r_re, mlt)
        extent = max(extent, np.nanmax(np.hypot(x, y)) * 1.1)

    earth = Circle((0, 0), radius=1.0, facecolor=COLOR_EARTH, edgecolor="black", lw=1.0, zorder=2)
    ax.add_patch(earth)
    if plane != "yz":
        ax.text(0, 0, "Earth", color="r", ha="center", va="center", fontsize=PANEL_ANNOTATION_SIZE, zorder=3)

    geo_ring = Circle((0, 0), radius=GEO_RADIUS_RE, facecolor="none", edgecolor="0.65", lw=0.9, ls="--", zorder=1)
    ax.add_patch(geo_ring)
    # ax.text(GEO_RADIUS_RE * 0.93, -0.6, "GEO", fontsize=8, color="0.45", ha="right", va="top")

    if plane == "xz":
        ax.annotate(
            "Sun",
            xy=(extent * 0.82, 0.0),
            xytext=(extent * 0.42, 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
            ha="center",
            va="center",
            fontsize=PANEL_ANNOTATION_SIZE,
        )
        ax.text(extent * 0.77, 1.0, "Sunward", ha="center", va="bottom", fontsize=PANEL_ANNOTATION_SIZE)
        ax.text(-extent * 0.77, 1.0, "Anti-sunward", ha="center", va="bottom", fontsize=PANEL_ANNOTATION_SIZE)
        ax.text(0.92, 0.95, "X-Z view", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE, weight="bold", ha="left", va="top")
    elif plane == "yz":
        # ax.text(extent * 0.77, 1.0, "Duskward", ha="center", va="bottom", fontsize=8)
        ax.text(-extent * 0.77, 1.0, "Dawnward", ha="center", va="bottom", fontsize=PANEL_ANNOTATION_SIZE)
        # ax.text(1.0, extent * 0.77, "Northward", ha="right", va="center", fontsize=8)
        ax.text(1.0, -extent * 0.77, "Southward", ha="right", va="center", fontsize=PANEL_ANNOTATION_SIZE)
        ax.text(1, 0.95, "Sun out of plane", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE, weight="bold", ha="right", va="top")
        ax.text(1.01, 0.95, "Y-Z view", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE, weight="bold", ha="left", va="top", rotation=90)
    else:
        ax.annotate(
            "Sun",
            xy=(extent * 0.82, 0.0),
            xytext=(extent * 0.42, 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
            ha="center",
            va="center",
            fontsize=PANEL_ANNOTATION_SIZE,
        )
        # ax.text(extent * 0.77, 1.0, "Dayside", ha="center", va="bottom", fontsize=8)
        ax.text(-extent * 0.77, 1.0, "Nightside", ha="center", va="bottom", fontsize=PANEL_ANNOTATION_SIZE)
        ax.text(1.01, 0.95, "X-Y view", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE, weight="bold", ha="left", va="top", rotation=90)

    for probe_name, (x, y, x_toi, y_toi, r_re, mlt) in tracks.items():
        color = PROBE_COLORS.get(probe_name, "C0")
        ax.plot(x, y, color=color, lw=1.0, alpha=0.8, zorder=3)
        ax.scatter([x[0]], [y[0]], s=18, color=color, edgecolor="black", linewidth=0.4, zorder=4, marker="o")
        ax.scatter([x[-1]], [y[-1]], s=24, color=color, edgecolor="black", linewidth=0.4, zorder=4, marker="s")
        ax.scatter([x_toi], [y_toi], s=68, color=color, edgecolor="black", linewidth=0.6, zorder=5)
        if plane != "yz":
            label = "MMS" if probe_name.lower() == "mms1" else probe_name.upper()
            ax.text(
                x_toi + 0.32,
                y_toi + 0.32,
                f"{label}\nR={r_re:.1f} Re\nMLT={mlt:.1f}",
                fontsize=PANEL_ANNOTATION_SIZE - 2,
                color=color,
                ha="left",
                va="bottom",
                zorder=6,
            )

    gx, gy = _goes_point(goes_sat, toi)
    probe = goes_sat.satellite.split("-")[-1]
    if plane == "yz":
        gy = 0.0
        ax.plot([0.0, gx], [0.0, 0.0], color=COLOR_GOES, lw=1.0, ls=":", alpha=0.9, zorder=5)
    ax.scatter([gx], [gy], s=100, marker="*", color=COLOR_GOES, edgecolor="white", linewidth=0.6, zorder=6)
    if plane != "yz":
        ax.text(
            gx + 0.4,
            gy - 0.35,
            f"GOES-{probe}",
            fontsize=PANEL_ANNOTATION_SIZE,
            color=COLOR_GOES,
            ha="left",
            va="top",
            zorder=7,
        )

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    if plane == "yz":
        ax.set_xlabel("GSM Y [Re]", fontsize=PANEL_ANNOTATION_SIZE)
        ax.set_ylabel("GSM Z [Re]", fontsize=PANEL_ANNOTATION_SIZE)
    elif plane == "xz":
        ax.set_xlabel("GSM X [Re]", fontsize=PANEL_ANNOTATION_SIZE)
        ax.set_ylabel("GSM Z [Re]", fontsize=PANEL_ANNOTATION_SIZE)
    else:
        ax.set_xlabel("GSM X [Re]", fontsize=PANEL_ANNOTATION_SIZE)
        ax.set_ylabel("GSM Y [Re]", fontsize=PANEL_ANNOTATION_SIZE)
    # ax.set_title(f"Spacecraft locations at {toi:%H:%M UT}", fontsize=10)
    if plane != "yz":
        ax.text(0.02, 0.02, "Not to scale", transform=ax.transAxes, fontsize=8, weight="bold", ha="left", va="bottom")
    # No grid for the consolidated GRL figure.


def _plot_goes_panel(
    ax,
    ax_r,
    ax_r2,
    goes_sat: CorroborationSatelliteData,
    eclipse_obscuration: MMSSeries | dict,
    omni_sat: OmniData,
    feature_window,
) -> None:
    series = goes_sat.series.get("mag_h") or goes_sat.series.get("mag_total")
    if series is not None:
        ax.plot(series.times, np.asarray(series.values, dtype=float), color=COLOR_GOES, lw=1.15)
    ax.set_ylabel(f"{goes_sat.satellite} H [nT]", color=COLOR_GOES, fontsize=PANEL_FONT_SIZE)
    ax.set_ylim(50, 90)
    _style_axis(ax, facecolor="white")

    if isinstance(eclipse_obscuration, dict):
        occ_t = np.asarray(eclipse_obscuration["times"], dtype=object)
        occ_v = np.asarray(eclipse_obscuration["values"], dtype=float)
    else:
        occ_t = np.asarray(eclipse_obscuration.times, dtype=object)
        occ_v = np.asarray(eclipse_obscuration.values, dtype=float)
    occ_n = 1.0 - _normalize_01(occ_v)
    ae = omni_sat.series.get("ae")
    al = omni_sat.series.get("al")
    if ae is not None:
        ax_r.plot(ae.times, np.asarray(ae.values, dtype=float), color=COLOR_AE, lw=1.35, label="AE")
    if al is not None:
        ax_r.plot(al.times, np.asarray(al.values, dtype=float), color=COLOR_AL, lw=1.35, ls="--", label="AL")
    ax_r.set_ylabel("AE / AL [nT]", fontsize=PANEL_FONT_SIZE)
    ax_r.tick_params(axis="y", labelsize=10)
    ax_r.set_ylim(400, -400)
    ax_r.legend(fontsize=10, frameon=False, loc="upper right")

    ax_r2.spines["right"].set_position(("axes", 1.11))
    ax_r2.spines["right"].set_visible(True)
    ax_r2.patch.set_visible(False)
    ax_r2.plot(occ_t, occ_n, color=COLOR_ECLIPSE, lw=1.0)
    ax_r2.set_ylim(0, 1.02)
    ax_r2.set_ylabel("Eclipse\n(norm.)", color=COLOR_ECLIPSE, fontsize=PANEL_FONT_SIZE)
    ax_r2.tick_params(axis="y", colors=COLOR_ECLIPSE)
    ax_r2.tick_params(axis="y", which="both", direction="in", labelsize=PANEL_FONT_SIZE)

    # ax.text(0.985, 0.1, "GOES H + eclipse", transform=ax.transAxes, fontsize=8, ha="right", va="bottom")


def _plot_omni_panel(
    ax,
    ax_r,
    omni_sat: OmniData,
    feature_window,
) -> None:
    by = omni_sat.series.get("by_gsm")
    bz = omni_sat.series.get("bz_gsm")
    bmag = omni_sat.series.get("b_mag")
    pdyn = omni_sat.series.get("pdyn")

    if bmag is not None:
        ax.scatter(
            bmag.times,
            np.asarray(bmag.values, dtype=float),
            color=COLOR_IMF_BMAG,
            s=13,
            marker="o",
            linewidths=0,
            label="|B|",
        )
    if by is not None:
        ax.scatter(
            by.times,
            np.asarray(by.values, dtype=float),
            color=COLOR_IMF_BY,
            s=13,
            marker="s",
            linewidths=0,
            label="By GSM",
        )
    if bz is not None:
        ax.scatter(
            bz.times,
            np.asarray(bz.values, dtype=float),
            color=COLOR_IMF_BZ,
            s=13,
            marker="D",
            linewidths=0,
            label="Bz GSM",
        )
    ax.set_ylabel("IMF [nT]", fontsize=PANEL_FONT_SIZE)
    # No grid for the consolidated GRL figure.
    ax.set_ylim(-10, 10)

    if pdyn is not None:
        ax_r.scatter(
            pdyn.times,
            np.asarray(pdyn.values, dtype=float),
            color=COLOR_PDY,
            s=13,
            marker="^",
            linewidths=0,
        )
    ax_r.set_ylabel("Pdyn [nPa]", color=COLOR_PDY, fontsize=PANEL_FONT_SIZE)
    ax_r.tick_params(axis="y", colors=COLOR_PDY)
    ax_r.tick_params(axis="y", which="both", direction="in", labelsize=PANEL_FONT_SIZE)
    ax_r.set_ylim(0, 10)

    ax.legend(ncol=3, fontsize=10, frameon=False, loc="upper right")
    _style_axis(ax, facecolor="white")
    # ax.text(0.985, 0.1, "IMF + Pdyn", transform=ax.transAxes, fontsize=8, ha="right", va="bottom")


def _plot_ae_panel(
    ax,
    omni_sat: OmniData,
    feature_window,
) -> None:
    ae = omni_sat.series.get("ae")
    if ae is not None:
        ax.plot(ae.times, np.asarray(ae.values, dtype=float), color=COLOR_AE, lw=1.1)
    ax.set_ylabel("AE [nT]", color=COLOR_AE)
    ax.tick_params(axis="y", colors=COLOR_AE)
    _style_axis(ax, facecolor="white")
    ax.text(0.985, 0.1, "AE", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE, ha="right", va="bottom")


def _plot_mms_velocity_panel(
    ax,
    ax_r,
    probe_data: MMSProbeData | None,
    feature_window,
) -> None:
    highlight_start = dt.datetime.combine(dt.date(2021, 12, 4), dt.time(7, 15))
    highlight_end = dt.datetime.combine(dt.date(2021, 12, 4), dt.time(7, 45))
    ax.axvspan(highlight_start, highlight_end, color="#f2c14e", alpha=0.16, zorder=0)

    if probe_data is not None:
        if probe_data.has("ion_bulkv"):
            times_v, vmag = _vector_magnitude(probe_data.get("ion_bulkv"))
            ax.plot(times_v, vmag, color=COLOR_MMS_V, lw=1.15, alpha=0.95, label="|V|")
            ax.plot(
                times_v,
                _smooth_series(vmag, window=31),
                color="black",
                lw=1.0,
                ls="--",
                alpha=0.9,
                label="|V| smoothed",
            )
        if probe_data.has("exb"):
            times_x, xmag = _vector_magnitude(probe_data.get("exb"))
            ax_r.plot(times_x, xmag, color=COLOR_MMS_EXB, lw=1.15, alpha=0.95, linestyle="--", label="|ExB|")
            ax_r.plot(
                times_x,
                _smooth_series(xmag, window=31),
                color="black",
                lw=1.0,
                ls="--",
                alpha=0.9,
                label="|ExB| smoothed",
            )

    ax.set_ylabel("MMS1 |V| [km/s]", color=COLOR_MMS_V, fontsize=PANEL_FONT_SIZE)
    ax_r.set_ylabel("MMS1 |ExB| [km/s]", color=COLOR_MMS_EXB, fontsize=PANEL_FONT_SIZE)
    ax.tick_params(axis="y", colors=COLOR_MMS_V)
    ax_r.tick_params(axis="y", colors=COLOR_MMS_EXB)
    ax_r.tick_params(axis="y", which="both", direction="in", labelsize=PANEL_FONT_SIZE)
    _style_axis(ax, facecolor="white")
    ax.text(0.985, 0.1, "black dashed: smoothed", transform=ax.transAxes, fontsize=PANEL_ANNOTATION_SIZE - 4, ha="right", va="bottom")
    # ax.text(0.985, 0.87, "MMS1 velocity", transform=ax.transAxes, fontsize=8, ha="right", va="top")


def plot_grl_summary_figure(
    mec_data: dict[str, MMSProbeData],
    fields_data: dict[str, MMSProbeData],
    goes_data: dict[str, CorroborationSatelliteData],
    omni_data: dict[str, OmniData],
    eclipse_obscuration: MMSSeries | dict,
    time_range: tuple[dt.datetime | str, dt.datetime | str],
    time_of_interest: dt.datetime | str,
    output_path: str | Path,
    feature_window: tuple[dt.datetime | str, dt.datetime | str] | None = None,
) -> None:
    """Render the consolidated GRL-style figure."""

    apply_publication_style(font_size=8)
    start = _as_datetime(time_range[0])
    end = _as_datetime(time_range[1])
    toi = _as_datetime(time_of_interest)

    if feature_window is None:
        feature_window = (
            dt.datetime.combine(start.date(), dt.time(7, 0)),
            dt.datetime.combine(start.date(), dt.time(7, 45)),
        )

    goes_sat = next(iter(goes_data.values()))
    omni_sat = next(iter(omni_data.values()))

    fig = plt.figure(figsize=(8.2, 10.8), dpi=300, facecolor="#fbfbfb")
    gs = GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.08, 1.0, 1.0, 1.0],
        wspace=0.18,
        hspace=0.3,
    )

    ax_map_xy = fig.add_subplot(gs[0, 0])
    ax_map_yz = fig.add_subplot(gs[0, 1])
    ax_goes = fig.add_subplot(gs[1, :])
    ax_omni = fig.add_subplot(gs[2, :], sharex=ax_goes)
    ax_mms_v = fig.add_subplot(gs[3, :], sharex=ax_goes)

    ax_goes_r = ax_goes.twinx()
    ax_omni_r = ax_omni.twinx()
    ax_mms_v_r = ax_mms_v.twinx()
    ax_goes_ecl = ax_goes.twinx()

    mms1_only = {key: value for key, value in mec_data.items() if key.lower() == "mms1"}
    _plot_location_schematic(ax_map_xy, mms1_only, goes_sat, toi, plane="xy")
    _plot_location_schematic(ax_map_yz, mms1_only, goes_sat, toi, plane="yz")
    _plot_goes_panel(ax_goes, ax_goes_r, ax_goes_ecl, goes_sat, eclipse_obscuration, omni_sat, feature_window)
    _plot_omni_panel(ax_omni, ax_omni_r, omni_sat, feature_window)
    _plot_mms_velocity_panel(ax_mms_v, ax_mms_v_r, fields_data.get("mms1"), feature_window)

    for ax in (ax_goes, ax_omni, ax_mms_v):
        ax.set_xlim(start, end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="x", which="both", direction="in", labelsize=PANEL_FONT_SIZE)
    ax_mms_v.tick_params(axis="x", labelbottom=True)
    ax_mms_v.set_xlabel("UT")

    for ax, letter in zip((ax_map_xy, ax_map_yz, ax_goes, ax_omni, ax_mms_v), "ABCDE", strict=False):
        _panel_letter(ax, letter)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
