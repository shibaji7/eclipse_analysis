"""Plotting helpers for MMS location and time-series figures."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from .paths import ensure_repo_paths
from .loaders import MMSProbeData, MMSSeries
from .runtime import configure_runtime
from publication_style import apply_publication_style


ensure_repo_paths()
configure_runtime()


PROBE_COLORS = {
    "mms1": "#d1495b",
    "mms2": "#2a9d8f",
    "mms3": "#457b9d",
    "mms4": "#f4a261",
}

EARTH_RADIUS_KM = 6371.0


@dataclass
class SeriesInput:
    times: np.ndarray
    values: np.ndarray
    label: str
    units: str = ""


def _as_datetime(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    from dateutil import parser as dparser

    return dparser.parse(str(value))


def _nearest_index(times: np.ndarray, target: dt.datetime) -> int:
    diffs = np.array([abs((t - target).total_seconds()) for t in times])
    return int(np.argmin(diffs))


def _sample_series(series: MMSSeries, target: dt.datetime):
    idx = _nearest_index(series.times, target)
    values = np.asarray(series.values)
    if values.ndim == 1:
        return float(values[idx])
    return np.asarray(values[idx])


def _position_to_re(vec: np.ndarray) -> np.ndarray:
    """Convert MEC position vectors to Earth radii when they look like km."""

    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return arr
    magnitude = np.nanmax(np.abs(arr))
    if magnitude > 1_000.0:
        return arr / EARTH_RADIUS_KM
    return arr


def _approx_mlt(x: float, y: float) -> float:
    return (np.degrees(np.arctan2(y, x)) / 15.0 + 12.0) % 24.0


def plot_mms_location_schematic(
    positions: dict[str, MMSProbeData],
    time_of_interest: dt.datetime | str,
    output_path: str | Path,
) -> dict[str, dict[str, float]]:
    """Plot a not-to-scale GSM schematic showing probe locations."""

    apply_publication_style(font_size=8)
    toi = _as_datetime(time_of_interest)
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=300)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f3ec")

    extent = 1.15
    if positions:
        radii = []
        for probe_data in positions.values():
            if probe_data.has("r"):
                r = probe_data.get("r")
                xy = np.asarray(r.values)
                if xy.ndim == 2 and xy.shape[1] >= 2:
                    radii.append(np.nanmax(np.linalg.norm(_position_to_re(xy)[:, :2], axis=1)))
        if radii:
            extent = max(10.0, min(30.0, max(radii) * 1.15))

    earth = Circle((0, 0), radius=1.0, facecolor="#1d3557", edgecolor="black", lw=1.0, zorder=2)
    ax.add_patch(earth)
    ax.text(0, 0, "Earth", color="white", ha="center", va="center", fontsize=11, weight="bold", zorder=3)
    ax.annotate(
        "Sun",
        xy=(extent * 0.82, 0.0),
        xytext=(extent * 0.35, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(extent * 0.78, 1.0, "Dayside", ha="center", va="bottom", fontsize=9)
    ax.text(-extent * 0.78, 1.0, "Nightside", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0, color="0.75", lw=0.8, zorder=0)

    summary: dict[str, dict[str, float]] = {}

    def _draw_track(target_ax, xvals, yvals, color, zorder=3, alpha=0.55, lw=1.0):
        if len(xvals) < 2:
            return
        target_ax.plot(xvals, yvals, color=color, lw=lw, alpha=alpha, zorder=zorder)
        target_ax.scatter([xvals[0]], [yvals[0]], s=18, color=color, edgecolor="black", linewidth=0.4, zorder=zorder + 1, marker="o")
        target_ax.scatter([xvals[-1]], [yvals[-1]], s=24, color=color, edgecolor="black", linewidth=0.5, zorder=zorder + 1, marker="s")

    for probe_name, probe_data in positions.items():
        if not probe_data.has("r"):
            continue
        r = probe_data.get("r")
        track = _position_to_re(np.asarray(r.values))
        if track.ndim != 2 or track.shape[1] < 2:
            vec = _position_to_re(_sample_series(r, toi))
            if vec.ndim == 0 or vec.size < 2:
                continue
            x, y = float(vec[0]), float(vec[1])
            x_track = np.asarray([x], dtype=float)
            y_track = np.asarray([y], dtype=float)
        else:
            x_track = np.asarray(track[:, 0], dtype=float)
            y_track = np.asarray(track[:, 1], dtype=float)
            idx_toi = _nearest_index(r.times, toi)
            x = float(x_track[idx_toi])
            y = float(y_track[idx_toi])
        r_re = float(np.linalg.norm(track[idx_toi, :3])) if track.ndim == 2 and track.shape[1] >= 3 else float(np.linalg.norm([x, y]))
        mlt = _approx_mlt(x, y)
        color = PROBE_COLORS.get(probe_name, "C0")
        _draw_track(ax, x_track, y_track, color)
        ax.scatter([x], [y], s=65, color=color, edgecolor="black", zorder=4, label=probe_name.upper())
        ax.plot([0, x], [0, y], color=color, lw=0.9, alpha=0.8, zorder=3)
        ax.text(
            x + 0.35,
            y + 0.35,
            f"{probe_name.upper()}\nR={r_re:.1f} Re\nMLT={mlt:.1f}",
            fontsize=8,
            color=color,
            ha="left",
            va="bottom",
            zorder=5,
        )
        summary[probe_name] = {
            "x": x,
            "y": y,
            "r_re": r_re,
            "mlt": mlt,
        }

    if len(summary) >= 2:
        xs = np.array([item["x"] for item in summary.values()], dtype=float)
        ys = np.array([item["y"] for item in summary.values()], dtype=float)
        cx = float(np.nanmean(xs))
        cy = float(np.nanmean(ys))
        spread = float(max(np.nanmax(np.abs(xs - cx)), np.nanmax(np.abs(ys - cy))))
        inset_span = max(0.35, spread * 4.0)
        inset = inset_axes(ax, width="36%", height="36%", loc="lower left", borderpad=1.1)
        inset.set_facecolor("white")
        inset.axhline(0, color="0.85", lw=0.6, zorder=0)
        inset.axvline(0, color="0.85", lw=0.6, zorder=0)
        inset.scatter([cx], [cy], s=12, color="#999999", edgecolor="none", zorder=1)
        for probe_name, probe_data in positions.items():
            if not probe_data.has("r"):
                continue
            r = probe_data.get("r")
            track = _position_to_re(np.asarray(r.values))
            color = PROBE_COLORS.get(probe_name, "C0")
            if track.ndim == 2 and track.shape[1] >= 2:
                _draw_track(inset, track[:, 0], track[:, 1], color, zorder=2, alpha=0.75, lw=0.8)
            if probe_name in summary:
                item = summary[probe_name]
                inset.scatter([item["x"]], [item["y"]], s=42, color=color, edgecolor="black", linewidth=0.5, zorder=3)
                inset.text(
                    item["x"] + inset_span * 0.03,
                    item["y"] + inset_span * 0.03,
                    probe_name.upper(),
                    fontsize=6,
                    color=color,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )
        inset.set_xlim(cx - inset_span, cx + inset_span)
        inset.set_ylim(cy - inset_span, cy + inset_span)
        inset.set_aspect("equal")
        inset.tick_params(labelsize=6, length=2)
        inset.set_title("MMS zoom", fontsize=7)
        inset.set_xlabel("X [Re]", fontsize=6)
        inset.set_ylabel("Y [Re]", fontsize=6)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("GSM X [Re]")
    ax.set_ylabel("GSM Y [Re]")
    ax.set_title(f"MMS location schematic at {toi:%Y-%m-%d %H:%M UT}")
    ax.text(0.02, 0.02, "Not to scale", transform=ax.transAxes, fontsize=8, weight="bold", ha="left", va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_mms_location_planes(
    positions: dict[str, MMSProbeData],
    time_of_interest: dt.datetime | str,
    output_path: str | Path,
) -> dict[str, dict[str, float]]:
    """Plot MMS trajectories in the XY, XZ, and YZ GSM planes."""

    apply_publication_style(font_size=8)
    toi = _as_datetime(time_of_interest)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8), dpi=300)
    plane_specs = [
        ("GSM X-Y", 0, 1, "X [Re]", "Y [Re]"),
        ("GSM X-Z", 0, 2, "X [Re]", "Z [Re]"),
        ("GSM Y-Z", 1, 2, "Y [Re]", "Z [Re]"),
    ]

    summary: dict[str, dict[str, float]] = {}
    all_tracks: list[np.ndarray] = []

    for probe_name, probe_data in positions.items():
        if not probe_data.has("r"):
            continue
        r = probe_data.get("r")
        track = _position_to_re(np.asarray(r.values))
        if track.ndim != 2 or track.shape[1] < 3:
            continue
        all_tracks.append(track[:, :3])
        idx_toi = _nearest_index(r.times, toi)
        summary[probe_name] = {
            "x": float(track[idx_toi, 0]),
            "y": float(track[idx_toi, 1]),
            "z": float(track[idx_toi, 2]),
            "r_re": float(np.linalg.norm(track[idx_toi, :3])),
            "mlt": _approx_mlt(float(track[idx_toi, 0]), float(track[idx_toi, 1])),
        }

    if all_tracks:
        stacked = np.vstack(all_tracks)
        mins = np.nanmin(stacked, axis=0)
        maxs = np.nanmax(stacked, axis=0)
        spans = maxs - mins
        margins = np.maximum(spans * 0.25, np.array([0.1, 0.1, 0.1]))
    else:
        mins = np.array([-1.0, -1.0, -1.0])
        maxs = np.array([1.0, 1.0, 1.0])
        margins = np.array([0.5, 0.5, 0.5])

    for ax, (title, i0, i1, xlabel, ylabel) in zip(axes, plane_specs):
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f3ec")
        ax.axhline(0, color="0.8", lw=0.7, zorder=0)
        ax.axvline(0, color="0.8", lw=0.7, zorder=0)
        ax.add_patch(Circle((0, 0), radius=1.0, facecolor="#1d3557", edgecolor="black", lw=1.0, zorder=1))
        ax.text(0, 0, "Earth", color="white", ha="center", va="center", fontsize=9, weight="bold", zorder=2)

        for probe_name, probe_data in positions.items():
            if not probe_data.has("r"):
                continue
            r = probe_data.get("r")
            track = _position_to_re(np.asarray(r.values))
            if track.ndim != 2 or track.shape[1] < 3:
                continue
            xvals = np.asarray(track[:, i0], dtype=float)
            yvals = np.asarray(track[:, i1], dtype=float)
            color = PROBE_COLORS.get(probe_name, "C0")
            ax.plot(xvals, yvals, color=color, lw=1.0, alpha=0.8, zorder=3)
            ax.scatter([xvals[0]], [yvals[0]], s=20, color=color, edgecolor="black", linewidth=0.4, marker="o", zorder=4)
            ax.scatter([xvals[-1]], [yvals[-1]], s=26, color=color, edgecolor="black", linewidth=0.5, marker="s", zorder=4)
            idx_toi = _nearest_index(r.times, toi)
            ax.scatter([xvals[idx_toi]], [yvals[idx_toi]], s=70, color=color, edgecolor="black", linewidth=0.8, zorder=5, label=probe_name.upper())

        if all_tracks:
            plane_points = np.concatenate([track[:, [i0, i1]] for track in all_tracks], axis=0)
            plane_min = np.nanmin(plane_points, axis=0)
            plane_max = np.nanmax(plane_points, axis=0)
            center = 0.5 * (plane_min + plane_max)
            span = float(max(np.max(plane_max - plane_min), 0.5))
            pad = 0.15 * span
            ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
            ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
        else:
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.02,
            0.02,
            "Trajectories shown over interval",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.grid(alpha=0.2, lw=0.5)

    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.16, top=0.88, wspace=0.24)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return summary


def _plot_multiline_panel(ax, datasets, key, ylabel, ylim=None, add_legend=False):
    any_data = False
    for probe_name, probe_data in datasets.items():
        if key not in probe_data.series:
            continue
        series = probe_data.series[key]
        values = np.asarray(series.values)
        times = series.times
        color = PROBE_COLORS.get(probe_name, None)
        if values.ndim == 1:
            ax.plot(times, values, color=color, lw=0.9, label=probe_name.upper())
        else:
            for idx, comp_label in enumerate(("x", "y", "z")):
                if idx < values.shape[1]:
                    ax.plot(times, values[:, idx], color=color, lw=0.9, label=f"{probe_name.upper()} {comp_label}" if idx == 0 else None)
        any_data = True
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if add_legend and any_data:
        ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right")
    return any_data


def _parse_totality_window(obscuration: SeriesInput | dict | None):
    if obscuration is None:
        return None
    if isinstance(obscuration, dict):
        times = np.asarray(obscuration["times"])
        values = np.asarray(obscuration["values"])
    else:
        times = np.asarray(obscuration.times)
        values = np.asarray(obscuration.values)
    if len(times) == 0:
        return None
    mask = values >= 0.99
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    return times[idx[0]], times[idx[-1]]


def plot_mms_timeseries(
    fields_plasma: dict[str, MMSProbeData],
    eclipse_obscuration: SeriesInput | dict,
    al_sml_index: SeriesInput | dict | None,
    time_range: tuple[dt.datetime | str, dt.datetime | str],
    output_path: str | Path,
    goes_data: dict[str, object] | None = None,
    lanl_data: dict[str, object] | None = None,
) -> None:
    """Render the stacked MMS figure."""

    apply_publication_style(font_size=8)
    start = _as_datetime(time_range[0])
    end = _as_datetime(time_range[1])

    if isinstance(eclipse_obscuration, dict):
        obsc_t = np.asarray(eclipse_obscuration["times"])
        obsc_v = np.asarray(eclipse_obscuration["values"])
    else:
        obsc_t = np.asarray(eclipse_obscuration.times)
        obsc_v = np.asarray(eclipse_obscuration.values)

    panels = [
        ("Occulted area [10^6 km^2]", None),
        ("FGM Bx [nT]", "fgm_b"),
        ("FGM By [nT]", "fgm_b"),
        ("FGM Bz [nT]", "fgm_b"),
        ("EDP E GSE [mV/m]", "edp_e"),
        ("E x B drift [km/s]", "exb"),
        ("Ion bulk velocity GSE [km/s]", "ion_bulkv"),
        ("Ion density [cm^-3]", "ion_numberdensity"),
        ("Ion temperature [eV]", "ion_temppara"),
        ("GOES MAG H proxy [nT]", None),
        ("GOES particle flux", None),
        ("LANL SOPA particle flux", None),
    ]
    if al_sml_index is not None:
        panels.append(("AL/SML", None))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 1.95 * len(panels)), sharex=True, dpi=300)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, key) in zip(axes, panels):
        if title == "Occulted area [10^6 km^2]":
            ax.plot(obsc_t, obsc_v, color="black", lw=1.2)
            ax.set_ylabel("Occulted area")
        elif title == "AL/SML" and al_sml_index is not None:
            if isinstance(al_sml_index, dict):
                idx_t = np.asarray(al_sml_index["times"])
                idx_v = np.asarray(al_sml_index["values"])
            else:
                idx_t = np.asarray(al_sml_index.times)
                idx_v = np.asarray(al_sml_index.values)
            ax.plot(idx_t, idx_v, color="#6a1b9a", lw=1.0)
            ax.set_ylabel("AL/SML")
        elif title == "GOES MAG H proxy [nT]":
            plotted = False
            if goes_data:
                for sat_name, sat_data in goes_data.items():
                    series = sat_data.series.get("mag_h") or sat_data.series.get("mag_total")
                    if series is None:
                        continue
                    ax.plot(series.times, np.asarray(series.values), color="#264653", lw=1.0, label=sat_name)
                    plotted = True
                if plotted:
                    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
            ax.set_ylabel("GOES MAG")
        elif title == "GOES particle flux":
            plotted = False
            if goes_data:
                for sat_name, sat_data in goes_data.items():
                    series = sat_data.series.get("particle_flux")
                    if series is None:
                        continue
                    values = np.asarray(series.values, dtype=float)
                    values = np.where(values > 0, values, np.nan)
                    ax.semilogy(series.times, values, color="#e76f51", lw=1.0, label=sat_name)
                    plotted = True
                if plotted:
                    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
            ax.set_ylabel("Flux")
            ax.set_yscale("log")
        elif title == "LANL SOPA particle flux":
            plotted = False
            if lanl_data:
                for sat_name, sat_data in lanl_data.items():
                    flux_series = []
                    for key_name, series in sat_data.series.items():
                        if re.search(r"flux|count", key_name, re.IGNORECASE):
                            flux_series.append((key_name, series))
                    if not flux_series:
                        flux_series = list(sat_data.series.items())[:1]
                    for idx, (key_name, series) in enumerate(flux_series[:3]):
                        values = np.asarray(series.values, dtype=float)
                        values = np.where(values > 0, values, np.nan)
                        linestyle = ("-", "--", ":")[idx % 3]
                        ax.semilogy(
                            series.times,
                            values,
                            color="#8d99ae",
                            lw=0.9,
                            linestyle=linestyle,
                            alpha=0.95,
                            label=f"{sat_name} {key_name}" if idx == 0 else None,
                        )
                        plotted = True
                if plotted:
                    ax.legend(ncol=2, fontsize=6.5, frameon=False, loc="upper right")
            ax.set_ylabel("Flux")
            ax.set_yscale("log")
        else:
            if key == "fgm_b":
                for probe_name, probe_data in fields_plasma.items():
                    if "b" not in probe_data.series:
                        continue
                    series = probe_data.series["b"]
                    values = np.asarray(series.values)
                    idx = {"FGM Bx [nT]": 0, "FGM By [nT]": 1, "FGM Bz [nT]": 2}[title]
                    if values.ndim == 2 and values.shape[1] > idx:
                        ax.plot(series.times, values[:, idx], color=PROBE_COLORS.get(probe_name, "C0"), lw=0.9, label=probe_name.upper())
                if title == "FGM Bx [nT]":
                    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right")
            elif key == "edp_e":
                for probe_name, probe_data in fields_plasma.items():
                    series = probe_data.series.get("e")
                    if series is None:
                        continue
                    values = np.asarray(series.values)
                    if values.ndim == 2:
                        color = PROBE_COLORS.get(probe_name, "C0")
                        comp_styles = ("-", "--", ":")
                        comp_labels = ("Ex", "Ey", "Ez")
                        for comp_idx, (comp_label, comp_style) in enumerate(zip(comp_labels, comp_styles, strict=False)):
                            if values.shape[1] > comp_idx:
                                ax.plot(
                                    series.times,
                                    values[:, comp_idx],
                                    color=color,
                                    lw=0.9,
                                    linestyle=comp_style,
                                    alpha=0.9,
                                    label=f"{probe_name.upper()} {comp_label}",
                                )
                ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper right")
            elif key == "exb":
                for probe_name, probe_data in fields_plasma.items():
                    series = probe_data.series.get("exb")
                    if series is None:
                        continue
                    values = np.asarray(series.values)
                    if values.ndim == 2 and values.shape[1] >= 3:
                        magnitude = np.linalg.norm(values, axis=1)
                        ax.plot(series.times, magnitude, color=PROBE_COLORS.get(probe_name, "C0"), lw=0.9, label=probe_name.upper())
                ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper right")
            elif key == "ion_bulkv":
                comp_labels = ("Vx", "Vy", "Vz")
                comp_styles = ("-", "--", ":")
                for probe_name, probe_data in fields_plasma.items():
                    series = probe_data.series.get("ion_bulkv")
                    if series is None:
                        continue
                    values = np.asarray(series.values)
                    if values.ndim == 2:
                        color = PROBE_COLORS.get(probe_name, "C0")
                        for comp_idx, (comp_label, comp_style) in enumerate(zip(comp_labels, comp_styles, strict=False)):
                            if values.shape[1] > comp_idx:
                                ax.plot(
                                    series.times,
                                    values[:, comp_idx],
                                    color=color,
                                    lw=0.9,
                                    linestyle=comp_style,
                                    alpha=0.9,
                                    label=f"{probe_name.upper()} {comp_label}",
                                )
                ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper right")
            elif key == "ion_numberdensity":
                for probe_name, probe_data in fields_plasma.items():
                    series = probe_data.series.get("ion_numberdensity")
                    if series is None:
                        continue
                    ax.plot(series.times, np.asarray(series.values), color=PROBE_COLORS.get(probe_name, "C0"), lw=0.9)
            elif key == "ion_temppara":
                for probe_name, probe_data in fields_plasma.items():
                    series = probe_data.series.get("ion_temppara")
                    if series is None:
                        continue
                    ax.plot(series.times, np.asarray(series.values), color=PROBE_COLORS.get(probe_name, "C0"), lw=0.9)
            ax.set_ylabel(title)

        ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlim(start, end)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("UT")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="png", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
