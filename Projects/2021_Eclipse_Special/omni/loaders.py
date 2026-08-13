"""OMNI solar wind and IMF loaders for the eclipse analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .paths import ensure_repo_paths
from .runtime import configure_runtime


ensure_repo_paths()
runtime_env = configure_runtime()

try:  # pragma: no cover - optional dependency
    import pyomnidata
except Exception:  # pragma: no cover - optional dependency
    pyomnidata = None

from pyspedas.tplot_tools import get_data, tplot_names  # noqa: E402


@dataclass
class OmniSeries:
    """Single OMNI time series."""

    times: np.ndarray
    values: np.ndarray
    name: str
    units: str = ""


@dataclass
class OmniData:
    """Container for the OMNI 1-minute products used in the plot."""

    series: dict[str, OmniSeries] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def add_series(self, key: str, series: OmniSeries) -> None:
        self.series[key] = series


def _as_datetime(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    from dateutil import parser as dparser

    return dparser.parse(str(value))


def _normalize_trange(time_range: Sequence[dt.datetime | str]) -> list[dt.datetime]:
    return [_as_datetime(value) for value in time_range]


def _datetime_array(times: Iterable[dt.datetime]) -> np.ndarray:
    return np.asarray([t.replace(tzinfo=None) if getattr(t, "tzinfo", None) else t for t in times], dtype=object)


def _select_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _df_times_from_pyomnidata(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_numeric(df["Date"], errors="coerce").astype("Int64")
    ut = pd.to_numeric(df["ut"], errors="coerce")

    def _row_time(date_value, ut_value):
        if pd.isna(date_value) or pd.isna(ut_value):
            return pd.NaT
        date_str = f"{int(date_value):08d}"
        base = dt.datetime.strptime(date_str, "%Y%m%d")
        return base + dt.timedelta(hours=float(ut_value))

    return pd.Series([_row_time(d, u) for d, u in zip(dates, ut, strict=False)])


def _compute_pdyn(density_cm3: np.ndarray, speed_kms: np.ndarray) -> np.ndarray:
    return 1.6726e-6 * density_cm3 * np.square(speed_kms)


def _build_omni_data_frame(year: int, res: int = 1) -> pd.DataFrame:
    if pyomnidata is None:
        raise ImportError("pyomnidata is not available")
    raw = pd.DataFrame(pyomnidata.GetOMNI(year, Res=res))
    if raw.empty:
        return raw
    raw["time"] = _df_times_from_pyomnidata(raw)
    return raw


def _series_from_frame(df: pd.DataFrame, column: str, name: str, units: str = "") -> OmniSeries | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    times = np.asarray(df["time"].to_numpy(dtype=object))
    if not np.any(mask):
        return None
    return OmniSeries(times=times[mask], values=values[mask], name=name, units=units)


def _combine_series(series_list: list[OmniSeries | None], name: str, units: str = "") -> OmniSeries | None:
    items = [series for series in series_list if series is not None and len(series.times) > 0]
    if not items:
        return None
    times = np.concatenate([np.asarray(item.times, dtype=object) for item in items])
    values = np.concatenate([np.asarray(item.values, dtype=float) for item in items])
    order = np.argsort(times)
    return OmniSeries(times=times[order], values=values[order], name=name, units=units)


def _extract_from_pyomnidata(time_range: Sequence[dt.datetime | str], res: int = 1) -> OmniData:
    start, end = _normalize_trange(time_range)
    frames = []
    for year in range(start.year, end.year + 1):
        frame = _build_omni_data_frame(year, res=res)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("pyomnidata returned no OMNI rows.")

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    if df.empty:
        raise ValueError("pyomnidata OMNI rows do not overlap the requested time range.")

    data = OmniData(metadata={"source": "pyomnidata", "resolution": f"{res} min"})
    bx_col = _select_column(df, ["BxGSE"])
    by_col = _select_column(df, ["ByGSM", "ByGSE"])
    bz_col = _select_column(df, ["BzGSM", "BzGSE"])
    bmag_col = _select_column(df, ["B"])
    speed_col = _select_column(df, ["FlowSpeed"])
    density_col = _select_column(df, ["ProtonDensity"])
    pdyn_col = _select_column(df, ["FlowPressure"])
    ae_col = _select_column(df, ["AE"])
    al_col = _select_column(df, ["AL"])

    bx = _series_from_frame(df, bx_col, "bx_gse", "nT") if bx_col else None
    by = _series_from_frame(df, by_col, "by_gsm", "nT") if by_col else None
    bz = _series_from_frame(df, bz_col, "bz_gsm", "nT") if bz_col else None
    bmag = _series_from_frame(df, bmag_col, "b_mag", "nT") if bmag_col else None
    speed = _series_from_frame(df, speed_col, "speed", "km/s") if speed_col else None
    density = _series_from_frame(df, density_col, "density", "cm^-3") if density_col else None
    pdyn = _series_from_frame(df, pdyn_col, "pdyn", "nPa") if pdyn_col else None
    ae = _series_from_frame(df, ae_col, "ae", "nT") if ae_col else None
    al = _series_from_frame(df, al_col, "al", "nT") if al_col else None

    if bmag is None and bx is not None and by is not None and bz is not None:
        common_times = bx.times
        bvec = np.column_stack([bx.values, by.values, bz.values])
        bmag = OmniSeries(times=common_times, values=np.linalg.norm(bvec, axis=1), name="b_mag", units="nT")

    if pdyn is None and density is not None and speed is not None:
        pdyn = OmniSeries(times=speed.times, values=_compute_pdyn(density.values, speed.values), name="pdyn", units="nPa")

    for key, series in (
        ("bx_gse", bx),
        ("by_gsm", by),
        ("bz_gsm", bz),
        ("b_mag", bmag),
        ("speed", speed),
        ("density", density),
        ("pdyn", pdyn),
        ("ae", ae),
        ("al", al),
    ):
        if series is not None:
            data.add_series(key, series)

    if pdyn_col:
        data.metadata["pdyn_source"] = pdyn_col
    elif density is not None and speed is not None:
        data.metadata["pdyn_source"] = "computed from ProtonDensity and FlowSpeed"
    if ae_col:
        data.metadata["ae_source"] = ae_col
    if al_col:
        data.metadata["al_source"] = al_col
    if bmag_col:
        data.metadata["b_mag_source"] = bmag_col
    elif bx is not None and by is not None and bz is not None:
        data.metadata["b_mag_source"] = "computed from BxGSE/By/Bz"

    return data


def _extract_from_pyspedas(time_range: Sequence[dt.datetime | str], res: int = 1, no_update: bool = False) -> OmniData:
    ensure_repo_paths()
    from pyspedas.projects import omni as pyspedas_omni  # noqa: E402

    trange = [_as_datetime(time_range[0]).strftime("%Y-%m-%d/%H:%M:%S"), _as_datetime(time_range[1]).strftime("%Y-%m-%d/%H:%M:%S")]
    before = set(tplot_names())
    pyspedas_omni.data(trange=trange, datatype="1min", level="hro2", time_clip=True, no_update=no_update)
    loaded = sorted(set(tplot_names()) - before)
    if not loaded:
        pyspedas_omni.data(trange=trange, datatype="1min", level="hro", time_clip=True, no_update=no_update)
        loaded = sorted(set(tplot_names()) - before)
    if not loaded:
        raise ValueError("pyspedas OMNI loader returned no tplot variables.")

    def _pick(patterns: Sequence[str]) -> str | None:
        for pattern in patterns:
            for name in loaded:
                if pattern.lower() in name.lower():
                    return name
        return None

    def _series(name: str | None, key: str, units: str = "") -> OmniSeries | None:
        if name is None:
            return None
        data = get_data(name)
        if data is None:
            return None
        times = np.asarray([dt.datetime.utcfromtimestamp(float(t)) for t in data.times], dtype=object)
        values = np.asarray(data.y, dtype=float)
        if values.ndim > 1:
            values = values.reshape(-1)
        mask = np.isfinite(values)
        return OmniSeries(times=times[mask], values=values[mask], name=key, units=units)

    bx = _series(_pick(["bx_gse"]), "bx_gse", "nT")
    by = _series(_pick(["by_gsm"]), "by_gsm", "nT") or _series(_pick(["by_gse"]), "by_gsm", "nT")
    bz = _series(_pick(["bz_gsm"]), "bz_gsm", "nT") or _series(_pick(["bz_gse"]), "bz_gsm", "nT")
    bmag = _series(_pick([" b ", "b_tot", "bmag"]), "b_mag", "nT")
    speed = _series(_pick(["flowspeed", "speed"]), "speed", "km/s")
    density = _series(_pick(["protondensity", "density", "np"]), "density", "cm^-3")
    pdyn = _series(_pick(["pressure", "pdyn"]), "pdyn", "nPa")
    ae = _series(_pick(["ae"]), "ae", "nT")
    al = _series(_pick(["al"]), "al", "nT")
    if pdyn is None and density is not None and speed is not None:
        pdyn = OmniSeries(times=speed.times, values=_compute_pdyn(density.values, speed.values), name="pdyn", units="nPa")
    if bmag is None and bx is not None and by is not None and bz is not None:
        bmag = OmniSeries(times=bx.times, values=np.linalg.norm(np.column_stack([bx.values, by.values, bz.values]), axis=1), name="b_mag", units="nT")

    data = OmniData(metadata={"source": "pyspedas", "level": "hro2_or_hro"})
    for key, series in (
        ("bx_gse", bx),
        ("by_gsm", by),
        ("bz_gsm", bz),
        ("b_mag", bmag),
        ("speed", speed),
        ("density", density),
        ("pdyn", pdyn),
        ("ae", ae),
        ("al", al),
    ):
        if series is not None:
            data.add_series(key, series)
    return data


def load_omni_1min_series(
    time_range: Sequence[dt.datetime | str],
    no_update: bool = False,
) -> dict[str, OmniData]:
    """Load 1-minute OMNI solar wind and IMF data for the requested time range."""

    try:
        data = _extract_from_pyomnidata(time_range, res=1)
    except Exception as pyomnidata_exc:
        try:
            data = _extract_from_pyspedas(time_range, res=1, no_update=no_update)
            data.metadata["fallback"] = f"pyomnidata failed: {pyomnidata_exc}"
        except Exception as pyspedas_exc:
            raise RuntimeError(f"Unable to load OMNI data: pyomnidata={pyomnidata_exc}; pyspedas={pyspedas_exc}") from pyspedas_exc

    return {"OMNI": data}
