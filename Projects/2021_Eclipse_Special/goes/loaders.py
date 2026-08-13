"""Load GOES-R and LANL geosynchronous corroboration data."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

import numpy as np

from .paths import ensure_repo_paths
from .runtime import configure_runtime


ensure_repo_paths()
runtime_env = configure_runtime()

from pyspedas.projects import goes as pyspedas_goes  # noqa: E402
from pyspedas.tplot_tools import get_data, tplot_names  # noqa: E402


GOES_LONGITUDES = {
    "16": -75.2,
    "17": -137.2,
}

LANL_LOCAL_TIME_SECTORS = {
    "LANL-04A": "pre-midnight",
    "LANL-97A": "evening-midnight",
    "LANL-01A": "post-midnight / morning",
    "LANL-02A": "post-midnight / dawn",
}

LANL_BASE_URL = "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_geo/data/"
GOES_R_BASE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"


@dataclass
class CorroborationSeries:
    """Single scalar or vector series loaded from a tplot variable or ASCII file."""

    times: np.ndarray
    values: np.ndarray
    name: str
    units: str = ""


@dataclass
class CorroborationSatelliteData:
    """Container for per-satellite GOES or LANL products."""

    satellite: str
    series: dict[str, CorroborationSeries] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def add_series(self, key: str, series: CorroborationSeries) -> None:
        self.series[key] = series

    def has(self, key: str) -> bool:
        return key in self.series

    def get(self, key: str, default=None):
        return self.series.get(key, default)


def _as_datetime(value: dt.datetime | str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    from dateutil import parser as dparser

    return dparser.parse(str(value))


def _normalize_trange(time_range: Sequence[dt.datetime | str]) -> list[str]:
    out = []
    for value in time_range:
        if isinstance(value, dt.datetime):
            out.append(value.strftime("%Y-%m-%d/%H:%M:%S"))
        else:
            out.append(str(value))
    return out


def _to_datetime_array(times: Iterable[float | dt.datetime]) -> np.ndarray:
    converted = []
    for value in times:
        if isinstance(value, dt.datetime):
            converted.append(value)
        else:
            converted.append(dt.datetime.utcfromtimestamp(float(value)))
    return np.array(converted, dtype=object)


def _load_names_after_call(loader, **kwargs) -> list[str]:
    before = set(tplot_names())
    loader(**kwargs)
    after = set(tplot_names())
    return sorted(after - before)


def _extract_tplot_series(var_name: str) -> CorroborationSeries | None:
    data = get_data(var_name)
    if data is None:
        return None
    times = _to_datetime_array(data.times)
    values = np.array(data.y)
    units = getattr(data, "units", "") or ""
    return CorroborationSeries(times=times, values=values, name=var_name, units=units)


def _nearest_midnight_distance(hour: float) -> float:
    return min(hour % 24.0, 24.0 - (hour % 24.0))


def select_goes_probe(time_range: Sequence[dt.datetime | str]) -> tuple[str, str]:
    """Pick the GOES-R satellite with the best local-time coverage."""

    start = _as_datetime(time_range[0])
    end = _as_datetime(time_range[1])
    midpoint = start + (end - start) / 2
    ut_hour = midpoint.hour + midpoint.minute / 60.0 + midpoint.second / 3600.0
    candidates = []
    for probe, lon in GOES_LONGITUDES.items():
        local_time = (ut_hour + lon / 15.0) % 24.0
        candidates.append((probe, local_time, _nearest_midnight_distance(local_time)))
    selected_probe, local_time, _ = min(candidates, key=lambda item: item[2])
    reason = (
        f"GOES-{selected_probe} is closest to local midnight at the event midpoint "
        f"({local_time:0.2f} LT from the geostationary longitude)."
    )
    return selected_probe, reason


def select_lanl_probe(time_range: Sequence[dt.datetime | str]) -> tuple[str, str]:
    """Pick the LANL spacecraft most likely to straddle the midnight sector."""

    _ = time_range
    selected_probe = "LANL-04A"
    reason = (
        "LANL-04A is the pre-midnight GEO probe in the active LANL archive and is the "
        "closest match to a 06:00-09:00 UT injection window."
    )
    return selected_probe, reason


def load_goes_r_series(
    time_range: Sequence[dt.datetime | str],
    probe: str | int | None = None,
    time_clip: bool = True,
    no_update: bool = False,
) -> dict[str, CorroborationSatelliteData]:
    """Load GOES-R MAG + SGPS products for the selected spacecraft."""

    trange = _normalize_trange(time_range)
    if probe is None:
        probe, reason = select_goes_probe(time_range)
    else:
        probe = str(probe)
        reason = "GOES probe selected explicitly by the user."

    out = CorroborationSatelliteData(satellite=f"GOES-{probe}", metadata={"selection_reason": reason})

    try:
        mag_names = _load_names_after_call(
            pyspedas_goes.mag,
            probe=probe,
            trange=trange,
            datatype="1min",
            time_clip=time_clip,
            no_update=no_update,
            prefix="probename",
        )
    except Exception as exc:  # pragma: no cover - depends on runtime network/cache
        out.metadata["load_error"] = str(exc)
        local_mag = _load_local_goes_mag_series(probe, time_range)
        if local_mag is not None:
            out.add_series("mag_h", local_mag)
            out.metadata["mag_proxy"] = "local b_vdh[:, 2] (H component proxy)"
        flux = _load_goes_r_sgps_series(probe, time_range)
        if flux is not None:
            out.add_series("particle_flux", flux)
            out.metadata["particle_flux_source"] = flux.name
            out.metadata["particle_flux_note"] = "AvgIntProtonFlux averaged across sensor units"
        return {out.satellite: out}

    b_vdh = None
    b_total = None
    for name in mag_names:
        if name.endswith("_mag_b_vdh"):
            b_vdh = _extract_tplot_series(name)
        elif name.endswith("_mag_b_total"):
            b_total = _extract_tplot_series(name)
    if b_vdh is not None:
        values = np.asarray(b_vdh.values)
        if values.ndim == 2 and values.shape[1] >= 3:
            out.add_series(
                "mag_h",
                CorroborationSeries(
                    times=b_vdh.times,
                    values=np.asarray(values[:, 2], dtype=float),
                    name=b_vdh.name,
                    units=b_vdh.units or "nT",
                ),
            )
            out.metadata["mag_proxy"] = "b_vdh[:, 2] (H component proxy)"
        else:
            out.add_series("mag_total", b_total or b_vdh)
            out.metadata["mag_proxy"] = "b_total"
    elif b_total is not None:
        out.add_series("mag_total", b_total)
        out.metadata["mag_proxy"] = "b_total"
    else:
        local_mag = _load_local_goes_mag_series(probe, time_range)
        if local_mag is not None:
            out.add_series("mag_h", local_mag)
            out.metadata["mag_proxy"] = "local b_vdh[:, 2] (H component proxy)"

    flux = _load_goes_r_sgps_series(probe, time_range)
    if flux is not None:
        out.add_series("particle_flux", flux)
        out.metadata["particle_flux_source"] = flux.name
        out.metadata["particle_flux_note"] = "AvgIntProtonFlux averaged across sensor units"

    return {out.satellite: out}


def _goes_r_sgps_dir(probe: str, value: dt.datetime) -> str:
    return urljoin(
        GOES_R_BASE_URL,
        f"goes{probe}/l2/data/sgps-l2-avg1m/{value:%Y/%m}/",
    )


def _download_goes_r_sgps_file(probe: str, day: dt.date) -> Path | None:
    local_root = Path(runtime_env["GOES_DATA_DIR"]) / f"goes{probe}" / "l2" / "data" / "sgps-l2-avg1m" / day.strftime("%Y") / day.strftime("%m")
    local_root.mkdir(parents=True, exist_ok=True)
    directory_url = _goes_r_sgps_dir(probe, dt.datetime.combine(day, dt.time()))
    day_token = day.strftime("%Y%m%d")
    local_matches = sorted(local_root.glob(f"dn_sgps-l2-avg1m_g{probe}_d{day_token}_v*.nc"))
    if local_matches:
        return local_matches[0]
    remote_name = None
    try:
        listing = _read_text_url(directory_url)
    except (URLError, OSError):
        listing = ""
    pattern = re.compile(rf'href="(dn_sgps-l2-avg1m_g{re.escape(str(probe))}_d{day_token}_v[^"]+\.nc)"')
    match = pattern.search(listing)
    if match:
        remote_name = match.group(1)
    if remote_name is None:
        return None
    local_path = local_root / remote_name
    if not local_path.exists():
        urlretrieve(urljoin(directory_url, remote_name), local_path)
    return local_path


def _datetime64_to_python(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times)
    if values.size == 0:
        return np.array([], dtype=object)
    ns = values.astype("datetime64[ns]").astype("int64")
    converted = [dt.datetime.utcfromtimestamp(int(value) / 1_000_000_000) for value in ns]
    return np.asarray(converted, dtype=object)


def _load_goes_r_sgps_series(probe: str, time_range: Sequence[dt.datetime | str]) -> CorroborationSeries | None:
    start = _as_datetime(time_range[0]).date()
    end = _as_datetime(time_range[1]).date()
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += dt.timedelta(days=1)

    try:
        import xarray as xr
    except Exception:  # pragma: no cover - dependency availability
        return None

    times_all: list[dt.datetime] = []
    values_all: list[float] = []
    for day in days:
        local_path = _download_goes_r_sgps_file(probe, day)
        if local_path is None:
            continue
        with xr.open_dataset(local_path) as ds:
            if "L2_SciData_TimeStamp" not in ds or "AvgIntProtonFlux" not in ds:
                continue
            times = _datetime64_to_python(np.asarray(ds["L2_SciData_TimeStamp"].values))
            flux = np.asarray(ds["AvgIntProtonFlux"].values, dtype=float)
            if flux.ndim == 2:
                flux_1d = np.nanmean(flux, axis=1)
            else:
                flux_1d = np.asarray(flux, dtype=float).reshape(-1)
            times_all.extend(times.tolist())
            values_all.extend(flux_1d.tolist())

    if not times_all:
        return None

    order = np.argsort(np.asarray(times_all, dtype=object))
    times_sorted = np.asarray(times_all, dtype=object)[order]
    values_sorted = np.asarray(values_all, dtype=float)[order]
    return CorroborationSeries(
        times=times_sorted,
        values=values_sorted,
        name=f"g{probe}_sgps_AvgIntProtonFlux",
        units="(file units)",
    )


def _find_local_goes_mag_file(probe: str, day: dt.date) -> Path | None:
    day_token = day.strftime("%Y%m%d")
    pattern = f"dn_magn-l2-avg1m_g{probe}_d{day_token}_v*.nc"
    search_roots = [
        Path(runtime_env["GOES_DATA_DIR"]),
        Path(__file__).resolve().parent,
    ]
    for root in search_roots:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _load_local_goes_mag_series(probe: str, time_range: Sequence[dt.datetime | str]) -> CorroborationSeries | None:
    start = _as_datetime(time_range[0]).date()
    end = _as_datetime(time_range[1]).date()
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += dt.timedelta(days=1)

    try:
        import xarray as xr
    except Exception:  # pragma: no cover - dependency availability
        return None

    times_all: list[dt.datetime] = []
    values_all: list[float] = []
    for day in days:
        local_path = _find_local_goes_mag_file(probe, day)
        if local_path is None:
            continue
        with xr.open_dataset(local_path) as ds:
            if "time" not in ds or "b_vdh" not in ds:
                continue
            times = _datetime64_to_python(np.asarray(ds["time"].values))
            values = np.asarray(ds["b_vdh"].values, dtype=float)
            if values.ndim == 2 and values.shape[1] >= 3:
                h_comp = values[:, 2]
            elif values.ndim == 1:
                h_comp = values
            elif "b_total" in ds:
                h_comp = np.asarray(ds["b_total"].values, dtype=float).reshape(-1)
            else:
                continue
            times_all.extend(times.tolist())
            values_all.extend(np.asarray(h_comp, dtype=float).tolist())

    if not times_all:
        return None

    order = np.argsort(np.asarray(times_all, dtype=object))
    times_sorted = np.asarray(times_all, dtype=object)[order]
    values_sorted = np.asarray(values_all, dtype=float)[order]
    return CorroborationSeries(
        times=times_sorted,
        values=values_sorted,
        name=f"g{probe}_magn_b_vdh",
        units="nT",
    )


def _read_text_url(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8", errors="replace")


def _parse_json_header_text(text: str) -> tuple[dict, list[str]]:
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("LANL file does not contain a JSON header.")

    header_lines = []
    depth = 0
    end_idx = None
    in_string = False
    escape = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        header_lines.append(line)
        for char in line:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break
        if end_idx is not None:
            break

    if end_idx is None:
        raise ValueError("LANL JSON header is malformed.")

    header_text = "\n".join(header_lines)
    header = json.loads(header_text)
    remainder = lines[end_idx + 1 :]
    return header, remainder


def _parse_time_tokens(tokens: list[str]) -> tuple[dt.datetime, int]:
    if not tokens:
        raise ValueError("No time tokens available")

    if re.match(r"^\d{4}-\d{2}-\d{2}", tokens[0]):
        if len(tokens) > 1 and re.match(r"^\d{2}:\d{2}:\d{2}", tokens[1]):
            return dt.datetime.fromisoformat(f"{tokens[0]}T{tokens[1]}"), 2
        return dt.datetime.fromisoformat(tokens[0]), 1

    if len(tokens) >= 6 and all(re.match(r"^-?\d+$", tok) for tok in tokens[:6]):
        year = int(tokens[0])
        month = int(tokens[1])
        day = int(tokens[2])
        hour = int(tokens[3])
        minute = int(tokens[4])
        second = int(float(tokens[5]))
        return dt.datetime(year, month, day, hour, minute, second), 6

    if len(tokens) >= 2 and re.match(r"^\d{8}$", tokens[0]) and re.match(r"^\d{2}:\d{2}:\d{2}", tokens[1]):
        return dt.datetime.strptime(f"{tokens[0]} {tokens[1]}", "%Y%m%d %H:%M:%S"), 2

    if len(tokens) >= 3 and re.match(r"^\d{4}$", tokens[0]) and re.match(r"^\d{3}$", tokens[1]):
        year = int(tokens[0])
        doy = int(tokens[1])
        seconds = float(tokens[2])
        base = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1, seconds=seconds)
        return base, 3

    raise ValueError(f"Unrecognized LANL time tokens: {tokens[:6]}")


def _parse_lanl_ascii(text: str) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    header, remainder = _parse_json_header_text(text)

    data_lines = []
    column_names = None
    for line in remainder:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if column_names is None:
            non_numeric = [tok for tok in tokens if re.search(r"[A-Za-z_]", tok)]
            if non_numeric and not re.match(r"^\d{4}", tokens[0]):
                column_names = tokens
                continue
        data_lines.append(tokens)

    if not data_lines:
        return header, np.array([], dtype=object), {}

    if column_names is None:
        ncols = len(data_lines[0])
        column_names = [f"col_{idx}" for idx in range(ncols)]

    times: list[dt.datetime] = []
    columns: dict[str, list[float]] = {name: [] for name in column_names[1:]}

    for tokens in data_lines:
        try:
            time_value, consumed = _parse_time_tokens(tokens)
        except ValueError:
            continue
        times.append(time_value)
        numeric_tokens = tokens[consumed:]
        if len(numeric_tokens) < len(column_names) - 1:
            numeric_tokens = numeric_tokens + ["nan"] * (len(column_names) - 1 - len(numeric_tokens))
        for name, token in zip(column_names[1:], numeric_tokens, strict=False):
            try:
                columns[name].append(float(token))
            except ValueError:
                columns[name].append(np.nan)

    return header, np.asarray(times, dtype=object), {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def _lanl_file_candidates(probe: str, instrument: str, time_range: Sequence[dt.datetime | str]) -> list[str]:
    start = _as_datetime(time_range[0]).date()
    end = _as_datetime(time_range[1]).date()
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += dt.timedelta(days=1)
    filenames = []
    for cur in dates:
        day = cur.strftime("%Y%m%d")
        filenames.append(f"{day}_{probe}_{instrument.upper()}_combined_{'10sec' if instrument.lower() == 'sopa' else '86sec'}.txt")
    return filenames


def _download_lanl_ascii(probe: str, instrument: str, time_range: Sequence[dt.datetime | str]) -> list[Path]:
    local_root = Path(runtime_env["LANL_DATA_DIR"]) / probe / instrument.lower()
    local_root.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for filename in _lanl_file_candidates(probe, instrument, time_range):
        local_path = local_root / filename
        if local_path.exists():
            out.append(local_path)
            continue
        remote_url = urljoin(LANL_BASE_URL, f"{probe}/{instrument.lower()}/{filename}")
        try:
            text = _read_text_url(remote_url)
        except (URLError, OSError, ValueError):
            continue
        local_path.write_text(text, encoding="utf-8")
        out.append(local_path)
    return out


def load_lanl_sopa_series(
    time_range: Sequence[dt.datetime | str],
    probe: str | None = None,
) -> dict[str, CorroborationSatelliteData]:
    """Load LANL SOPA fluxes from the NOAA NCEI JSON-headed ASCII files."""

    if probe is None:
        probe, reason = select_lanl_probe(time_range)
    else:
        reason = "LANL probe selected explicitly by the user."

    out = CorroborationSatelliteData(satellite=probe, metadata={"selection_reason": reason, "instrument": "SOPA"})
    local_files = _download_lanl_ascii(probe, "sopa", time_range)
    if not local_files:
        return {probe: out}

    combined_times: list[dt.datetime] = []
    combined_channels: dict[str, list[float]] = {}
    combined_units: dict[str, str] = {}
    header_blob = {}

    for file_path in local_files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        header, times, columns = _parse_lanl_ascii(text)
        header_blob = header
        if times.size == 0:
            continue
        combined_times.extend(list(times))
        for name, values in columns.items():
            combined_channels.setdefault(name, []).extend(values.tolist())
            if name not in combined_units:
                unit = ""
                if isinstance(header, dict):
                    fields = header.get("fields") or header.get("columns") or []
                    if isinstance(fields, list):
                        for field in fields:
                            if isinstance(field, dict) and field.get("name") == name:
                                unit = field.get("units", "") or ""
                                break
                combined_units[name] = unit

    if not combined_times:
        return {probe: out}

    order = np.argsort(np.asarray(combined_times, dtype=object))
    times_sorted = np.asarray(combined_times, dtype=object)[order]
    for name, values in combined_channels.items():
        arr = np.asarray(values, dtype=float)[order]
        out.add_series(
            name,
            CorroborationSeries(
                times=times_sorted,
                values=arr,
                name=name,
                units=combined_units.get(name, ""),
            ),
        )

    out.metadata["header"] = json.dumps(header_blob, sort_keys=True)[:2000]
    return {probe: out}
