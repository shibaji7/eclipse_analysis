"""pyspedas-backed MMS loaders used for the eclipse analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .paths import ensure_repo_paths
from .runtime import configure_runtime


ensure_repo_paths()
runtime_env = configure_runtime()

from pyspedas.projects import mms as pyspedas_mms  # noqa: E402
from pyspedas.projects.mms import mms_config  # noqa: E402
from pyspedas.tplot_tools import get_data, tplot_names  # noqa: E402


mms_config.CONFIG["local_data_dir"] = runtime_env["MMS_DATA_DIR"]
mms_config.CONFIG["mirror_data_dir"] = None


@dataclass
class MMSSeries:
    """Single scalar or vector series loaded from a tplot variable."""

    times: np.ndarray
    values: np.ndarray
    name: str
    units: str = ""


@dataclass
class MMSProbeData:
    """Container for per-probe MMS products."""

    probe: str
    series: dict[str, MMSSeries] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def add_series(self, key: str, series: MMSSeries) -> None:
        self.series[key] = series

    def has(self, key: str) -> bool:
        return key in self.series

    def get(self, key: str, default=None):
        return self.series.get(key, default)


def _normalize_probe(probe: str | int) -> str:
    return str(probe).replace("mms", "").strip()


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


def _extract_series(var_name: str) -> MMSSeries | None:
    data = get_data(var_name)
    if data is None:
        return None
    times = _to_datetime_array(data.times)
    values = np.array(data.y)
    units = getattr(data, "units", "") or ""
    return MMSSeries(times=times, values=values, name=var_name, units=units)


def _pick_name(names: Sequence[str], patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        for name in names:
            if re.search(pattern, name):
                return name
    return None


def _extract_mec_dataset(probe: str, names: Sequence[str]) -> MMSProbeData:
    out = MMSProbeData(probe=probe)
    prefix = f"mms{probe}_"

    r_name = _pick_name(
        names,
        [
            rf"^{prefix}mec_r_gsm$",
            rf"^{prefix}mec_r_gse$",
            rf"^{prefix}mec_r_.*gsm.*$",
            rf"^{prefix}mec_r_.*gse.*$",
        ],
    )
    v_name = _pick_name(
        names,
        [
            rf"^{prefix}mec_v_gsm$",
            rf"^{prefix}mec_v_gse$",
            rf"^{prefix}mec_v_.*gsm.*$",
            rf"^{prefix}mec_v_.*gse.*$",
        ],
    )
    for key, name in (("r", r_name), ("v", v_name)):
        if name:
            series = _extract_series(name)
            if series is not None:
                out.add_series(key, series)

    # MEC MLT is not always exposed as a direct tplot variable, so compute a
    # first-pass approximation from the GSM xy-plane if needed.
    if out.has("r"):
        r = out.get("r").values
        if r.ndim == 2 and r.shape[1] >= 2:
            mlt = (np.degrees(np.arctan2(r[:, 1], r[:, 0])) / 15.0 + 12.0) % 24.0
            out.add_series(
                "mlt",
                MMSSeries(
                    times=out.get("r").times,
                    values=mlt,
                    name=f"mms{probe}_approx_mlt",
                    units="hours",
                ),
            )
    return out


def _extract_fgm_dataset(probe: str, names: Sequence[str]) -> MMSProbeData:
    out = MMSProbeData(probe=probe)
    prefix = f"mms{probe}_fgm_"
    b_name = _pick_name(
        names,
        [
            rf"^{prefix}b_gsm_.*$",
            rf"^{prefix}b_gse_.*$",
            rf"^{prefix}b_.*gsm.*$",
            rf"^{prefix}b_.*gse.*$",
        ],
    )
    if b_name:
        series = _extract_series(b_name)
        if series is not None:
            out.add_series("b", series)
    return out


def _extract_edp_dataset(probe: str, names: Sequence[str]) -> MMSProbeData:
    out = MMSProbeData(probe=probe)
    prefix = f"mms{probe}_edp_"
    e_name = _pick_name(
        names,
        [
            rf"^{prefix}dce_gse_.*$",
            rf"^{prefix}dce_dsl_.*$",
            rf"^{prefix}dce_.*gse.*$",
            rf"^{prefix}dce_.*dsl.*$",
        ],
    )
    if e_name:
        series = _extract_series(e_name)
        if series is not None:
            out.add_series("e", series)
            out.metadata["e_name"] = e_name
    return out


def _extract_fpi_dataset(probe: str, names: Sequence[str]) -> MMSProbeData:
    out = MMSProbeData(probe=probe)
    species_map = {"dis": "ion", "des": "electron"}
    for species, label in species_map.items():
        prefix = f"mms{probe}_{species}_"
        mappings = {
            "bulkv": [rf"^{prefix}bulkv_gse_.*$", rf"^{prefix}bulkv_.*gse.*$"],
            "numberdensity": [rf"^{prefix}numberdensity_.*$"],
            "temppara": [rf"^{prefix}temppara_.*$", rf"^{prefix}temppara_.*$"],
            "tempperp": [rf"^{prefix}tempperp_.*$", rf"^{prefix}tempperp_.*$"],
            "temptensor": [rf"^{prefix}temptensor_.*$", rf"^{prefix}temptensor_.*$"],
            "energyspectr_omni": [rf"^{prefix}energyspectr_omni_.*$"],
        }
        for key, patterns in mappings.items():
            name = _pick_name(names, patterns)
            if name:
                series = _extract_series(name)
                if series is not None:
                    out.add_series(f"{label}_{key}", series)
    return out


def _interp_vector_series(source: MMSSeries, target_times: np.ndarray) -> np.ndarray | None:
    source_times = np.asarray(source.times)
    source_values = np.asarray(source.values, dtype=float)
    if source_times.size == 0 or source_values.size == 0:
        return None

    source_seconds = np.array([t.timestamp() for t in source_times], dtype=float)
    target_seconds = np.array([t.timestamp() for t in target_times], dtype=float)
    if source_values.ndim == 1:
        return np.interp(target_seconds, source_seconds, source_values)
    if source_values.ndim != 2:
        return None

    out = np.empty((target_seconds.size, source_values.shape[1]), dtype=float)
    for idx in range(source_values.shape[1]):
        out[:, idx] = np.interp(target_seconds, source_seconds, source_values[:, idx])
    return out


def load_mms_mec(
    probes: Sequence[str | int],
    time_range: Sequence[dt.datetime | str],
    data_rate: str = "srvy",
    level: str = "l2",
    time_clip: bool = True,
    no_update: bool = False,
    spdf: bool = False,
) -> dict[str, MMSProbeData]:
    """Load MMS MEC position/velocity data for each probe."""

    trange = _normalize_trange(time_range)
    out: dict[str, MMSProbeData] = {}
    for probe in probes:
        probe_id = _normalize_probe(probe)
        names = _load_names_after_call(
            pyspedas_mms.mms_load_mec,
            trange=trange,
            probe=probe_id,
            data_rate=data_rate,
            level=level,
            time_clip=time_clip,
            no_update=no_update,
            spdf=spdf,
        )
        out[f"mms{probe_id}"] = _extract_mec_dataset(probe_id, names)
    return out


def load_mms_fgm_fpi(
    probes: Sequence[str | int],
    time_range: Sequence[dt.datetime | str],
    mode: str = "srvy",
    level: str = "l2",
    time_clip: bool = True,
    no_update: bool = False,
    spdf: bool = False,
) -> dict[str, MMSProbeData]:
    """Load MMS FGM and FPI products for each probe."""

    trange = _normalize_trange(time_range)
    out: dict[str, MMSProbeData] = {}
    for probe in probes:
        probe_id = _normalize_probe(probe)
        probe_data = MMSProbeData(probe=probe_id)

        fgm_names = _load_names_after_call(
            pyspedas_mms.mms_load_fgm,
            trange=trange,
            probe=probe_id,
            data_rate=mode,
            level=level,
            time_clip=time_clip,
            no_update=no_update,
            spdf=spdf,
        )
        fgm_data = _extract_fgm_dataset(probe_id, fgm_names)
        probe_data.series.update(fgm_data.series)

        try:
            edp_names = _load_names_after_call(
                pyspedas_mms.edp,
                trange=trange,
                probe=probe_id,
                data_rate="fast",
                level=level,
                datatype="dce",
                time_clip=time_clip,
                no_update=no_update,
                spdf=spdf,
            )
        except Exception:
            edp_names = []
        edp_data = _extract_edp_dataset(probe_id, edp_names)
        probe_data.series.update(edp_data.series)

        try:
            fpi_names = _load_names_after_call(
                pyspedas_mms.mms_load_fpi,
                trange=trange,
                probe=probe_id,
                data_rate="fast",
                level=level,
                datatype="dis-moms",
                time_clip=time_clip,
                no_update=no_update,
                spdf=spdf,
            )
        except Exception:
            if mode == "fast":
                raise
            fpi_names = _load_names_after_call(
                pyspedas_mms.mms_load_fpi,
                trange=trange,
                probe=probe_id,
                data_rate=mode,
                level=level,
                datatype="dis-moms",
                time_clip=time_clip,
                no_update=no_update,
                spdf=spdf,
            )
            probe_data.metadata["mode_fpi"] = mode
        else:
            probe_data.metadata["mode_fpi"] = "fast"

        fpi_data = _extract_fpi_dataset(probe_id, fpi_names)
        probe_data.series.update(fpi_data.series)

        if probe_data.has("e") and probe_data.has("b"):
            e_series = probe_data.get("e")
            b_series = probe_data.get("b")
            e_values = np.asarray(e_series.values, dtype=float)
            b_interp = _interp_vector_series(b_series, e_series.times)
            if b_interp is not None and e_values.ndim == 2 and b_interp.ndim == 2:
                e_vec = e_values[:, :3]
                b_vec = b_interp[:, :3]
                b_tesla = b_vec * 1e-9
                e_volts_per_m = e_vec * 1e-3
                with np.errstate(divide="ignore", invalid="ignore"):
                    exb = np.cross(e_volts_per_m, b_tesla) / np.sum(b_tesla ** 2, axis=1, keepdims=True)
                exb_kms = exb / 1000.0
                probe_data.add_series(
                    "exb",
                    MMSSeries(
                        times=e_series.times,
                        values=exb_kms,
                        name=f"mms{probe_id}_edp_exb",
                        units="km/s",
                    ),
                )

        probe_data.metadata["mode_fgm"] = mode
        probe_data.metadata["mode_edp"] = "fast"
        out[f"mms{probe_id}"] = probe_data
    return out


def load_supermag_indices(
    time_range: Sequence[dt.datetime | str],
    uid: str,
    flagstring: str = "all,swiall,imfall",
) -> pd.DataFrame:
    """Load SuperMAG indices from the repo's existing SuperMAG client.

    This is intentionally generic because the AL/SML source is still being
    finalized elsewhere in the manuscript workflow.
    """

    ensure_repo_paths()
    from supermag import SuperMAGGetIndices  # noqa: E402

    start = time_range[0]
    if isinstance(start, dt.datetime):
        start = start.strftime("%Y-%m-%d/%H:%M:%S")
    extent = int(
        (
            (time_range[1] if isinstance(time_range[1], dt.datetime) else pd.to_datetime(time_range[1]))
            - (time_range[0] if isinstance(time_range[0], dt.datetime) else pd.to_datetime(time_range[0]))
        ).total_seconds()
    )
    _, data = SuperMAGGetIndices(uid, start, extent, flagstring)
    if "tval" in data.columns:
        data["tval"] = pd.to_datetime(data["tval"], unit="s", utc=True).dt.tz_convert(None)
    return data


def load_ground_obscuration_series(
    time_range: Sequence[dt.datetime | str],
    glat: float,
    glon: float,
    alt_km: float = 150.0,
    dataset_dir: str | Path = "/home/chakras4/Research/Individual_Studies/eclipse_analysis/database/December2021",
    product: str = "193",
) -> MMSSeries:
    """Load the ground eclipse obscuration series sampled from the local eclipse maps.

    The ``193`` files are the eclipse image product used elsewhere in the codebase.
    ``of`` is the obscuration field; we sample the nearest grid cell to the requested
    ground latitude/longitude at each available time slice.
    """

    import xarray as xr

    start = pd.to_datetime(time_range[0])
    end = pd.to_datetime(time_range[1])
    base_dir = Path(dataset_dir)
    step = dt.timedelta(minutes=5)
    current = start
    times: list[dt.datetime] = []
    values: list[float] = []

    while current <= end:
        file_path = base_dir / f"{current:%Y%m%d%H%M%S}_{int(alt_km)}km_{product}_1.nc"
        if file_path.exists():
            ds = xr.open_dataset(file_path)
            try:
                glat_vals = np.asarray(ds["glat"].values)
                glon_vals = np.asarray(ds["glon"].values)
                lat_idx = int(np.argmin(np.abs(glat_vals - glat)))
                lon_idx = int(np.argmin(np.abs(glon_vals - glon)))
                times.append(current.to_pydatetime() if hasattr(current, "to_pydatetime") else current)
                values.append(float(np.asarray(ds["of"].values)[lat_idx, lon_idx]))
            finally:
                ds.close()
        current = current + step

    return MMSSeries(
        times=np.asarray(times, dtype=object),
        values=np.asarray(values, dtype=float),
        name=f"ground_obscuration_{product}",
        units="fraction",
    )


def load_global_eclipse_window_series(
    dataset_dir: str | Path = "/home/chakras4/Research/Individual_Studies/eclipse_analysis/database/December2021",
    product: str = "193",
    altitude_km: float = 150.0,
    obscuration_threshold: float = 0.0,
) -> MMSSeries:
    """Load the global eclipse window from the local eclipse maps.

    This scans the full set of ``193`` files and finds the first and last map
    where any grid cell exceeds ``obscuration_threshold``. The returned series
    is a two-point window that can be used for plotting and shading.
    """

    import xarray as xr

    base_dir = Path(dataset_dir)
    pattern = f"*_{{}}km_{product}_1.nc".format(int(altitude_km))
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No eclipse maps found in {base_dir} matching {pattern}")

    start_time = None
    end_time = None

    for file_path in files:
        ds = xr.open_dataset(file_path)
        try:
            obscuration = np.asarray(ds["of"].values)
            if np.nanmax(obscuration) > obscuration_threshold:
                stamp = dt.datetime.strptime(file_path.name.split("_")[0], "%Y%m%d%H%M%S")
                if start_time is None:
                    start_time = stamp
                end_time = stamp
        finally:
            ds.close()

    if start_time is None or end_time is None:
        raise ValueError(
            f"No eclipse window found in {base_dir} with product={product} "
            f"and obscuration_threshold={obscuration_threshold}"
        )

    return MMSSeries(
        times=np.asarray([start_time, end_time], dtype=object),
        values=np.asarray([1.0, 1.0], dtype=float),
        name=f"global_eclipse_window_{product}",
        units="binary",
    )


def load_global_occulted_area_series(
    dataset_dir: str | Path = "/home/chakras4/Research/Individual_Studies/eclipse_analysis/database/December2021",
    product: str = "193",
    altitude_km: float = 150.0,
    obscuration_threshold: float = 0.0,
) -> MMSSeries:
    """Integrate the daytime occulted area over the local eclipse maps.

    The returned values are the obscured area on the 150 km shell in units of
    million square kilometers. Only the sunlit portion of the map contributes;
    grid cells with solar zenith angle above 90 degrees are excluded.
    """

    import xarray as xr

    base_dir = Path(dataset_dir)
    files = sorted(base_dir.glob(f"*_{{}}km_{product}_1.nc".format(int(altitude_km))))
    if not files:
        raise FileNotFoundError(f"No eclipse maps found in {base_dir} for product {product}")

    earth_radius_km = 6371.0 + float(altitude_km)
    deg2rad = np.pi / 180.0

    times: list[dt.datetime] = []
    values: list[float] = []

    for file_path in files:
        ds = xr.open_dataset(file_path)
        try:
            obscuration = np.asarray(ds["of"].values, dtype=float)
            sza = np.asarray(ds["sza"].values, dtype=float)
            glat = np.asarray(ds["glat"].values, dtype=float)
            glon = np.asarray(ds["glon"].values, dtype=float)

            day_mask = np.isfinite(obscuration) & np.isfinite(sza) & (sza <= 90.0) & (obscuration > obscuration_threshold)
            if not np.any(day_mask):
                area_km2 = 0.0
            else:
                lat_edges = np.empty(glat.size + 1, dtype=float)
                lat_edges[1:-1] = 0.5 * (glat[:-1] + glat[1:])
                lat_edges[0] = max(-90.0, glat[0] - 0.5)
                lat_edges[-1] = min(90.0, glat[-1] + 0.5)
                lat_width = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
                lon_width = deg2rad * 1.0
                row_area = (earth_radius_km ** 2) * lon_width * lat_width
                area_grid = np.repeat(row_area[:, None], glon.size, axis=1)
                area_km2 = float(np.nansum(obscuration[day_mask] * area_grid[day_mask]))

            stamp = pd.to_datetime(ds["time"].values).to_pydatetime()
            times.append(stamp)
            values.append(area_km2 / 1_000_000.0)
        finally:
            ds.close()

    return MMSSeries(
        times=np.asarray(times, dtype=object),
        values=np.asarray(values, dtype=float),
        name=f"global_occulted_area_{product}",
        units="10^6 km^2",
    )
