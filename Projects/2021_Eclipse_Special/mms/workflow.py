"""High-level MMS figure workflow."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .loaders import load_global_occulted_area_series, load_mms_fgm_fpi, load_mms_mec
from .plots import MMSSeries, SeriesInput, plot_mms_location_planes, plot_mms_location_schematic, plot_mms_timeseries
from .grl_summary import plot_grl_summary_figure
from goes.loaders import load_goes_r_series, load_lanl_sopa_series
from omni.loaders import load_omni_1min_series


def build_mms_figures(
    probes,
    time_range,
    time_of_interest,
    output_dir,
    eclipse_obscuration=None,
    obscuration_product="193",
    obscuration_threshold=0.0,
    al_sml_index=None,
    mode=None,
    mec_mode=None,
    fgm_mode=None,
    fpi_mode="fast",
):
    """Load MMS data and render the two figures requested in the spec."""

    if mec_mode is None:
        mec_mode = mode or "srvy"
    if fgm_mode is None:
        fgm_mode = mode or "srvy"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mec = load_mms_mec(probes, time_range, data_rate=mec_mode)
    fields = load_mms_fgm_fpi(probes, time_range, mode=fgm_mode)
    for probe_data in fields.values():
        probe_data.metadata["mode_fpi"] = fpi_mode

    if eclipse_obscuration is None:
        eclipse_obscuration = load_global_occulted_area_series(
            product=obscuration_product,
            obscuration_threshold=obscuration_threshold,
        )
    try:
        goes = load_goes_r_series(time_range=time_range)
    except Exception as exc:  # pragma: no cover - depends on runtime network/cache
        goes = {}
        print(f"GOES load failed: {exc}")
    try:
        lanl = load_lanl_sopa_series(time_range=time_range)
    except Exception as exc:  # pragma: no cover - depends on runtime network/cache
        lanl = {}
        print(f"LANL load failed: {exc}")

    location_path = output_dir / "mms_location_schematic.png"
    planes_path = output_dir / "mms_location_planes.png"
    timeseries_path = output_dir / "mms_timeseries.png"

    plot_mms_location_schematic(mec, time_of_interest, location_path)
    plot_mms_location_planes(mec, time_of_interest, planes_path)
    plot_mms_timeseries(
        fields,
        eclipse_obscuration,
        al_sml_index,
        time_range,
        timeseries_path,
        goes_data=goes,
        lanl_data=lanl,
    )
    return {
        "mec": mec,
        "fields": fields,
        "goes": goes,
        "lanl": lanl,
        "location_path": location_path,
        "planes_path": planes_path,
        "timeseries_path": timeseries_path,
    }


def build_grl_summary_figure(
    probes,
    time_range,
    time_of_interest,
    output_dir,
    feature_window=None,
    no_update=False,
    eclipse_obscuration=None,
    obscuration_product="193",
    obscuration_threshold=0.0,
    mode=None,
    mec_mode=None,
    fgm_mode=None,
    fpi_mode="fast",
):
    """Load the core event products and render the consolidated GRL figure."""

    if mec_mode is None:
        mec_mode = mode or "srvy"
    if fgm_mode is None:
        fgm_mode = mode or "srvy"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mec = load_mms_mec(probes, time_range, data_rate=mec_mode, no_update=no_update)
    fields = load_mms_fgm_fpi(probes, time_range, mode=fgm_mode, no_update=no_update)
    for probe_data in fields.values():
        probe_data.metadata["mode_fpi"] = fpi_mode

    if eclipse_obscuration is None:
        eclipse_obscuration = load_global_occulted_area_series(
            product=obscuration_product,
            obscuration_threshold=obscuration_threshold,
        )

    goes = load_goes_r_series(time_range=time_range, no_update=no_update)
    omni = load_omni_1min_series(time_range=time_range, no_update=no_update)

    output_path = output_dir / "grl_consolidated_figure.png"
    plot_grl_summary_figure(
        mec,
        fields,
        goes,
        omni,
        eclipse_obscuration,
        time_range,
        time_of_interest,
        output_path,
        feature_window=feature_window,
    )
    return {
        "mec": mec,
        "fields": fields,
        "goes": goes,
        "omni": omni,
        "eclipse_obscuration": eclipse_obscuration,
        "location_path": output_path,
        "output_path": output_path,
    }