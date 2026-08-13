"""High-level OMNI-only figure workflow."""

from __future__ import annotations

from pathlib import Path

from .loaders import load_omni_1min_series
from .plots import plot_omni_timeseries
from mms.loaders import load_global_occulted_area_series


def build_omni_figure(
    time_range,
    output_dir,
    feature_window=None,
    no_update=False,
    eclipse_obscuration=None,
    obscuration_product="193",
    obscuration_threshold=0.0,
):
    """Load OMNI data and render the aligned solar-wind/IMF figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni = load_omni_1min_series(time_range=time_range, no_update=no_update)
    if eclipse_obscuration is None:
        eclipse_obscuration = load_global_occulted_area_series(
            product=obscuration_product,
            obscuration_threshold=obscuration_threshold,
        )
    output_path = output_dir / "omni_timeseries.png"
    plot_omni_timeseries(
        omni,
        time_range,
        output_path,
        feature_window=feature_window,
        eclipse_obscuration=eclipse_obscuration,
    )
    return {
        "omni": omni,
        "eclipse_obscuration": eclipse_obscuration,
        "output_path": output_path,
    }
