"""High-level GOES-only figure workflow."""

from __future__ import annotations

from pathlib import Path

from .loaders import load_goes_r_series
from .plots import plot_goes_timeseries
from mms.loaders import load_global_occulted_area_series


def build_goes_figure(
    time_range,
    output_dir,
    probe=None,
    time_clip=True,
    no_update=False,
    eclipse_obscuration=None,
    obscuration_product="193",
    obscuration_threshold=0.0,
):
    """Load GOES-R data and render the GOES-only time-series figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    goes = load_goes_r_series(time_range=time_range, probe=probe, time_clip=time_clip, no_update=no_update)
    if eclipse_obscuration is None:
        eclipse_obscuration = load_global_occulted_area_series(
            product=obscuration_product,
            obscuration_threshold=obscuration_threshold,
        )
    output_path = output_dir / "goes_timeseries.png"
    plot_goes_timeseries(goes, time_range, output_path, eclipse_obscuration=eclipse_obscuration)
    return {
        "goes": goes,
        "eclipse_obscuration": eclipse_obscuration,
        "output_path": output_path,
    }
