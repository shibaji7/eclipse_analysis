"""GOES and LANL corroboration loaders for the eclipse analysis."""

from .loaders import (
    CorroborationSatelliteData,
    CorroborationSeries,
    load_goes_r_series,
    load_lanl_sopa_series,
    select_goes_probe,
    select_lanl_probe,
)
from .plots import plot_goes_timeseries
from .workflow import build_goes_figure
