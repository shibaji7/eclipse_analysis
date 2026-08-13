"""OMNI solar wind and IMF helpers for the eclipse analysis."""

from .loaders import OmniData, OmniSeries, load_omni_1min_series
from .plots import plot_omni_timeseries
from .workflow import build_omni_figure
